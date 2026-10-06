import asyncio
import os
import re
import time
from typing import Protocol

import httpx

try:  # pick up GROQ_API_KEY etc. from a .env file in the working directory, if present
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


class DailyLimitError(RuntimeError):
    """The provider's per-day quota is exhausted. Retrying won't help until it resets."""


class LLM(Protocol):
    async def chat(self, messages: list[dict]) -> str: ...


def _tool_use_failure(r: httpx.Response) -> str | None:
    """If this is gpt-oss's spurious "model called a tool" 400, return its failed_generation, else None."""
    if r.status_code != 400:
        return None
    try:
        err = r.json().get("error") or {}
    except ValueError:
        return None
    if err.get("code") != "tool_use_failed":
        return None
    return err.get("failed_generation") or ""


def _answer_from_failed_generation(text: str) -> str:
    """Pull the answer out of e.g. '{"name": "assistant<|channel|>final", "arguments": The code is **7791**.}'.
    Not valid JSON (the value is often unquoted), so take everything after "arguments": and peel off the
    closing brace, surrounding quotes and markdown bold."""
    m = re.search(r'"arguments"\s*:\s*(.*?)\s*\}?\s*$', text, re.DOTALL)
    if not m:
        return ""
    answer = m.group(1).strip()
    if len(answer) >= 2 and answer[0] == answer[-1] == '"':
        answer = answer[1:-1]
    return answer.replace("**", "").strip()


class GroqLLM:
    """Groq's OpenAI-compatible endpoint, paced for the free tier (30 req/min)."""

    URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float = 60.0,
        retries: int = 4,
        min_interval: float | None = None,
        temperature: float = 0.2,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.model = model or os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
        self.api_key = api_key or os.environ["GROQ_API_KEY"]
        self.timeout, self.retries = timeout, retries
        self.min_interval = min_interval if min_interval is not None else float(os.getenv("GROQ_MIN_INTERVAL", "2.2"))
        self.reasoning_effort = os.getenv("GROQ_REASONING_EFFORT", "low")
        self.temperature = temperature
        self.transport = transport  # tests inject httpx.MockTransport here
        self.last_usage: dict | None = None
        # how often the gpt-oss tool_use_failed glitch hit us, so eval reports can say so
        self.stats = {"tool_use_retries": 0, "fallback_recoveries": 0}
        self._last_call = 0.0
        self._lock = asyncio.Lock()

    def _payload(self, messages: list[dict]) -> dict:
        payload = {"model": self.model, "messages": messages, "temperature": self.temperature}
        if self.model.startswith("openai/gpt-oss"):
            # reasoning tokens count toward rate limits; keep them small and out of the answer
            payload["reasoning_effort"] = self.reasoning_effort
            payload["include_reasoning"] = False
        return payload

    async def chat(self, messages: list[dict]) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        async with self._lock, httpx.AsyncClient(timeout=self.timeout, transport=self.transport) as client:
            last_err, failed_generation = None, None
            for attempt in range(self.retries + 1):
                wait = self.min_interval - (time.monotonic() - self._last_call)
                if wait > 0:
                    await asyncio.sleep(wait)
                self._last_call = time.monotonic()
                try:
                    r = await client.post(self.URL, json=self._payload(messages), headers=headers)
                    if r.status_code == 429 and re.search(r"per day|\bTPD\b|\bRPD\b", r.text):
                        raise DailyLimitError(f"daily quota exhausted for {self.model}: {r.text[:240]}")
                    if r.status_code == 429:  # per-minute limit: honour retry-after, then try again
                        delay = float(r.headers.get("retry-after", 2 ** attempt * 5))
                        await asyncio.sleep(min(delay, 90))
                        last_err = RuntimeError(f"429 rate limited: {r.text[:200]}")
                        failed_generation = None
                        continue
                    gen = _tool_use_failure(r)
                    if gen is not None:  # gpt-oss sampling glitch, not a bad request: just ask again
                        last_err = RuntimeError(f"Groq tool_use_failed (400): {r.text[:300]}")
                        failed_generation = gen
                        if attempt < self.retries:  # the final failure isn't a retry; it goes to the fallback
                            self.stats["tool_use_retries"] += 1
                        continue
                    if 400 <= r.status_code < 500:  # our request is wrong; retrying won't help
                        raise RuntimeError(f"Groq rejected the request ({r.status_code}): {r.text[:300]}")
                    r.raise_for_status()
                    data = r.json()
                    self.last_usage = data.get("usage")
                    return data["choices"][0]["message"]["content"] or ""
                except (httpx.HTTPStatusError, httpx.TransportError) as e:
                    last_err, failed_generation = e, None
                    await asyncio.sleep(2 ** attempt)
        if failed_generation is not None:  # the last attempt still hit the glitch; the answer is usually in there
            answer = _answer_from_failed_generation(failed_generation)
            if answer:
                self.last_usage = None  # Groq reports no usage for a rejected call; don't reuse the previous one
                self.stats["fallback_recoveries"] += 1
                return answer
        raise RuntimeError(f"Groq call failed after {self.retries + 1} attempts: {last_err}")
