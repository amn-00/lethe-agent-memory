import asyncio
import os
import time
from typing import Protocol

import httpx


class LLM(Protocol):
    async def chat(self, messages: list[dict]) -> str: ...


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
    ):
        self.model = model or os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
        self.api_key = api_key or os.environ["GROQ_API_KEY"]
        self.timeout, self.retries = timeout, retries
        self.min_interval = min_interval if min_interval is not None else float(os.getenv("GROQ_MIN_INTERVAL", "2.2"))
        self.reasoning_effort = os.getenv("GROQ_REASONING_EFFORT", "low")
        self.last_usage: dict | None = None
        self._last_call = 0.0
        self._lock = asyncio.Lock()

    def _payload(self, messages: list[dict]) -> dict:
        payload = {"model": self.model, "messages": messages, "temperature": 0.2}
        if self.model.startswith("openai/gpt-oss"):
            # reasoning tokens count toward rate limits; keep them small and out of the answer
            payload["reasoning_effort"] = self.reasoning_effort
            payload["include_reasoning"] = False
        return payload

    async def chat(self, messages: list[dict]) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        async with self._lock, httpx.AsyncClient(timeout=self.timeout) as client:
            last_err = None
            for attempt in range(self.retries + 1):
                wait = self.min_interval - (time.monotonic() - self._last_call)
                if wait > 0:
                    await asyncio.sleep(wait)
                self._last_call = time.monotonic()
                try:
                    r = await client.post(self.URL, json=self._payload(messages), headers=headers)
                    if r.status_code == 429:  # rate limited: honour retry-after, then try again
                        delay = float(r.headers.get("retry-after", 2 ** attempt * 5))
                        await asyncio.sleep(min(delay, 90))
                        last_err = RuntimeError(f"429 rate limited: {r.text[:200]}")
                        continue
                    if 400 <= r.status_code < 500:  # our request is wrong; retrying won't help
                        raise RuntimeError(f"Groq rejected the request ({r.status_code}): {r.text[:300]}")
                    r.raise_for_status()
                    data = r.json()
                    self.last_usage = data.get("usage")
                    return data["choices"][0]["message"]["content"] or ""
                except (httpx.HTTPStatusError, httpx.TransportError) as e:
                    last_err = e
                    await asyncio.sleep(2 ** attempt)
        raise RuntimeError(f"Groq call failed after {self.retries + 1} attempts: {last_err}")
