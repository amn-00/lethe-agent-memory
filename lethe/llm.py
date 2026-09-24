import os
from typing import Protocol

import httpx


class LLM(Protocol):
    async def chat(self, messages: list[dict]) -> str: ...


class GroqLLM:
    URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, model: str | None = None, api_key: str | None = None, timeout: float = 30.0, retries: int = 2):
        self.model = model or os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
        self.api_key = api_key or os.environ["GROQ_API_KEY"]
        self.timeout, self.retries = timeout, retries

    async def chat(self, messages: list[dict]) -> str:
        payload = {"model": self.model, "messages": messages, "temperature": 0.2}
        headers = {"Authorization": f"Bearer {self.api_key}"}
        last_err = None
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for _ in range(self.retries + 1):
                try:
                    r = await client.post(self.URL, json=payload, headers=headers)
                    r.raise_for_status()
                    return r.json()["choices"][0]["message"]["content"]
                except (httpx.HTTPStatusError, httpx.TransportError) as e:
                    last_err = e
        raise RuntimeError(f"Groq call failed after {self.retries + 1} attempts: {last_err}")
