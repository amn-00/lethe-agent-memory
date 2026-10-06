import asyncio

import httpx
import pytest

from lethe.llm import GroqLLM

FAILED_GENERATION = '{"name": "assistant<|channel|>final", "arguments": The gate code is **7791**.}'


def tool_use_failed() -> httpx.Response:
    return httpx.Response(400, json={"error": {
        "message": "Tool choice is none, but model called a tool",
        "type": "invalid_request_error",
        "code": "tool_use_failed",
        "failed_generation": FAILED_GENERATION,
    }})


def ok(text: str) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": text}}], "usage": {"total_tokens": 5}})


def make_llm(responses: list[httpx.Response]) -> tuple[GroqLLM, list]:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return responses[min(len(calls), len(responses)) - 1]

    llm = GroqLLM(model="openai/gpt-oss-20b", api_key="test", retries=2, min_interval=0.0,
                  transport=httpx.MockTransport(handler))
    return llm, calls


def test_tool_use_failed_is_retried():
    llm, calls = make_llm([tool_use_failed(), ok("The gate code is 7791.")])
    assert asyncio.run(llm.chat([{"role": "user", "content": "gate code?"}])) == "The gate code is 7791."
    assert len(calls) == 2
    assert llm.stats == {"tool_use_retries": 1, "fallback_recoveries": 0}


def test_persistent_tool_use_failed_falls_back_to_failed_generation():
    llm, calls = make_llm([tool_use_failed()])
    assert asyncio.run(llm.chat([{"role": "user", "content": "gate code?"}])) == "The gate code is 7791."
    assert len(calls) == 3  # retries=2 -> 3 attempts before falling back
    assert llm.last_usage is None
    # the third failure goes to the fallback, so it's a recovery, not a retry
    assert llm.stats == {"tool_use_retries": 2, "fallback_recoveries": 1}


def test_other_400_fails_immediately():
    llm, calls = make_llm([httpx.Response(400, json={"error": {"message": "bad model", "code": "model_not_found"}})])
    with pytest.raises(RuntimeError, match="rejected the request"):
        asyncio.run(llm.chat([{"role": "user", "content": "hi"}]))
    assert len(calls) == 1
    assert llm.stats == {"tool_use_retries": 0, "fallback_recoveries": 0}
