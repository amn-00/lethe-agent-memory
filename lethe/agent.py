"""One chat turn: recall -> prompt -> LLM -> used-tracking -> store -> evict.
Shared by the API server and the eval harness so both measure the same thing."""

from .llm import LLM
from .memory import AgentMemory

SYSTEM = (
    "You are a helpful assistant with long-term memory. "
    "Below are memories from earlier in this conversation. They may be incomplete; "
    "use them only when relevant, and prefer newer information if two memories conflict. "
    "Keep answers short."
)

CANNED_REPLY = "Got it."


def estimate_tokens(messages: list[dict]) -> int:
    """Rough fallback (~4 chars per token) when the provider doesn't report usage."""
    return sum(len(m["content"]) for m in messages) // 4


class MemoryAgent:
    def __init__(self, memory: AgentMemory, llm: LLM):
        self.memory = memory
        self.llm = llm

    def build_messages(self, session_id: str, message: str, hits) -> list[dict]:
        mem_block = "\n".join(f"- {h.memory.text}" for h in hits) or "(none)"
        return [
            {"role": "system", "content": f"{SYSTEM}\n\nMemories:\n{mem_block}"},
            *self.memory.store.recent_messages(session_id, self.memory.cfg.recent_window),
            {"role": "user", "content": message},
        ]

    async def chat(self, session_id: str, message: str, respond: bool = True) -> dict:
        """respond=False skips the LLM call and replies with a canned acknowledgement.
        The eval uses this for filler turns to stay inside free-tier rate limits."""
        mem = self.memory
        turn = mem.store.next_turn(session_id)
        hits = mem.recall(session_id, message)
        messages = self.build_messages(session_id, message, hits)

        prompt_tokens = None
        if respond:
            answer = await self.llm.chat(messages)
            usage = getattr(self.llm, "last_usage", None) or {}
            prompt_tokens = usage.get("prompt_tokens") or estimate_tokens(messages)
        else:
            answer = CANNED_REPLY

        usage_report = mem.mark_used(session_id, hits, answer, message)
        mem.store.add_message(session_id, "user", message)
        mem.store.add_message(session_id, "assistant", answer)
        mem.add(session_id, message)
        evicted = mem.maintain(session_id)

        return {
            "turn": turn,
            "answer": answer,
            "prompt_tokens": prompt_tokens,
            "recalled": [
                {"memory_id": h.memory.id, "text": h.memory.text, "similarity": h.similarity, "reloaded": h.reloaded}
                for h in hits
            ],
            "usage": usage_report,
            "evicted": evicted,
        }
