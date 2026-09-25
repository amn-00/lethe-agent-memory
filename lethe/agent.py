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

ORDER_NOTE = (
    "Memories are listed oldest first, each tagged with the turn it was said (current turn: {turn}). "
    "If memories conflict, the latest one is the current truth; earlier ones describe the past."
)

CANNED_REPLY = "Got it."


def format_memories(hits, current_turn: int, ordered: bool = True) -> tuple[str, str]:
    """Returns (memory block, extra system note).
    ordered=True sorts by when each memory was said and tags it with its turn, so the model
    can tell 'i moved to indore' (turn 7) from 'now i'm in hyderabad' (turn 13).
    Unordered similarity-ranked lists made RAG get update chains backwards on the held-out set."""
    if not hits:
        return "(none)", ""
    if not ordered:
        return "\n".join(f"- {h.memory.text}" for h in hits), ""
    hits = sorted(hits, key=lambda h: h.memory.created_turn)
    block = "\n".join(f"- [turn {h.memory.created_turn}] {h.memory.text}" for h in hits)
    return block, ORDER_NOTE.format(turn=current_turn)


def estimate_tokens(messages: list[dict]) -> int:
    """Rough fallback (~4 chars per token) when the provider doesn't report usage."""
    return sum(len(m["content"]) for m in messages) // 4


class MemoryAgent:
    def __init__(self, memory: AgentMemory, llm: LLM, ordered: bool = True, extractor=None):
        """extractor: optional FactExtractor. Without it every user message is stored as a memory;
        with it, messages are queued and only extracted facts are stored."""
        self.memory = memory
        self.llm = llm
        self.ordered = ordered
        self.extractor = extractor
        self.pending: dict[str, list[tuple[int, str]]] = {}

    async def flush(self, session_id: str) -> list[str]:
        """Extract facts from queued messages and store them. Returns the facts added."""
        batch = self.pending.pop(session_id, [])
        if not batch or self.extractor is None:
            return []
        facts = await self.extractor.extract(batch)
        for turn, fact in facts:
            self.memory.add(session_id, fact, turn=turn)
        self.memory.store.log(
            session_id, self.memory.store.current_turn(session_id), "extract",
            detail={"messages": len(batch), "facts": [f for _, f in facts]},
        )
        return [f for _, f in facts]

    def build_messages(self, session_id: str, message: str, hits) -> list[dict]:
        turn = self.memory.store.current_turn(session_id)
        mem_block, note = format_memories(hits, turn, self.ordered)
        system = f"{SYSTEM} {note}".strip()
        return [
            {"role": "system", "content": f"{system}\n\nMemories:\n{mem_block}"},
            *self.memory.store.recent_messages(session_id, self.memory.cfg.recent_window),
            {"role": "user", "content": message},
        ]

    async def chat(self, session_id: str, message: str, respond: bool = True) -> dict:
        """respond=False skips the LLM call and replies with a canned acknowledgement.
        The eval uses this for filler turns to stay inside free-tier rate limits."""
        mem = self.memory
        turn = mem.store.next_turn(session_id)
        extracted = []
        if self.extractor is not None and respond:
            extracted = await self.flush(session_id)  # facts from earlier turns must be searchable before answering
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
        if self.extractor is None:
            mem.add(session_id, message)
        else:
            self.pending.setdefault(session_id, []).append((turn, message))
            if len(self.pending[session_id]) >= self.extractor.batch_size:
                extracted += await self.flush(session_id)
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
            "extracted": extracted,
            "evicted": evicted,
        }
