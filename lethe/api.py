import os

from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from .llm import LLM
from .memory import AgentMemory

SYSTEM = (
    "You are a helpful assistant with long-term memory. "
    "Below are memories from earlier in this conversation. They may be incomplete; "
    "use them only when relevant, and prefer newer information if two memories conflict."
)


class ChatIn(BaseModel):
    message: str


def create_app(memory: AgentMemory, llm: LLM) -> FastAPI:
    app = FastAPI(title="lethe")
    cfg = memory.cfg

    @app.post("/sessions/{session_id}/chat")
    async def chat(session_id: str, body: ChatIn):
        turn = memory.store.next_turn(session_id)
        hits = await run_in_threadpool(memory.recall, session_id, body.message)

        mem_block = "\n".join(f"- {h.memory.text}" for h in hits) or "(none)"
        messages = [
            {"role": "system", "content": f"{SYSTEM}\n\nMemories:\n{mem_block}"},
            *memory.store.recent_messages(session_id, cfg.recent_window),
            {"role": "user", "content": body.message},
        ]
        answer = await llm.chat(messages)

        usage = memory.mark_used(session_id, hits, answer, body.message)
        memory.store.add_message(session_id, "user", body.message)
        memory.store.add_message(session_id, "assistant", answer)
        await run_in_threadpool(memory.add, session_id, body.message)
        evicted = memory.maintain(session_id)

        return {
            "turn": turn,
            "answer": answer,
            "recalled": [
                {"memory_id": h.memory.id, "text": h.memory.text, "similarity": h.similarity, "reloaded": h.reloaded}
                for h in hits
            ],
            "usage": usage,
            "evicted": evicted,
        }

    @app.get("/sessions/{session_id}/memories")
    def memories(session_id: str):
        return memory.snapshot(session_id)

    @app.get("/sessions/{session_id}/ops")
    def ops(session_id: str, limit: int = 200):
        return memory.store.ops(session_id, limit)

    @app.get("/health")
    def health():
        return {"ok": True}

    return app


def build_default_app() -> FastAPI:
    """Run with: uvicorn lethe.api:build_default_app --factory"""
    from .index import FastEmbedder, VectorIndex
    from .llm import GroqLLM
    from .models import PolicyConfig
    from .store import Store

    data = os.getenv("LETHE_DATA", "data")
    os.makedirs(data, exist_ok=True)
    memory = AgentMemory(
        Store(f"{data}/lethe.db"),
        VectorIndex(f"{data}/chroma"),
        FastEmbedder(),
        PolicyConfig(active_budget=5, grace_turns=2),
    )
    return create_app(memory, GroqLLM())
