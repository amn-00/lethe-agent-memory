import os

from fastapi import FastAPI
from pydantic import BaseModel

from .agent import MemoryAgent
from .llm import LLM
from .memory import AgentMemory


class ChatIn(BaseModel):
    message: str


def create_app(memory: AgentMemory, llm: LLM) -> FastAPI:
    app = FastAPI(title="lethe")
    agent = MemoryAgent(memory, llm)

    @app.post("/sessions/{session_id}/chat")
    async def chat(session_id: str, body: ChatIn):
        return await agent.chat(session_id, body.message)

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


def config_from_env():
    from .models import PolicyConfig

    return PolicyConfig(
        active_budget=int(os.getenv("LETHE_BUDGET", "40")),
        grace_turns=int(os.getenv("LETHE_GRACE", "3")),
    )


def build_default_app() -> FastAPI:
    """Run with: uvicorn lethe.api:build_default_app --factory
    Small budget for demos:  $env:LETHE_BUDGET="5"; $env:LETHE_GRACE="2" """
    from .index import FastEmbedder, VectorIndex
    from .llm import GroqLLM
    from .store import Store

    data = os.getenv("LETHE_DATA", "data")
    os.makedirs(data, exist_ok=True)
    memory = AgentMemory(
        Store(f"{data}/lethe.db"), VectorIndex(f"{data}/chroma"), FastEmbedder(), config_from_env()
    )
    return create_app(memory, GroqLLM())
