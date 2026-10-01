import json
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .agent import MemoryAgent
from .llm import LLM
from .memory import AgentMemory


class ChatIn(BaseModel):
    message: str


STATIC = Path(__file__).parent / "static"
RESULTS = Path(os.getenv("LETHE_RESULTS", Path(__file__).resolve().parents[1] / "evals" / "results"))


def latest_eval(split: str, results_dir: Path = RESULTS) -> dict:
    """Newest result per condition across saved runs of this split (ordered-memory runs only, no dry runs),
    so runs that covered different conditions can be shown together."""
    conditions, meta = {}, None
    for f in sorted(results_dir.glob(f"run-{split}-*.json")):
        if "dry" in f.name:
            continue
        run = json.loads(f.read_text(encoding="utf-8"))
        if run["meta"].get("memory_format") != "ordered":
            continue
        for cond, summary in run["summary"].items():
            conditions[cond] = {**summary, "run": f.name, "judged": bool(run["meta"].get("scoring", {}).get("judged"))}
        meta = run["meta"]
    return {"split": split, "conditions": conditions, "meta": meta}


def create_app(memory: AgentMemory, llm: LLM, extractor=None) -> FastAPI:
    app = FastAPI(title="lethe")
    agent = MemoryAgent(memory, llm, extractor=extractor)

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/config")
    def config():
        cfg = memory.cfg
        return {
            "budget": cfg.active_budget, "grace": cfg.grace_turns, "extract": extractor is not None,
            "model": getattr(llm, "model", "unknown"),
        }

    @app.get("/eval/{split}")
    def eval_results(split: str):
        return latest_eval(split)

    @app.post("/sessions/{session_id}/chat")
    async def chat(session_id: str, body: ChatIn):
        out = await agent.chat(session_id, body.message)
        out["pending"] = len(agent.pending.get(session_id, []))  # messages waiting to be turned into facts
        return out

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
    Small budget for demos:  $env:LETHE_BUDGET="5"; $env:LETHE_GRACE="2"
    Fact extraction is on by default; $env:LETHE_EXTRACT="0" stores raw messages instead."""
    from .index import FastEmbedder, VectorIndex
    from .llm import GroqLLM
    from .store import Store

    data = os.getenv("LETHE_DATA", "data")
    os.makedirs(data, exist_ok=True)
    memory = AgentMemory(
        Store(f"{data}/lethe.db"), VectorIndex(f"{data}/chroma"), FastEmbedder(), config_from_env()
    )
    llm = GroqLLM()
    extractor = None
    if os.getenv("LETHE_EXTRACT", "1") == "1":
        from .extract import FactExtractor

        extractor = FactExtractor(llm)
    return create_app(memory, llm, extractor)
