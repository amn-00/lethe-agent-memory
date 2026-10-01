import json
import os
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .agent import MemoryAgent
from .llm import LLM
from .memory import AgentMemory


class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=500)


class DemoLimits:
    """Guardrails for a public demo running on one API key: caps messages per chat and per day.
    0 disables a cap. In-memory, so it resets on restart, which is fine for a demo."""

    def __init__(self, per_session: int = 0, per_day: int = 0):
        self.per_session, self.per_day = per_session, per_day
        self.sessions: dict[str, int] = {}
        self.day, self.today = time.strftime("%Y-%m-%d"), 0

    def check(self, session_id: str):
        if time.strftime("%Y-%m-%d") != self.day:
            self.day, self.today = time.strftime("%Y-%m-%d"), 0
        if self.per_day and self.today >= self.per_day:
            raise HTTPException(429, "The demo has reached its daily message limit. Try again tomorrow, or run lethe locally.")
        if self.per_session and self.sessions.get(session_id, 0) >= self.per_session:
            raise HTTPException(429, f"This chat has reached its {self.per_session}-message limit. Start a new chat to keep going.")
        self.sessions[session_id] = self.sessions.get(session_id, 0) + 1
        self.today += 1


STATIC = Path(__file__).parent / "static"


def results_dir() -> Path:
    """LETHE_RESULTS if set, else evals/results next to the package (editable install), else under the working dir."""
    if os.getenv("LETHE_RESULTS"):
        return Path(os.environ["LETHE_RESULTS"])
    beside = Path(__file__).resolve().parents[1] / "evals" / "results"
    return beside if beside.exists() else Path.cwd() / "evals" / "results"


RESULTS = results_dir()


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


def create_app(memory: AgentMemory, llm: LLM, extractor=None, limits: DemoLimits | None = None) -> FastAPI:
    app = FastAPI(title="lethe")
    agent = MemoryAgent(memory, llm, extractor=extractor)
    limits = limits or DemoLimits()

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/config")
    def config():
        cfg = memory.cfg
        return {
            "budget": cfg.active_budget, "grace": cfg.grace_turns, "extract": extractor is not None,
            "model": getattr(llm, "model", "unknown"), "messages_per_chat": limits.per_session or None,
        }

    @app.get("/eval/{split}")
    def eval_results(split: str):
        return latest_eval(split)

    @app.post("/sessions/{session_id}/chat")
    async def chat(session_id: str, body: ChatIn):
        limits.check(session_id)
        try:
            out = await agent.chat(session_id, body.message)
        except Exception as e:
            if type(e).__name__ == "DailyLimitError":
                raise HTTPException(429, "The model's daily quota is used up. Try again tomorrow.")
            raise HTTPException(502, "The model didn't respond. Try again in a moment.")
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
    limits = DemoLimits(
        per_session=int(os.getenv("LETHE_DEMO_PER_CHAT", "0")), per_day=int(os.getenv("LETHE_DEMO_PER_DAY", "0"))
    )
    return create_app(memory, llm, extractor, limits)
