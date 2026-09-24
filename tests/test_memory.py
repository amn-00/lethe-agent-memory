import hashlib
import math
import uuid

import pytest
from fastapi.testclient import TestClient

from lethe import AgentMemory, PolicyConfig, Store
from lethe.api import create_app
from lethe.index import VectorIndex
from lethe.models import ACTIVE, ARCHIVE
from lethe.policy import tokens


class FakeEmbedder:
    """Hashed bag-of-words: texts sharing words get high cosine similarity. Offline + deterministic."""

    def embed(self, texts):
        out = []
        for t in texts:
            v = [0.0] * 256
            for w in tokens(t):
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 256] += 1.0
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        return out


class EchoLLM:
    """Answers by repeating the first memory line, so 'used' tracking is predictable."""

    async def chat(self, messages):
        lines = [l[2:] for l in messages[0]["content"].splitlines() if l.startswith("- ")]
        return lines[0] if lines else "I don't know."


def make_memory(**cfg) -> AgentMemory:
    cfg.setdefault("min_similarity", 0.3)  # the fake bag-of-words embedder scores lower than bge
    return AgentMemory(
        Store(":memory:"),
        VectorIndex(path=None, collection=f"test-{uuid.uuid4().hex[:8]}"),
        FakeEmbedder(),
        PolicyConfig(**cfg),
    )


def test_recall_finds_relevant_memory():
    mem = make_memory()
    mem.store.next_turn("s")
    mem.add("s", "my dog is called bruno and loves football")
    mem.add("s", "i work as a backend engineer in noida")
    hits = mem.recall("s", "what is my dog called")
    assert hits and "bruno" in hits[0].memory.text


def test_eviction_respects_budget_and_grace():
    mem = make_memory(active_budget=3, grace_turns=2)
    for i in range(6):
        mem.store.next_turn("s")
        mem.add("s", f"fact number {i} about topic{i}")
        mem.maintain("s")
    active = mem.store.list_memories("s", ACTIVE)
    archived = mem.store.list_memories("s", ARCHIVE)
    assert len(active) == 3 and len(archived) == 3
    assert {m.text for m in archived} == {f"fact number {i} about topic{i}" for i in range(3)}  # oldest go first


def test_used_memories_outscore_ignored_ones():
    mem = make_memory(active_budget=1, grace_turns=0)
    mem.store.next_turn("s")
    used = mem.add("s", "favourite club is barcelona")
    ignored = mem.add("s", "favourite club rivalry chatter nonsense")
    for _ in range(3):
        mem.store.next_turn("s")
        hits = mem.recall("s", "favourite club")
        mem.mark_used("s", hits, "Your favourite club is Barcelona.", "favourite club")
    evicted = mem.maintain("s")
    assert [e["memory_id"] for e in evicted] == [ignored.id]
    assert mem.store.get(used.id).tier == ACTIVE


def test_reload_pulls_evicted_memory_back():
    mem = make_memory(active_budget=1, grace_turns=0, reload_threshold=0.6)
    mem.store.next_turn("s")
    old = mem.add("s", "passport renewal appointment tuesday")
    mem.store.next_turn("s")
    mem.add("s", "grocery list eggs milk")
    mem.maintain("s")
    assert mem.store.get(old.id).tier == ARCHIVE

    mem.store.next_turn("s")
    hits = mem.recall("s", "when is my passport renewal appointment")
    assert hits[0].memory.id == old.id and hits[0].reloaded
    assert mem.store.get(old.id).tier == ACTIVE
    assert any(o["op"] == "reload" for o in mem.store.ops("s"))


def test_sessions_are_isolated():
    mem = make_memory()
    mem.store.next_turn("a")
    mem.add("a", "secret code is pineapple")
    mem.store.next_turn("b")
    assert mem.recall("b", "secret code pineapple") == []


def test_chat_endpoint_end_to_end():
    mem = make_memory()
    client = TestClient(create_app(mem, EchoLLM()))
    client.post("/sessions/u1/chat", json={"message": "my sister lives in pune"})
    r = client.post("/sessions/u1/chat", json={"message": "where does my sister live"}).json()
    assert r["turn"] == 2
    assert "pune" in r["answer"]
    assert r["usage"][0]["used"] is True
    snap = client.get("/sessions/u1/memories").json()
    assert len(snap) == 2 and {"score", "used_rate", "tier"} <= snap[0].keys()
    assert client.get("/sessions/u1/ops").json()


def test_query_echo_does_not_count_as_used():
    from lethe.policy import answer_overlap
    assert answer_overlap("favourite club rivalry chatter", "your favourite club is barcelona", "favourite club") == 0.0
    assert answer_overlap("favourite club is barcelona", "your favourite club is barcelona", "favourite club") == 1.0


def test_retrieval_alone_does_not_keep_memory_alive():
    """Noise that keeps getting retrieved but never used must still age out."""
    mem = make_memory(active_budget=1, grace_turns=2)
    mem.store.next_turn("s")
    noise = mem.add("s", "football chatter random stuff")
    mem.store.next_turn("s")
    fact = mem.add("s", "football match sunday morning")
    for _ in range(3):
        mem.store.next_turn("s")
        hits = mem.recall("s", "football")
        mem.mark_used("s", hits, "Your match is on sunday morning.", "football")
    evicted = mem.maintain("s")
    assert [e["memory_id"] for e in evicted] == [noise.id]
    assert mem.store.get(fact.id).tier == "active"


def test_ignored_retrieval_lowers_score():
    """Regression: a retrieved-but-ignored memory must score below a never-retrieved one (live-run bug)."""
    from lethe.models import MemoryRecord
    from lethe.policy import retention_score
    cfg = PolicyConfig()
    fact = MemoryRecord("a", "s", "my sister lives in pune", "active", 1, 1)
    noise = MemoryRecord("b", "s", "the weather is nice today", "active", 3, 3, retrieved_count=1, used_count=0)
    assert retention_score(noise, 6, cfg)["score"] < retention_score(fact, 6, cfg)["score"]
