import asyncio
import json
import uuid

from lethe import AgentMemory, PolicyConfig, Store
from lethe.agent import MemoryAgent
from lethe.extract import FactExtractor, parse_facts
from lethe.index import VectorIndex

from tests.test_memory import FakeEmbedder


class ScriptedLLM:
    """Plays an extractor for any prompt containing 'extract durable'; otherwise answers 'ok'."""

    def __init__(self, facts_by_keyword: dict[str, str]):
        self.facts_by_keyword = facts_by_keyword
        self.extract_calls = 0

    async def chat(self, messages):
        prompt = messages[-1]["content"]
        if "extract durable personal facts" not in prompt:
            return "ok"
        self.extract_calls += 1
        facts = []
        for line in prompt.splitlines():
            if line.startswith("[t"):
                turn, text = line[2:].split("] ", 1)
                for kw, fact in self.facts_by_keyword.items():
                    if kw in text:
                        facts.append({"turn": int(turn), "fact": fact})
        return json.dumps({"facts": facts})


def make(llm, batch_size=8, **cfg):
    mem = AgentMemory(
        Store(":memory:"), VectorIndex(None, f"x-{uuid.uuid4().hex[:8]}"), FakeEmbedder(),
        PolicyConfig(min_similarity=0.3, reload_threshold=0.3, context_similarity=0.1, **cfg),
    )
    return MemoryAgent(mem, llm, extractor=FactExtractor(llm, batch_size=batch_size)), mem


def test_parse_facts_is_lenient_but_safe():
    assert parse_facts('{"facts": [{"turn": 3, "fact": "My sister lives in Pune."}]}', 9) == [(3, "My sister lives in Pune.")]
    fenced = '```json\n{"facts": [{"fact": "I ride a bike."}]}\n```'
    assert parse_facts(fenced, 9) == [(9, "I ride a bike.")]  # missing turn -> default
    assert parse_facts('{"facts": []}', 1) == []
    assert parse_facts("sorry, I can't do that", 1) is None
    assert parse_facts('{"facts": "oops"}', 1) is None


def test_chatter_is_dropped_and_facts_keep_their_source_turn():
    llm = ScriptedLLM({"three body": "I'm currently reading The Three-Body Problem."})
    agent, mem = make(llm)
    for msg in ["lol ok", "started a new book, the three body problem", "my tea got cold"]:
        asyncio.run(agent.chat("s", msg, respond=False))
    assert mem.store.list_memories("s") == []  # nothing stored until a flush
    out = asyncio.run(agent.chat("s", "what am i reading?"))  # answering forces a flush
    stored = mem.store.list_memories("s")
    assert [m.text for m in stored] == ["I'm currently reading The Three-Body Problem."]
    assert stored[0].created_turn == 2  # turn of the message it came from, not the flush turn
    assert out["extracted"] and out["recalled"] and "Three-Body" in out["recalled"][0]["text"]


def test_batching_limits_extraction_calls():
    llm = ScriptedLLM({})
    agent, _ = make(llm, batch_size=4)
    for i in range(12):
        asyncio.run(agent.chat("s", f"filler message {i}", respond=False))
    assert llm.extract_calls == 3  # 12 messages / batch of 4, no per-message calls


def test_unparseable_extraction_falls_back_to_raw_messages():
    class BrokenLLM:
        async def chat(self, messages):
            return "not json at all"

    agent, mem = make(BrokenLLM(), batch_size=2)
    asyncio.run(agent.chat("s", "my dog is called bruno", respond=False))
    asyncio.run(agent.chat("s", "lol ok", respond=False))
    assert {m.text for m in mem.store.list_memories("s")} == {"my dog is called bruno", "lol ok"}
    assert agent.extractor.failures == 1


def test_search_has_no_side_effects():
    llm = ScriptedLLM({})
    agent, mem = make(llm)
    m = mem.add("s", "my dog is called bruno")
    before = mem.store.get(m.id)
    assert [x.id for x in mem.search("s", "dog bruno", min_similarity=0.3)] == [m.id]
    after = mem.store.get(m.id)
    assert (after.retrieved_count, after.last_access_turn, after.tier) == (before.retrieved_count, before.last_access_turn, before.tier)
    assert not any(o["op"] == "retrieve" for o in mem.store.ops("s"))


def test_extractor_sees_related_known_facts_across_batches():
    """Dev t16: 'started a new book' landed in a later batch than 'now reading project hail mary',
    so the extractor couldn't tell it was an update. It must now see the related known fact."""
    seen_prompts = []

    class RecordingLLM(ScriptedLLM):
        async def chat(self, messages):
            seen_prompts.append(messages[-1]["content"])
            return await super().chat(messages)

    llm = RecordingLLM({"hail mary": "I'm now reading Project Hail Mary book.", "new book": "I'm now reading The Three Body Problem book."})
    agent, mem = make(llm, batch_size=1)
    asyncio.run(agent.chat("s", "finished dune, now reading project hail mary book", respond=False))
    asyncio.run(agent.chat("s", "started a new book, the three body problem", respond=False))
    assert "Facts already in memory" in seen_prompts[-1]
    assert "Project Hail Mary" in seen_prompts[-1]
