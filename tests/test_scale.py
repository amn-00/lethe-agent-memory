import asyncio
import json
from pathlib import Path

from evals.fakes import FakeEmbedder, MemoryEchoLLM
from evals.run import WindowAgent, expand, make_agent
from evals.scale import CHECKPOINTS, calibrate, project, run_all_scale, summarize_scale, tokens, trace_task
from lethe.agent import CANNED_REPLY

SCALE = json.loads((Path(__file__).parent.parent / "evals" / "scale.json").read_text(encoding="utf-8"))
FAKE_THRESHOLDS = {"min_similarity": 0.3, "reload_threshold": 0.3}  # fake embedder scores lower than bge
DATA = {"filler": SCALE["filler"], "replies": SCALE["replies"]}
TASK = {"id": "x1", "category": "long_chat", "script": [
    {"say": "my piano teacher is mr fernandes"}, {"filler": 30},
    {"ask": "who is my piano teacher?", "expect": ["fernandes"], "gold": "mr fernandes"},
]}
SETUPS = ["no_memory", "full_history", "naive_rag", "lethe_extract"]


def _trace(cond):
    return asyncio.run(trace_task(cond, TASK, DATA, MemoryEchoLLM(), FakeEmbedder(), 6, 2, FAKE_THRESHOLDS))


def test_scale_set_layout():
    """Two 400-message chats with a question at exactly messages 100, 200, 300 and 400."""
    assert len(SCALE["tasks"]) == 2
    for task in SCALE["tasks"]:
        turns = expand(task, SCALE["filler"])
        assert len(turns) == 400
        assert [i for i, t in enumerate(turns, start=1) if "ask" in t] == CHECKPOINTS
    assert len(SCALE["replies"]) >= 50 and len(set(SCALE["replies"])) == len(SCALE["replies"])


def test_scripted_reply_replaces_the_canned_one_and_default_is_unchanged():
    window = WindowAgent(MemoryEchoLLM(), window=None)
    out = asyncio.run(window.chat("hi", respond=False, reply="Sounds lovely."))
    assert out["answer"] == "Sounds lovely." and window.history[-1]["content"] == "Sounds lovely."
    assert out["prompt_tokens_reported"] is None and out["prompt_tokens_est"] > 0

    agent, memory = make_agent("lethe", MemoryEchoLLM(), FakeEmbedder(), 6, 2, FAKE_THRESHOLDS)
    out = asyncio.run(agent.chat("s", "lol", respond=False, reply="Ha, fair enough."))
    assert out["answer"] == "Ha, fair enough." and memory.store.recent_messages("s", 2)[-1]["content"] == "Ha, fair enough."
    assert asyncio.run(agent.chat("s", "ok", respond=False))["answer"] == CANNED_REPLY


def test_trace_has_one_entry_per_message_for_every_setup():
    for cond in SETUPS:
        rows, trace = _trace(cond)
        assert [s["turn"] for s in trace] == list(range(1, 33))
        assert all(s["est"] > 0 and s["messages"] >= 2 for s in trace)
        assert len(rows) == 1 and rows[0]["turn"] == 32


def test_full_history_grows_every_turn_while_lethe_stays_flat():
    _, full = _trace("full_history")
    assert all(b["est"] > a["est"] for a, b in zip(full, full[1:], strict=False))
    assert full[-1]["messages"] == 1 + 2 * 31 + 1  # system, 31 earlier exchanges, the question
    rows, lethe = _trace("lethe_extract")
    assert max(s["messages"] for s in lethe) == 6  # system, 4 recent messages, the new one
    assert lethe[-1]["est"] < full[-1]["est"]
    assert rows[0]["correct"]  # the fact still reached the prompt (echo LLM)


def test_traces_are_deterministic_and_size_live_extraction():
    assert _trace("full_history")[1] == _trace("full_history")[1]  # same filler and scripted replies every time
    _, lethe = _trace("lethe_extract")
    assert all(s["live_extract_est"] > 400 for s in lethe)  # the extraction prompt template alone is ~455
    assert lethe[-1]["extract_calls"] >= 4  # every 8 messages, plus the flush before answering


def test_prompt_refused_as_too_large_is_recorded_not_scored():
    """Groq's free tier refuses a prompt over 8,000 tokens per minute; full history reaches that by message 400.
    The question must be marked not sent and left out of accuracy, and the run must carry on."""
    from lethe.llm import RequestTooLargeError

    class SmallLimitLLM(MemoryEchoLLM):
        async def chat(self, messages):
            if len(messages) > 20:
                raise RequestTooLargeError("request too large")
            return await super().chat(messages)

    rows, trace = asyncio.run(trace_task("full_history", TASK, DATA, SmallLimitLLM(), FakeEmbedder(), 6, 2, FAKE_THRESHOLDS))
    assert len(trace) == 32 and trace[-1]["question"] and trace[-1]["est"] > trace[-2]["est"]  # still sized
    assert rows[0]["sent"] is False and rows[0]["correct"] is False
    s = summarize_scale({"full_history": {"x1": {"rows": rows, "trace": trace}}})["full_history"]
    assert s["not_sent"] == 1 and s["questions"] == 0 and s["by_turn"]["32"] == [None]


def test_report_prints_on_a_windows_console():
    """Regression: the first real run saved its results, then crashed printing ✓/✗ to a cp1252 console."""
    from evals.scale import console_safe

    assert console_safe("✓ ✗ ≈ ok · 7–8", "cp1252") == "? ? ? ok · 7–8"
    assert console_safe("✓ ok", "utf-8") == "✓ ok"


def test_calibration_fits_per_message_overhead():
    trace = [{"est": e, "messages": m, "reported": e + 5 * m + 60} for e, m in [(100, 6), (2000, 200), (4000, 400)]]
    cal = calibrate([trace])
    assert cal["source"] == "this run" and cal["per_message"] == 5 and cal["fixed"] == 60 and cal["max_residual"] == 0
    assert tokens({"est": 100, "messages": 10}, cal) == 210
    assert calibrate([[{"est": 5, "messages": 2, "reported": None}]])["source"].startswith("default")


def test_projection_adds_overhead_and_assumed_completions():
    cal = {"per_message": 5, "fixed": 60}
    trace = [{"question": False, "est": 10, "messages": 3},
             {"question": True, "est": 100, "messages": 6, "extract_calls": 2, "extract_tokens": 1000}]
    rows = [{"question": "q?", "gold": "g", "answer": "a"}]
    p = project({"lethe_extract": {"x": {"rows": rows, "trace": trace}}}, completion=50, cal=cal)
    assert p["answers"] == (100 + 5 * 6 + 60) + 50
    assert p["extraction"] == 1000 + 2 * (5 + 60 + 50)
    assert p["answering_model"] == p["answers"] + p["extraction"] and p["judge_model"] > 0


def test_summary_and_resume(tmp_path):
    ckpt = tmp_path / "ckpt.jsonl"
    first = asyncio.run(run_all_scale(["full_history", "lethe_extract"], [TASK], DATA, MemoryEchoLLM(), FakeEmbedder(),
                                      6, 2, FAKE_THRESHOLDS, ckpt, False))
    assert len(ckpt.read_text().splitlines()) == 2

    calls = []

    class CountingLLM(MemoryEchoLLM):
        async def chat(self, messages):
            calls.append(1)
            return await super().chat(messages)

    again = asyncio.run(run_all_scale(["full_history", "lethe_extract"], [TASK], DATA, CountingLLM(), FakeEmbedder(),
                                      6, 2, FAKE_THRESHOLDS, ckpt, True))
    assert not calls and again == first  # everything came back from the checkpoint

    s = summarize_scale(first)
    assert s["full_history"]["prompt_tokens_whole_chat"] == sum(s["full_history"]["prompt_tokens_curve"])
    assert s["lethe_extract"]["largest_prompt"] < s["full_history"]["largest_prompt"]
    assert s["lethe_extract"]["extract_calls"] >= 4 and s["lethe_extract"]["live_extract_est"] > 0
