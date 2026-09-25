import asyncio
import json
import re
from pathlib import Path

from evals.fakes import FakeEmbedder, MemoryEchoLLM
from evals.run import expand, is_correct, run_task, summarize

import pytest

EVALS = Path(__file__).parent.parent / "evals"
DATA = json.loads((EVALS / "tasks.json").read_text(encoding="utf-8"))
HELDOUT = json.loads((EVALS / "heldout.json").read_text(encoding="utf-8"))


FAKE_THRESHOLDS = {"min_similarity": 0.3, "reload_threshold": 0.3}  # fake embedder scores lower than bge


def _run_on(data, condition, task, budget=6, grace=2):
    return asyncio.run(
        run_task(condition, task, data["filler"], MemoryEchoLLM(), FakeEmbedder(), budget, grace, False, FAKE_THRESHOLDS)
    )


def _run(condition, task, budget=6, grace=2):
    return _run_on(DATA, condition, task, budget, grace)


@pytest.mark.parametrize("data", [DATA, HELDOUT], ids=["dev", "heldout"])
def test_filler_never_leaks_an_answer(data):
    for task in data["tasks"]:
        for step in task["script"]:
            for pattern in step.get("expect", []):
                leaks = [f for f in data["filler"] if re.search(pattern, f, re.IGNORECASE)]
                assert not leaks, f"{task['id']}: filler {leaks} matches answer pattern {pattern}"


def test_heldout_is_actually_fresh():
    """No filler line or fact sentence shared with the dev set."""
    assert not set(DATA["filler"]) & set(HELDOUT["filler"])
    dev_says = {s["say"] for t in DATA["tasks"] for s in t["script"] if "say" in s}
    held_says = {s["say"] for t in HELDOUT["tasks"] for s in t["script"] if "say" in s}
    assert not dev_says & held_says


@pytest.mark.parametrize("data", [DATA, HELDOUT], ids=["dev", "heldout"])
def test_full_history_can_answer_everything(data):
    """Sanity check on the task set itself: with the whole conversation visible, every
    expected answer must be present, otherwise the task (not the system) is broken."""
    for task in data["tasks"]:
        for r in _run_on(data, "full_history", task):
            assert r["correct"], f"{task['id']}: '{r['question']}' unanswerable even with full history"


def test_expand_is_deterministic_and_counts_filler():
    task = DATA["tasks"][0]
    a, b = expand(task, DATA["filler"]), expand(task, DATA["filler"])
    assert a == b
    assert len(a) == 2 + task["script"][1]["filler"]


def test_scoring_uses_word_boundaries():
    assert is_correct("Your manager is Priya.", [r"\bpriya\b"])
    assert not is_correct("Priyanka trains you at the gym.", [r"\bpriya\b"])


def test_baselines_bracket_the_problem():
    task = DATA["tasks"][0]  # fact at turn 1, 12 filler turns, then the question
    assert all(r["correct"] for r in _run("full_history", task))
    assert not any(r["correct"] for r in _run("no_memory", task))


def test_lethe_stays_within_budget_and_still_recalls():
    task = DATA["tasks"][10]  # long gap: 26 filler turns
    rows = _run("lethe", task, budget=6, grace=2)
    assert rows[0]["active_memories"] <= 6 + 2  # budget plus grace-window slack
    assert rows[0]["correct"]


def test_summary_shape():
    rows = _run("lethe", DATA["tasks"][4]) + _run("naive_rag", DATA["tasks"][4])
    s = summarize(rows)
    assert set(s) == {"lethe", "naive_rag"}
    assert {"accuracy", "by_category", "avg_prompt_tokens", "avg_active_memories"} <= s["lethe"].keys()


def test_normalize_handles_unicode_hyphens_and_spaces():
    """Regression from the held-out run: gpt-oss wrote P2\u2011117 and the regex missed it."""
    from evals.judge import regex_correct
    assert regex_correct("You rent spot **P2\u2011117**.", [r"p\s*-?\s*2\s*-?\s*117"])
    assert regex_correct("You're using VS\u202fCode.", [r"vs code"])


def test_judge_verdict_parsing():
    from evals.judge import parse_verdict
    assert parse_verdict("CORRECT") is True
    assert parse_verdict("incorrect.") is False  # 'INCORRECT' contains 'CORRECT'; must not pass
    assert parse_verdict("hmm") is None


def test_judge_catches_what_regex_cannot():
    """A reversed answer contains the expected name, so regex passes it; the judge must see the reference."""
    from evals.judge import judge_correct, regex_correct

    class StrictJudge:
        async def chat(self, messages):
            ans = messages[0]["content"].split("Assistant's answer: ")[1].split("\n")[0]
            return "INCORRECT" if ans.startswith("Before Farah") else "CORRECT"

    answer = "Before Farah, your team lead was Omkar."
    assert regex_correct(answer, [r"\bfarah\b"])  # the false pass we saw on the held-out set
    assert asyncio.run(judge_correct(StrictJudge(), "who was my lead before the current one?", "farah", answer)) is False


def test_every_question_has_a_gold_answer():
    for data in (DATA, HELDOUT):
        for task in data["tasks"]:
            for step in task["script"]:
                if "ask" in step:
                    assert step.get("gold"), f"{task['id']}: '{step['ask']}' has no reference answer"


def test_extract_conditions_run_and_report_cost():
    from evals.run import summarize
    rows = _run("lethe_extract", DATA["tasks"][4])  # fake LLM can't produce JSON -> raw fallback path
    assert rows and all("recalled_texts" in r for r in rows)
    s = summarize(rows)["lethe_extract"]
    assert s["extract_calls_per_task"] > 0 and s["extract_failures"] > 0


def test_interrupted_run_resumes_without_redoing_finished_tasks(tmp_path):
    """Regression: a daily-quota 429 mid-run used to throw away every finished task."""
    import argparse

    from evals.run import run_all
    from lethe.llm import DailyLimitError

    class QuotaLLM(MemoryEchoLLM):
        def __init__(self, fail_after):
            self.calls, self.fail_after = 0, fail_after

        async def chat(self, messages):
            self.calls += 1
            if self.fail_after is not None and self.calls > self.fail_after:
                raise DailyLimitError("tokens per day exhausted")
            return await super().chat(messages)

    args = argparse.Namespace(budget=6, grace=2, llm_every_turn=False, memory_format="ordered")
    tasks, ckpt = DATA["tasks"][:4], tmp_path / "ckpt.jsonl"
    try:
        asyncio.run(run_all(["lethe"], tasks, DATA["filler"], QuotaLLM(2), FakeEmbedder(), args, FAKE_THRESHOLDS, ckpt, False))
        raise AssertionError("should have stopped")
    except DailyLimitError:
        pass
    assert len(ckpt.read_text().splitlines()) == 2  # two single-question tasks finished before the quota hit

    llm = QuotaLLM(None)
    rows = asyncio.run(run_all(["lethe"], tasks, DATA["filler"], llm, FakeEmbedder(), args, FAKE_THRESHOLDS, ckpt, True))
    assert llm.calls == 2  # only the two unfinished tasks were run again
    assert [r["task"] for r in rows] == ["t01", "t02", "t03", "t04"]


def test_daily_limit_is_not_retried():
    import httpx

    from lethe.llm import DailyLimitError, GroqLLM

    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(429, text='{"error":{"message":"Rate limit reached ... tokens per day (TPD): Limit 200000"}}')

    llm = GroqLLM(api_key="x", min_interval=0)
    transport = httpx.MockTransport(handler)
    orig = httpx.AsyncClient
    httpx.AsyncClient = lambda **kw: orig(transport=transport, **kw)
    try:
        asyncio.run(llm.chat([{"role": "user", "content": "hi"}]))
        raise AssertionError("should have raised")
    except DailyLimitError:
        pass
    finally:
        httpx.AsyncClient = orig
    assert len(calls) == 1  # failed fast instead of sleeping through retries
