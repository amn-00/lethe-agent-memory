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
