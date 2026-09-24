import asyncio
import json
import re
from pathlib import Path

from evals.fakes import FakeEmbedder, MemoryEchoLLM
from evals.run import expand, is_correct, run_task, summarize

DATA = json.loads((Path(__file__).parent.parent / "evals" / "tasks.json").read_text(encoding="utf-8"))


FAKE_THRESHOLDS = {"min_similarity": 0.3, "reload_threshold": 0.3}  # fake embedder scores lower than bge


def _run(condition, task, budget=6, grace=2):
    return asyncio.run(
        run_task(condition, task, DATA["filler"], MemoryEchoLLM(), FakeEmbedder(), budget, grace, False, FAKE_THRESHOLDS)
    )


def test_filler_never_leaks_an_answer():
    for task in DATA["tasks"]:
        for step in task["script"]:
            for pattern in step.get("expect", []):
                leaks = [f for f in DATA["filler"] if re.search(pattern, f, re.IGNORECASE)]
                assert not leaks, f"{task['id']}: filler {leaks} matches answer pattern {pattern}"


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
