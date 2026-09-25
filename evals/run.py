"""Benchmark lethe against baselines on scripted multi-turn conversations.

    python -m evals.run                      # real run on the dev set (needs GROQ_API_KEY)
    python -m evals.run --split heldout      # held-out set: run once per version, never tune on it
    python -m evals.run --dry-run            # offline smoke test, fake LLM + embedder
    python -m evals.run --conditions lethe naive_rag --limit 3

Only question turns call the LLM; filler turns get a canned reply so a full run fits
Groq's free tier. Pass --llm-every-turn for full fidelity (much slower, far more calls).
"""

import argparse
import asyncio
import hashlib
import json
import random
import re
import time
import uuid
from collections import defaultdict
from pathlib import Path

from lethe import AgentMemory, PolicyConfig, Store
from lethe.agent import CANNED_REPLY, MemoryAgent, estimate_tokens
from lethe.index import VectorIndex
from lethe.models import ACTIVE

from evals.judge import judge_correct, regex_correct

HERE = Path(__file__).parent
CONDITIONS = ["no_memory", "full_history", "naive_rag", "lethe", "naive_extract", "lethe_extract"]
DEFAULT_CONDITIONS = ["no_memory", "full_history", "naive_rag", "lethe"]
SPLITS = {"dev": "tasks.json", "heldout": "heldout.json"}
BASE_SYSTEM = "You are a helpful assistant. Keep answers short."


class WindowAgent:
    """Baselines with no memory store: last N messages (no_memory) or everything (full_history)."""

    def __init__(self, llm, window: int | None):
        self.llm, self.window, self.history = llm, window, []

    async def chat(self, message: str, respond: bool = True) -> dict:
        context = self.history if self.window is None else self.history[-self.window:]
        messages = [{"role": "system", "content": BASE_SYSTEM}, *context, {"role": "user", "content": message}]
        prompt_tokens = None
        if respond:
            answer = await self.llm.chat(messages)
            usage = getattr(self.llm, "last_usage", None) or {}
            prompt_tokens = usage.get("prompt_tokens") or estimate_tokens(messages)
        else:
            answer = CANNED_REPLY
        self.history += [{"role": "user", "content": message}, {"role": "assistant", "content": answer}]
        return {"answer": answer, "prompt_tokens": prompt_tokens, "recalled": []}


def expand(task: dict, filler_pool: list[str]) -> list[dict]:
    """Turn a script into concrete turns. Filler is seeded by task id, so every condition sees the same conversation."""
    rng = random.Random(task["id"])
    pool, turns = [], []
    for step in task["script"]:
        if "filler" in step:
            for _ in range(step["filler"]):
                if not pool:
                    pool = filler_pool[:]
                    rng.shuffle(pool)
                turns.append({"say": pool.pop()})
        else:
            turns.append(step)
    return turns


def is_correct(answer: str, patterns: list[str]) -> bool:
    return regex_correct(answer, patterns)


async def score_rows(rows: list[dict], judge_llm=None) -> dict:
    """Regex-score every row; if a judge is given, judge every row too and make that the primary score."""
    agree = judged = 0
    for i, r in enumerate(rows):
        r["correct_regex"] = regex_correct(r["answer"], r["expect"])
        r["correct_judge"] = None
        if judge_llm is not None:
            r["correct_judge"] = await judge_correct(judge_llm, r["question"], r["gold"], r["answer"])
            if r["correct_judge"] is not None:
                judged += 1
                agree += r["correct_judge"] == r["correct_regex"]
            if (i + 1) % 20 == 0:
                print(f"  judged {i + 1}/{len(rows)}")
        r["correct"] = r["correct_judge"] if r["correct_judge"] is not None else r["correct_regex"]
    return {"judged": judged, "agreement": round(agree / judged, 3) if judged else None}


def make_agent(
    condition: str, llm, embedder, budget: int, grace: int, overrides: dict | None = None, ordered: bool = True
):
    if condition == "no_memory":
        return WindowAgent(llm, window=4), None
    if condition == "full_history":
        return WindowAgent(llm, window=None), None
    overrides = overrides or {}
    cfg = PolicyConfig(active_budget=budget, grace_turns=grace, **overrides)
    if condition.startswith("naive"):
        cfg = PolicyConfig(active_budget=10**9, **overrides)  # same retrieval, never evicts
    extractor = None
    if condition.endswith("_extract"):
        from lethe.extract import FactExtractor

        extractor = FactExtractor(llm)
    memory = AgentMemory(Store(":memory:"), VectorIndex(None, f"eval-{uuid.uuid4().hex[:10]}"), embedder, cfg)
    return MemoryAgent(memory, llm, ordered=ordered, extractor=extractor), memory


async def run_task(condition, task, filler, llm, embedder, budget, grace, every_turn, overrides=None, ordered=True):
    agent, memory = make_agent(condition, llm, embedder, budget, grace, overrides, ordered)
    sid, results, facts = task["id"], [], []
    for turn in expand(task, filler):
        is_probe = "ask" in turn
        text = turn["ask"] if is_probe else turn["say"]
        respond = is_probe or every_turn
        if memory is not None:
            out = await agent.chat(sid, text, respond=respond)
            facts += out.get("extracted", [])
        else:
            out = await agent.chat(text, respond=respond)
        if is_probe:
            results.append({
                "task": task["id"], "category": task["category"], "condition": condition,
                "question": text, "answer": out["answer"], "gold": turn["gold"], "expect": turn["expect"],
                "correct": is_correct(out["answer"], turn["expect"]),
                "prompt_tokens": out["prompt_tokens"],
                "memories_injected": len(out["recalled"]),
                "reloaded": sum(r["reloaded"] for r in out["recalled"]),
                "recalled_texts": [x["text"] for x in out["recalled"]],
                "active_memories": len(memory.store.list_memories(sid, ACTIVE)) if memory else None,
            })
    ex = getattr(agent, "extractor", None)
    for r in results:  # task-level extraction cost, so accuracy gains can be weighed against it
        r["task_extract_calls"] = ex.calls if ex else 0
        r["task_extract_tokens"] = ex.tokens if ex else 0
        r["task_extract_failures"] = ex.failures if ex else 0
        r["task_facts"] = facts if ex else []
    return results


def summarize(rows: list[dict]) -> dict:
    by_cond = defaultdict(list)
    for r in rows:
        by_cond[r["condition"]].append(r)
    summary = {}
    for cond, rs in by_cond.items():
        cats = defaultdict(list)
        for r in rs:
            cats[r["category"]].append(r["correct"])
        active = [r["active_memories"] for r in rs if r["active_memories"] is not None]
        per_task = {r["task"]: r for r in rs}.values()
        summary[cond] = {
            "accuracy": round(sum(r["correct"] for r in rs) / len(rs), 3),
            "regex_accuracy": round(sum(r.get("correct_regex", r["correct"]) for r in rs) / len(rs), 3),
            "by_category": {c: round(sum(v) / len(v), 3) for c, v in sorted(cats.items())},
            "avg_prompt_tokens": round(sum(r["prompt_tokens"] for r in rs) / len(rs), 1),
            "avg_memories_injected": round(sum(r["memories_injected"] for r in rs) / len(rs), 2),
            "avg_active_memories": round(sum(active) / len(active), 1) if active else None,
            "extract_calls_per_task": round(sum(r.get("task_extract_calls", 0) for r in per_task) / len(per_task), 1),
            "extract_tokens_per_task": round(sum(r.get("task_extract_tokens", 0) for r in per_task) / len(per_task)),
            "extract_failures": sum(r.get("task_extract_failures", 0) for r in per_task),
            "probes": len(rs),
        }
    return summary


def to_markdown(summary: dict, meta: dict) -> str:
    conds = [c for c in CONDITIONS if c in summary]
    cats = sorted({c for s in summary.values() for c in s["by_category"]})
    lines = [
        f"# lethe eval ({meta['split']}): {meta['timestamp']}",
        "",
        f"model `{meta['model']}` · budget {meta['budget']} · grace {meta['grace']} · "
        f"{meta['tasks']} tasks · {meta['probes_per_condition']} questions per condition"
        + (f" · overrides {meta['overrides']}" if meta["overrides"] else "")
        + f" · memories {meta.get('memory_format', 'plain')}"
        + (" · DRY RUN (fake LLM)" if meta["dry_run"] else ""),
        "",
    ]
    judged = meta.get("scoring", {}).get("judged")
    if judged:
        lines += [
            f"Scored by LLM judge `{meta['scoring']['judge_model']}` against reference answers; "
            f"regex scorer shown alongside (agreement {meta['scoring']['agreement']:.0%}).",
            "",
        ]
    lines += [
        "| condition | accuracy | regex accuracy | avg prompt tokens | avg memories injected | avg active memories "
        "| extraction calls / tokens per conversation |",
        "|---|---|---|---|---|---|---|",
    ]
    for c in conds:
        s = summary[c]
        am = "-" if s["avg_active_memories"] is None else s["avg_active_memories"]
        ex = "-" if not s.get("extract_calls_per_task") else f"{s['extract_calls_per_task']} / {s['extract_tokens_per_task']}"
        if s.get("extract_failures"):
            ex += f" ({s['extract_failures']} parse failures)"
        lines.append(
            f"| {c} | {s['accuracy']:.0%} | {s['regex_accuracy']:.0%} | {s['avg_prompt_tokens']} "
            f"| {s['avg_memories_injected']} | {am} | {ex} |"
        )
    lines += ["", "| category | " + " | ".join(conds) + " |", "|---|" + "---|" * len(conds)]
    for cat in cats:
        vals = [f"{summary[c]['by_category'].get(cat, 0):.0%}" for c in conds]
        lines.append(f"| {cat} | " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def checkpoint_path(args, model: str, overrides: dict) -> Path:
    """One checkpoint per exact configuration, so --resume can never mix results from different settings."""
    sig = json.dumps({
        "split": args.split, "model": model, "budget": args.budget, "grace": args.grace, "overrides": overrides,
        "memory_format": args.memory_format, "every_turn": args.llm_every_turn, "dry_run": args.dry_run,
    }, sort_keys=True)
    return HERE / "results" / f"checkpoint-{args.split}-{hashlib.sha1(sig.encode()).hexdigest()[:10]}.jsonl"


async def run_all(conditions, tasks, filler, llm, embedder, args, overrides, ckpt: Path, resume: bool) -> list[dict]:
    """Runs every (condition, task) pair, appending each finished pair to a checkpoint file.
    With resume=True, pairs already in the checkpoint are skipped."""
    done: dict[tuple[str, str], list[dict]] = {}
    if resume and ckpt.exists():
        for line in ckpt.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            done[(rec["condition"], rec["task"])] = rec["rows"]
        print(f"resuming: {len(done)} finished task runs loaded from {ckpt.name}")
    elif ckpt.exists():
        ckpt.unlink()
    ckpt.parent.mkdir(exist_ok=True)

    rows = []
    for cond in conditions:
        for task in tasks:
            if (cond, task["id"]) in done:
                rows += done[(cond, task["id"])]
                continue
            t0 = time.time()
            res = await run_task(
                cond, task, filler, llm, embedder, args.budget, args.grace, args.llm_every_turn, overrides,
                ordered=args.memory_format == "ordered",
            )
            with ckpt.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"condition": cond, "task": task["id"], "rows": res}) + "\n")
            rows += res
            ok = sum(r["correct"] for r in res)
            print(f"[{cond:>12}] {task['id']} {task['category']:<13} {ok}/{len(res)} correct  ({time.time() - t0:.1f}s)")
    return rows


async def main(args):
    data = json.loads((HERE / SPLITS[args.split]).read_text(encoding="utf-8"))
    tasks = data["tasks"][: args.limit] if args.limit else data["tasks"]

    if args.dry_run:
        from evals.fakes import FakeEmbedder, MemoryEchoLLM

        embedder, llm, model = FakeEmbedder(), MemoryEchoLLM(), "fake-echo"
    else:
        from lethe.index import FastEmbedder
        from lethe.llm import GroqLLM

        embedder, llm = FastEmbedder(), GroqLLM()
        model = llm.model

    overrides = {}
    if args.dry_run:  # the fake embedder scores lower than bge; use thresholds calibrated for it
        overrides = {"min_similarity": 0.3, "reload_threshold": 0.3}
    if args.min_similarity is not None:
        overrides["min_similarity"] = args.min_similarity
    if args.reload_threshold is not None:
        overrides["reload_threshold"] = args.reload_threshold

    from lethe.llm import DailyLimitError

    ckpt = checkpoint_path(args, model, overrides)
    try:
        rows = await run_all(args.conditions, tasks, data["filler"], llm, embedder, args, overrides, ckpt, args.resume)
    except DailyLimitError as e:
        saved = len(ckpt.read_text(encoding="utf-8").splitlines()) if ckpt.exists() else 0
        total = len(args.conditions) * len(tasks)
        raise SystemExit(
            f"\nStopped: {e}\nProgress saved ({saved}/{total} task runs). "
            f"After the quota resets, rerun the same command with --resume."
        )

    judge_llm, judge_model = None, None
    if args.judge and not args.dry_run:
        from lethe.llm import GroqLLM

        judge_model = args.judge_model
        judge_llm = GroqLLM(model=judge_model, temperature=0.0)
        print(f"\njudging {len(rows)} answers with {judge_model} ...")
    try:
        scoring = await score_rows(rows, judge_llm)
    except DailyLimitError as e:
        raise SystemExit(
            f"\nStopped while judging: {e}\nAll answers are saved; rerun the same command with --resume "
            f"after the quota resets (answers are reused, only judging is redone)."
        )
    scoring["judge_model"] = judge_model

    summary = summarize(rows)
    meta = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M"), "split": args.split, "model": model, "budget": args.budget, "grace": args.grace,
        "tasks": len(tasks), "probes_per_condition": len(rows) // max(len(args.conditions), 1),
        "dry_run": args.dry_run, "llm_every_turn": args.llm_every_turn, "overrides": overrides,
        "scoring": scoring, "memory_format": args.memory_format,
    }
    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    stamp = f"{args.split}-" + time.strftime("%Y%m%d-%H%M%S") + ("-dry" if args.dry_run else "")
    (out_dir / f"run-{stamp}.json").write_text(json.dumps({"meta": meta, "summary": summary, "rows": rows}, indent=2))
    md = to_markdown(summary, meta)
    (out_dir / f"latest-{args.split}{'-dry' if args.dry_run else ''}.md").write_text(md, encoding="utf-8")
    ckpt.unlink(missing_ok=True)  # finished cleanly; the results file is the record now
    print("\n" + md)


def cli():
    p = argparse.ArgumentParser(description="Benchmark lethe against memory baselines.")
    p.add_argument("--split", default="dev", choices=list(SPLITS), help="dev = tune here; heldout = report here")
    p.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS, choices=CONDITIONS,
                   help="naive_extract / lethe_extract store LLM-extracted facts instead of raw messages")
    p.add_argument("--budget", type=int, default=6)
    p.add_argument("--grace", type=int, default=2)
    p.add_argument("--limit", type=int, default=0, help="only run the first N tasks")
    p.add_argument("--llm-every-turn", action="store_true", help="call the LLM on filler turns too")
    p.add_argument("--dry-run", action="store_true", help="offline: fake LLM + fake embedder")
    p.add_argument("--memory-format", default="ordered", choices=["ordered", "plain"],
                   help="ordered = oldest-first with turn tags; plain = similarity-ranked (the old behaviour)")
    p.add_argument("--resume", action="store_true", help="continue an interrupted run with the same settings")
    p.add_argument("--judge", action="store_true", help="grade answers with an LLM judge (primary score)")
    p.add_argument("--judge-model", default="openai/gpt-oss-120b")
    p.add_argument("--min-similarity", type=float, default=None, help="override recall threshold")
    p.add_argument("--reload-threshold", type=float, default=None, help="override archive reload threshold")
    asyncio.run(main(p.parse_args()))


if __name__ == "__main__":
    cli()
