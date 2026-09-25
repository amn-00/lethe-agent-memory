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
CONDITIONS = ["no_memory", "full_history", "naive_rag", "lethe"]
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


def make_agent(condition: str, llm, embedder, budget: int, grace: int, overrides: dict | None = None):
    if condition == "no_memory":
        return WindowAgent(llm, window=4), None
    if condition == "full_history":
        return WindowAgent(llm, window=None), None
    overrides = overrides or {}
    cfg = PolicyConfig(active_budget=budget, grace_turns=grace, **overrides)
    if condition == "naive_rag":
        cfg = PolicyConfig(active_budget=10**9, **overrides)  # same retrieval, never evicts
    memory = AgentMemory(Store(":memory:"), VectorIndex(None, f"eval-{uuid.uuid4().hex[:10]}"), embedder, cfg)
    return MemoryAgent(memory, llm), memory


async def run_task(condition, task, filler, llm, embedder, budget, grace, every_turn, overrides=None):
    agent, memory = make_agent(condition, llm, embedder, budget, grace, overrides)
    sid, results = task["id"], []
    for turn in expand(task, filler):
        is_probe = "ask" in turn
        text = turn["ask"] if is_probe else turn["say"]
        respond = is_probe or every_turn
        if memory is not None:
            out = await agent.chat(sid, text, respond=respond)
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
                "active_memories": len(memory.store.list_memories(sid, ACTIVE)) if memory else None,
            })
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
        summary[cond] = {
            "accuracy": round(sum(r["correct"] for r in rs) / len(rs), 3),
            "regex_accuracy": round(sum(r.get("correct_regex", r["correct"]) for r in rs) / len(rs), 3),
            "by_category": {c: round(sum(v) / len(v), 3) for c, v in sorted(cats.items())},
            "avg_prompt_tokens": round(sum(r["prompt_tokens"] for r in rs) / len(rs), 1),
            "avg_memories_injected": round(sum(r["memories_injected"] for r in rs) / len(rs), 2),
            "avg_active_memories": round(sum(active) / len(active), 1) if active else None,
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
        "| condition | accuracy | regex accuracy | avg prompt tokens | avg memories injected | avg active memories |",
        "|---|---|---|---|---|---|",
    ]
    for c in conds:
        s = summary[c]
        am = "-" if s["avg_active_memories"] is None else s["avg_active_memories"]
        lines.append(
            f"| {c} | {s['accuracy']:.0%} | {s['regex_accuracy']:.0%} | {s['avg_prompt_tokens']} "
            f"| {s['avg_memories_injected']} | {am} |"
        )
    lines += ["", "| category | " + " | ".join(conds) + " |", "|---|" + "---|" * len(conds)]
    for cat in cats:
        vals = [f"{summary[c]['by_category'].get(cat, 0):.0%}" for c in conds]
        lines.append(f"| {cat} | " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


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

    rows = []
    for cond in args.conditions:
        for task in tasks:
            t0 = time.time()
            res = await run_task(
                cond, task, data["filler"], llm, embedder, args.budget, args.grace, args.llm_every_turn, overrides
            )
            rows += res
            ok = sum(r["correct"] for r in res)
            print(f"[{cond:>12}] {task['id']} {task['category']:<13} {ok}/{len(res)} correct  ({time.time() - t0:.1f}s)")

    judge_llm, judge_model = None, None
    if args.judge and not args.dry_run:
        from lethe.llm import GroqLLM

        judge_model = args.judge_model
        judge_llm = GroqLLM(model=judge_model, temperature=0.0)
        print(f"\njudging {len(rows)} answers with {judge_model} ...")
    scoring = await score_rows(rows, judge_llm)
    scoring["judge_model"] = judge_model

    summary = summarize(rows)
    meta = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M"), "split": args.split, "model": model, "budget": args.budget, "grace": args.grace,
        "tasks": len(tasks), "probes_per_condition": len(rows) // max(len(args.conditions), 1),
        "dry_run": args.dry_run, "llm_every_turn": args.llm_every_turn, "overrides": overrides,
        "scoring": scoring,
    }
    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    stamp = f"{args.split}-" + time.strftime("%Y%m%d-%H%M%S") + ("-dry" if args.dry_run else "")
    (out_dir / f"run-{stamp}.json").write_text(json.dumps({"meta": meta, "summary": summary, "rows": rows}, indent=2))
    md = to_markdown(summary, meta)
    (out_dir / f"latest-{args.split}{'-dry' if args.dry_run else ''}.md").write_text(md, encoding="utf-8")
    print("\n" + md)


def cli():
    p = argparse.ArgumentParser(description="Benchmark lethe against memory baselines.")
    p.add_argument("--split", default="dev", choices=list(SPLITS), help="dev = tune here; heldout = report here")
    p.add_argument("--conditions", nargs="+", default=CONDITIONS, choices=CONDITIONS)
    p.add_argument("--budget", type=int, default=6)
    p.add_argument("--grace", type=int, default=2)
    p.add_argument("--limit", type=int, default=0, help="only run the first N tasks")
    p.add_argument("--llm-every-turn", action="store_true", help="call the LLM on filler turns too")
    p.add_argument("--dry-run", action="store_true", help="offline: fake LLM + fake embedder")
    p.add_argument("--judge", action="store_true", help="grade answers with an LLM judge (primary score)")
    p.add_argument("--judge-model", default="openai/gpt-oss-120b")
    p.add_argument("--min-similarity", type=float, default=None, help="override recall threshold")
    p.add_argument("--reload-threshold", type=float, default=None, help="override archive reload threshold")
    asyncio.run(main(p.parse_args()))


if __name__ == "__main__":
    cli()
