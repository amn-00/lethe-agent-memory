"""Re-grade an existing results file without regenerating any answers.

    python -m evals.rescore --split heldout            # regex only (fixed normalisation), free
    python -m evals.rescore --split heldout --judge    # + LLM judge, one call per answer
"""

import argparse
import asyncio
import glob
import json
import time
from pathlib import Path

from evals.run import HERE, SPLITS, score_rows, summarize, to_markdown


def latest_run(split: str) -> Path:
    files = [f for f in sorted(glob.glob(str(HERE / "results" / f"run-{split}-*.json"))) if "dry" not in f]
    if not files:
        raise SystemExit(f"no results found for split '{split}' in evals/results/")
    return Path(files[-1])


async def main(args):
    path = Path(args.file) if args.file else latest_run(args.split)
    run = json.loads(path.read_text(encoding="utf-8"))
    tasks = json.loads((HERE / SPLITS[run["meta"].get("split", args.split)]).read_text(encoding="utf-8"))
    ref = {(t["id"], s["ask"]): s for t in tasks["tasks"] for s in t["script"] if "ask" in s}
    for r in run["rows"]:  # older runs didn't store gold/expect
        r.setdefault("gold", ref[(r["task"], r["question"])]["gold"])
        r.setdefault("expect", ref[(r["task"], r["question"])]["expect"])

    judge_llm = None
    if args.judge:
        from lethe.llm import GroqLLM

        judge_llm = GroqLLM(model=args.judge_model, temperature=0.0)
        print(f"judging {len(run['rows'])} answers from {path.name} with {args.judge_model} ...")
    scoring = await score_rows(run["rows"], judge_llm)
    scoring["judge_model"] = args.judge_model if args.judge else None

    meta = {**run["meta"], "scoring": scoring, "rescored_at": time.strftime("%Y-%m-%d %H:%M")}
    summary = summarize(run["rows"])
    out = path.with_name(path.stem + "-rescored.json")
    out.write_text(json.dumps({"meta": meta, "summary": summary, "rows": run["rows"]}, indent=2))
    md = to_markdown(summary, meta)
    (HERE / "results" / f"latest-{meta['split']}.md").write_text(md, encoding="utf-8")

    flips = [r for r in run["rows"] if r["correct_judge"] is not None and r["correct_judge"] != r["correct_regex"]]
    if flips:
        print(f"\n{len(flips)} answers where judge and regex disagree:")
        for r in flips:
            print(f"  [{r['condition']}] {r['task']} regex={r['correct_regex']} judge={r['correct_judge']} | "
                  f"{r['question']} -> {ascii(r['answer'][:110])}")
    print("\n" + md)


def cli():
    p = argparse.ArgumentParser(description="Re-grade a saved eval run.")
    p.add_argument("--split", default="heldout", choices=list(SPLITS))
    p.add_argument("--file", default=None, help="specific results json (default: latest for the split)")
    p.add_argument("--judge", action="store_true")
    p.add_argument("--judge-model", default="openai/gpt-oss-120b")
    asyncio.run(main(p.parse_args()))


if __name__ == "__main__":
    cli()
