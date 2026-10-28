"""Scale test: does lethe keep prompts small as a chat grows, without losing the facts?

    python -m evals.scale --preview                  # offline: local embedder + stand-in LLM, projected Groq tokens
    python -m evals.scale --preview --fake-embedder  # fully offline, no model download
    python -m evals.scale --judge                    # real run on Groq (costs quota: run --preview first)
    python -m evals.scale --judge --resume           # continue after a daily-quota stop

Two chats of 400 messages (evals/scale.json), four questions each, at messages 100, 200, 300 and 400.
It's a test set like heldout2: never tune on it.

Measured vs computed:
- accuracy: real answers on the question turns only, graded like evals/run.py (regex, plus the judge with --judge).
- prompt size on every message, for every setup: the exact prompt that setup would send, counted offline
  (chars / 4, agent.estimate_tokens), then calibrated per setup against Groq's reported prompt_tokens on the
  turns that did call it.
- extraction (lethe): real calls, batched every 8 messages as in the other evals. Plus a computed size of what it
  would cost in a live chat, where the agent extracts before every answer, i.e. once per message.
Filler turns get short scripted replies (scale.json "replies") instead of "Got it.", so full history grows like a
real chat. One run, 8 questions per setup: accuracy is a sanity check here, not a comparison.
"""

import argparse
import asyncio
import hashlib
import json
import random
import sys
import time
from pathlib import Path

from evals.judge import JUDGE_PROMPT
from evals.run import (
    CONDITIONS as ALL_CONDITIONS,
)
from evals.run import (
    HERE,
    LLM_STAT_KEYS,
    expand,
    is_correct,
    llm_stats_snapshot,
    make_agent,
    new_llm_stats,
    score_rows,
)
from lethe.agent import estimate_tokens
from lethe.llm import RequestTooLargeError
from lethe.models import ACTIVE, ARCHIVE

DATA = HERE / "scale.json"
CONDITIONS = ["no_memory", "full_history", "naive_rag", "lethe_extract"]
CHECKPOINTS = [100, 200, 300, 400]
NAMES = {"no_memory": "No memory", "full_history": "Full history", "naive_rag": "Search everything",
         "lethe": "lethe (raw messages)", "naive_extract": "Search everything + facts", "lethe_extract": "lethe"}


class PreviewLLM:
    """Stand-in for --preview. Short answers of realistic length: an echo, like the fake in evals/fakes.py,
    would feed its own huge answers back into full history and inflate the projection. Its output isn't
    JSON, so extraction falls back to storing raw messages; that can only overestimate memory prompts."""

    last_usage = None

    async def chat(self, messages):
        return "I'm not sure, could you remind me?"


async def trace_task(condition, task, data, llm, embedder, budget, grace, overrides=None):
    """Runs one chat for one setup. Returns (rows for the question turns, one trace entry per message)."""
    agent, memory = make_agent(condition, llm, embedder, budget, grace, overrides, ordered=True)
    sid, rows, trace = task["id"], [], []
    replies = random.Random(f"{task['id']}-replies")  # same scripted replies for every setup
    ex = getattr(agent, "extractor", None)
    for n, turn in enumerate(expand(task, data["filler"]), start=1):
        is_probe = "ask" in turn
        text = turn["ask"] if is_probe else turn["say"]
        reply = None if is_probe else replies.choice(data["replies"])
        sent = is_probe
        if memory is not None:
            out = await agent.chat(sid, text, respond=is_probe, reply=reply)
        else:
            try:
                out = await agent.chat(text, respond=is_probe, reply=reply)
            except RequestTooLargeError:
                # full history outgrew what the provider accepts in one request (Groq free tier: 8,000 tokens per
                # minute). Nothing was sent and nothing was stored, so size the same prompt offline and go on.
                out, sent = await agent.chat(text, respond=False), False
        step = {"turn": n, "question": is_probe, "est": out["prompt_tokens_est"], "messages": out["prompt_messages"],
                "reported": out["prompt_tokens_reported"]}
        if memory is not None:
            step["active"] = len(memory.store.list_memories(sid, ACTIVE))
            step["archived"] = len(memory.store.list_memories(sid, ARCHIVE))
        if ex is not None:
            step["extract_calls"], step["extract_tokens"] = ex.calls, ex.tokens
            # in a live chat the agent flushes before every answer, so each message is extracted on its own
            live = ex.build_prompt([(n, text)], agent.extraction_context(sid, [(n, text)]))
            step["live_extract_est"] = estimate_tokens([{"content": live}])
        trace.append(step)
        if is_probe:
            rows.append({
                "task": task["id"], "category": task["category"], "condition": condition, "turn": n,
                "question": text, "answer": out["answer"], "gold": turn["gold"], "expect": turn["expect"],
                "sent": sent,  # False: the provider refused the prompt as too large; not scored
                "correct": sent and is_correct(out["answer"], turn["expect"]),
                "prompt_tokens": out["prompt_tokens"],
                "memories_injected": len(out["recalled"]),
                "reloaded": sum(r["reloaded"] for r in out["recalled"]),
                "recalled_texts": [x["text"] for x in out["recalled"]],
                "active_memories": step.get("active"),
            })
    return rows, trace


# Groq's prompt_tokens ≈ chars/4 estimate + PER_MESSAGE × messages + FIXED: the chat format adds a few tokens
# around every message, plus a fixed preamble. Fitted on 234 answer calls of the saved heldout2 run (full_history
# and no_memory, prompts rebuilt offline); worst miss 12.6 tokens. It measures gpt-oss's prompt format only;
# nothing in lethe is tuned on it. Used when a run has no reported counts of its own (the preview).
DEFAULT_OVERHEAD = {"per_message": 5.37, "fixed": 59.4}


def calibrate(traces: list[list[dict]]) -> dict:
    """Fits reported - estimate = per_message × messages + fixed over every turn that called the provider (all setups
    pooled, so prompts of very different message counts pin down both numbers). Falls back to DEFAULT_OVERHEAD
    when there's nothing to fit (preview) or the message counts don't vary."""
    pts = [(s["messages"], s["reported"] - s["est"]) for t in traces for s in t if s.get("reported")]
    n = len(pts)
    sx, sy = sum(m for m, _ in pts), sum(d for _, d in pts)
    var = n * sum(m * m for m, _ in pts) - sx * sx
    if n < 3 or var == 0:
        return {**DEFAULT_OVERHEAD, "calls": n, "max_residual": None, "source": "default (heldout2 fit)"}
    a = (n * sum(m * d for m, d in pts) - sx * sy) / var
    c = (sy - a * sx) / n
    worst = max(abs(d - (a * m + c)) for m, d in pts)
    return {"per_message": round(a, 2), "fixed": round(c, 1), "calls": n, "max_residual": round(worst, 1), "source": "this run"}


def tokens(step: dict, cal: dict) -> int:
    """Calibrated prompt size of one traced turn."""
    return round(step["est"] + cal["per_message"] * step["messages"] + cal["fixed"])


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 1) if xs else None


def summarize_scale(results: dict, cal: dict | None = None) -> dict:
    """results: {condition: {task_id: {"rows": [...], "trace": [...]}}}. Averages over chats.
    cal: from calibrate() over every setup's traces; computed here if not given."""
    cal = cal or calibrate([r["trace"] for by_task in results.values() for r in by_task.values()])
    summary = {}
    for cond, by_task in results.items():
        traces = [r["trace"] for r in by_task.values()]
        rows = [x for r in by_task.values() for x in r["rows"]]
        scored = [r for r in rows if r.get("sent", True)]
        length = min(len(t) for t in traces)
        curve = [round(sum(tokens(t[i], cal) for t in traces) / len(traces)) for i in range(length)]
        s = {
            "correct": sum(r["correct"] for r in scored),
            "questions": len(scored),
            "not_sent": len(rows) - len(scored),
            "accuracy": round(sum(r["correct"] for r in scored) / len(scored), 3) if scored else None,
            # None marks a question the provider refused as too large
            "by_turn": {str(t): [r["correct"] if r.get("sent", True) else None for r in rows if r["turn"] == t]
                        for t in sorted({r["turn"] for r in rows})},
            "prompt_tokens_at": {str(c): curve[c - 1] for c in CHECKPOINTS if c <= length},
            "prompt_tokens_curve": curve,
            # prompts only, as if every message got a reply; completions aren't counted
            "prompt_tokens_whole_chat": sum(curve),
            "largest_prompt": max(tokens(x, cal) for t in traces for x in t),
        }
        if "active" in traces[0][0]:
            s["active_at"] = {str(c): _mean(t[c - 1]["active"] for t in traces) for c in CHECKPOINTS if c <= length}
            s["archived_at_end"] = _mean(t[-1]["archived"] for t in traces)
            s["reloads_at_questions"] = sum(r["reloaded"] for r in rows)
        if "extract_calls" in traces[0][-1]:
            s["extract_calls"] = _mean(t[-1]["extract_calls"] for t in traces)
            s["extract_tokens"] = _mean(t[-1]["extract_tokens"] for t in traces)
            # each live extraction prompt is one user message, so it gets the same calibration
            s["live_extract_est"] = _mean(
                sum(tokens({"est": x["live_extract_est"], "messages": 1}, cal) for x in t) for t in traces
            )
        summary[cond] = s
    return summary


def project(results: dict, completion: int, cal: dict = DEFAULT_OVERHEAD) -> dict:
    """Tokens a real run would spend, from a preview's traces: calibrated prompt sizes plus an assumed completion
    size per call (gpt-oss counts its hidden reasoning too). Judge tokens go to a different model's quota."""
    answers = extraction = judge = 0
    one_message = cal["per_message"] + cal["fixed"]  # extraction and judge prompts are a single user message
    for by_task in results.values():
        for r in by_task.values():
            questions = [s for s in r["trace"] if s["question"]]
            answers += sum(tokens(s, cal) for s in questions) + completion * len(questions)
            last = r["trace"][-1]
            if "extract_calls" in last:  # the preview's extract_tokens are chars/4 estimates of prompt + reply
                extraction += round(last["extract_tokens"] + (one_message + completion) * last["extract_calls"])
            for x in r["rows"]:
                prompt = JUDGE_PROMPT.format(question=x["question"], gold=x["gold"], answer=x["answer"])
                judge += round(estimate_tokens([{"content": prompt}]) + one_message + completion)
    return {"answers": answers, "extraction": extraction, "answering_model": answers + extraction, "judge_model": judge}


def preview_report(results: dict, completion: int, daily_limit: int, per_minute: int = 8000) -> str:
    p, summary = project(results, completion), summarize_scale(results, DEFAULT_OVERHEAD)
    margin = round(p["answering_model"] * 1.25)
    lines = [
        "# scale test preview (no API calls)",
        "",
        f"Projected tokens for one real run, assuming {completion} completion tokens per call:",
        f"- answering model: {p['answering_model']:,} ({p['answers']:,} answers + {p['extraction']:,} extraction); "
        f"with a 25% margin {margin:,}, against a daily limit of {daily_limit:,}"
        + (" -> fits in one day" if margin <= daily_limit else " -> does NOT fit in one day"),
        f"- judge model (separate quota): {p['judge_model']:,}",
        "",
        "| setup | est. prompt tokens per answer at msg 100 | 200 | 300 | 400 | largest single prompt | extraction calls per chat |",
        "|---|---|---|---|---|---|---|",
    ]
    for cond, s in summary.items():
        at = [str(s["prompt_tokens_at"].get(str(c), "-")) for c in CHECKPOINTS]
        lines.append(f"| {NAMES.get(cond, cond)} | " + " | ".join(at) + f" | {s['largest_prompt']:,} | {s.get('extract_calls', '-')} |")
    over = {c: sum(1 for by_task in results[c].values() for st in by_task["trace"]
                   if st["question"] and tokens(st, DEFAULT_OVERHEAD) > per_minute) for c in summary}
    biggest = max(summary.items(), key=lambda kv: kv[1]["largest_prompt"])
    lines += ["", f"Largest single prompt: {NAMES.get(biggest[0], biggest[0])}, about {biggest[1]['largest_prompt']:,} tokens. "
              f"Per-minute limit {per_minute:,}: question prompts over it, which Groq will refuse: "
              + (", ".join(f"{NAMES.get(c, c)} {n}" for c, n in over.items() if n) or "none") + ".",
              f"Prompt sizes: chars / 4 + {DEFAULT_OVERHEAD['per_message']} per message + {DEFAULT_OVERHEAD['fixed']} "
              "(fitted on Groq's reported counts in the saved heldout2 run; the preview makes no calls of its own)."]
    return "\n".join(lines) + "\n"


def to_markdown_scale(summary: dict, meta: dict) -> str:
    conds = [c for c in ALL_CONDITIONS if c in summary]
    tasks, per_chat = meta["tasks"], len(CHECKPOINTS)
    lines = [f"# lethe scale test: {meta['timestamp']}", "",
             f"model `{meta['model']}` · budget {meta['budget']} · grace {meta['grace']} · {tasks} chats × 400 messages · "
             f"{per_chat} questions per chat · 1 run", ""]
    scoring = meta.get("scoring") or {}
    if scoring.get("judged"):
        lines += [f"Graded by `{scoring['judge_model']}` against reference answers (agreement with regex "
                  f"{scoring['agreement']:.0%}).", ""]
    lines += [
        f"Accuracy is measured on the question turns only ({tasks * per_chat} per setup, one run): a sanity check, "
        "not a comparison. Prompt sizes are computed for every message as if each one got a reply "
        "(offline estimate plus the calibrated per-message overhead below); completions aren't counted.", "",
        "| setup | correct | prompt tokens per answer at msg 100 | 200 | 300 | 400 | prompt tokens over the whole chat | memories held at msg 400 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for c in conds:
        s = summary[c]
        at = [f"{s['prompt_tokens_at'].get(str(t), 0):,}" for t in CHECKPOINTS]
        held = s.get("active_at", {}).get("400")
        held = "-" if held is None else held
        lines.append(f"| {NAMES.get(c, c)} | {s['correct']}/{s['questions']} | " + " | ".join(at)
                     + f" | {s['prompt_tokens_whole_chat']:,} | {held} |")
    for c in conds:
        s = summary[c]
        if "extract_calls" in s:
            lines += ["", f"Extraction ({NAMES.get(c, c)}): measured {s['extract_calls']} calls and {s['extract_tokens']:,.0f} "
                      "tokens per chat, batched every 8 messages as in this eval. In a live chat the agent extracts before "
                      f"every answer, so once per message: about {s['live_extract_est']:,.0f} prompt tokens per chat "
                      "(computed offline and calibrated, prompts only)."]
    lines += ["", "| setup | " + " | ".join(f"msg {t}" for t in CHECKPOINTS) + " |", "|---|" + "---|" * len(CHECKPOINTS)]
    for c in conds:
        marks = [" ".join("not sent" if ok is None else "✓" if ok else "✗" for ok in summary[c]["by_turn"].get(str(t), []))
                 or "-" for t in CHECKPOINTS]
        lines.append(f"| {NAMES.get(c, c)} | " + " | ".join(marks) + " |")
    refused = {c: summary[c]["not_sent"] for c in conds if summary[c]["not_sent"]}
    if refused:
        lines += ["", "Not sent: " + ", ".join(f"{NAMES.get(c, c)} {n} question(s)" for c, n in refused.items())
                  + ". Groq refused the prompt as larger than the free tier accepts in one request (8,000 tokens per minute"
                  " for gpt-oss-20b), so those questions aren't scored; their prompt sizes are still counted above."]
    cal = meta["calibration"]
    fit = (f"fitted on {cal['calls']} answer calls of this run, worst miss {cal['max_residual']} tokens"
           if cal["source"] == "this run" else f"{cal['source']}: this run had nothing to fit")
    lines += ["", f"Calibration: Groq's prompt tokens ≈ chars / 4 + {cal['per_message']} per message + {cal['fixed']} ({fit})."]
    glitches = (meta.get("llm_glitches") or {}).get("answering")
    if glitches:
        lines += ["", f"Groq `tool_use_failed` glitch: {glitches['tool_use_retries']} retries, "
                  f"{glitches['fallback_recoveries']} answers recovered from `failed_generation`."]
    return "\n".join(lines) + "\n"


def console_safe(text: str, encoding: str | None = None) -> str:
    """The report uses ✓, ✗ and ≈. A Windows console (cp1252) can't print them, which crashed the first real run
    right after its results were saved. Swap what the console can't show; the saved files keep the real characters."""
    enc = encoding or getattr(sys.stdout, "encoding", None) or "utf-8"
    return text.encode(enc, errors="replace").decode(enc)


def checkpoint_path(model: str, budget: int, grace: int) -> Path:
    sig = json.dumps({"split": "scale", "model": model, "budget": budget, "grace": grace}, sort_keys=True)
    return HERE / "results" / f"checkpoint-scale-{hashlib.sha1(sig.encode()).hexdigest()[:10]}.jsonl"


async def run_all_scale(conditions, tasks, data, llm, embedder, budget, grace, overrides, ckpt: Path | None,
                        resume: bool, llm_stats: dict | None = None) -> dict:
    """Every (setup, chat) pair. With a checkpoint path, each finished pair is appended to it, and resume=True
    skips pairs already there (same pattern as evals/run.py). ckpt=None (preview) writes nothing."""
    stats = llm_stats if llm_stats is not None else new_llm_stats()
    done = {}
    if ckpt is not None:
        if resume and ckpt.exists():
            for line in ckpt.read_text(encoding="utf-8").splitlines():
                rec = json.loads(line)
                done[(rec["condition"], rec["task"])] = rec
                for k in LLM_STAT_KEYS:
                    stats[k] += rec["llm_stats"].get(k, 0)
            print(f"resuming: {len(done)} finished chat runs loaded from {ckpt.name}")
        elif ckpt.exists():
            ckpt.unlink()
        ckpt.parent.mkdir(exist_ok=True)
    results: dict = {}
    for cond in conditions:
        for task in tasks:
            if (cond, task["id"]) in done:
                rec = done[(cond, task["id"])]
            else:
                t0, before = time.time(), llm_stats_snapshot(llm)
                rows, trace = await trace_task(cond, task, data, llm, embedder, budget, grace, overrides)
                after = llm_stats_snapshot(llm)
                delta = {k: after[k] - before[k] for k in LLM_STAT_KEYS}
                for k in LLM_STAT_KEYS:
                    stats[k] += delta[k]
                rec = {"condition": cond, "task": task["id"], "rows": rows, "trace": trace, "llm_stats": delta}
                if ckpt is not None:
                    with ckpt.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(rec) + "\n")
                ok = sum(r["correct"] for r in rows)
                print(f"[{cond:>13}] {task['id']} {ok}/{len(rows)} correct  ({time.time() - t0:.1f}s)")
            results.setdefault(cond, {})[task["id"]] = {"rows": rec["rows"], "trace": rec["trace"]}
    return results


async def main(args):
    data = json.loads(DATA.read_text(encoding="utf-8"))
    tasks = data["tasks"][: args.limit] if args.limit else data["tasks"]
    if args.fake_embedder and not args.preview:
        raise SystemExit("--fake-embedder is for --preview only; real results need the real embedder.")
    if args.fake_embedder:
        from evals.fakes import FakeEmbedder

        embedder, overrides = FakeEmbedder(), {"min_similarity": 0.3, "reload_threshold": 0.3}
    else:
        from lethe.index import FastEmbedder

        embedder, overrides = FastEmbedder(), {}

    if args.preview:
        results = await run_all_scale(args.conditions, tasks, data, PreviewLLM(), embedder, args.budget, args.grace,
                                      overrides, ckpt=None, resume=False)
        print(console_safe("\n" + preview_report(results, args.completion_tokens, args.daily_limit, args.per_minute_limit)))
        return

    from lethe.llm import DailyLimitError, GroqLLM

    llm = GroqLLM()
    ckpt, stats = checkpoint_path(llm.model, args.budget, args.grace), new_llm_stats()
    try:
        results = await run_all_scale(args.conditions, tasks, data, llm, embedder, args.budget, args.grace,
                                      overrides, ckpt, args.resume, stats)
    except DailyLimitError as e:
        saved = len(ckpt.read_text(encoding="utf-8").splitlines()) if ckpt.exists() else 0
        raise SystemExit(f"\nStopped: {e}\nProgress saved ({saved}/{len(args.conditions) * len(tasks)} chat runs). "
                         "After the quota resets, rerun the same command with --resume.") from e

    rows = [x for by_task in results.values() for r in by_task.values() for x in r["rows"]]
    judge_llm = GroqLLM(model=args.judge_model, temperature=0.0) if args.judge else None
    try:
        scoring = await score_rows(rows, judge_llm)  # updates r["correct"] in place, so the summary sees it
    except DailyLimitError as e:
        raise SystemExit(f"\nStopped while judging: {e}\nAnswers are saved; rerun with --resume after the reset.") from e
    scoring["judge_model"] = args.judge_model if args.judge else None

    cal = calibrate([r["trace"] for by_task in results.values() for r in by_task.values()])
    summary = summarize_scale(results, cal)
    meta = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M"), "split": "scale", "model": llm.model, "budget": args.budget,
        "grace": args.grace, "tasks": len(tasks), "conditions": args.conditions, "scoring": scoring, "calibration": cal,
        "llm_glitches": {"answering": stats, "judging": llm_stats_snapshot(judge_llm) if judge_llm else None},
    }
    out_dir = HERE / "results"
    stamp = time.strftime("%Y%m%d-%H%M%S")
    (out_dir / f"run-scale-{stamp}.json").write_text(json.dumps({"meta": meta, "summary": summary, "results": results}))
    md = to_markdown_scale(summary, meta)
    (out_dir / "latest-scale.md").write_text(md, encoding="utf-8")
    ckpt.unlink(missing_ok=True)
    print(console_safe("\n" + md))


def cli():
    p = argparse.ArgumentParser(description="Scale test: tokens and memory as a chat grows to 400 messages.")
    p.add_argument("--preview", action="store_true", help="offline: stand-in LLM, prints projected Groq tokens")
    p.add_argument("--fake-embedder", action="store_true", help="with --preview: no embedding model download")
    p.add_argument("--conditions", nargs="+", default=CONDITIONS, choices=ALL_CONDITIONS)
    p.add_argument("--budget", type=int, default=6)
    p.add_argument("--grace", type=int, default=2)
    p.add_argument("--limit", type=int, default=0, help="only the first N chats")
    p.add_argument("--judge", action="store_true", help="grade answers with an LLM judge (primary score)")
    p.add_argument("--judge-model", default="openai/gpt-oss-120b")
    p.add_argument("--resume", action="store_true", help="continue an interrupted run")
    p.add_argument("--completion-tokens", type=int, default=150, help="preview: assumed completion tokens per call")
    p.add_argument("--daily-limit", type=int, default=200_000, help="preview: answering model's tokens per day")
    p.add_argument("--per-minute-limit", type=int, default=8_000, help="preview: largest prompt the provider accepts")
    asyncio.run(main(p.parse_args()))


if __name__ == "__main__":
    cli()
