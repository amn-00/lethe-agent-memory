# lethe

Long-horizon memory for LLM agents. Decides what to **retain**, **evict**, and **reload** across a conversation, and tracks which retrieved memories the answer actually **used**.

> Status: core library, API, and eval harness built. Benchmark numbers in `evals/results/latest.md`.

## How it works

Every chat turn:

1. **Recall**: embed the query, search active memories in Chroma. Also check the archive; evicted memories scoring above `reload_threshold` get pulled back to active.
2. **Answer**: the LLM (Groq) gets the recalled memories plus a short window of recent messages.
3. **Used vs ignored**: each recalled memory is scored by lexical overlap with the answer, excluding words that came from the question (an answer echoing the query isn't evidence of use).
4. **Store**: the user message becomes a new memory.
5. **Evict**: if active memories exceed the budget, the lowest retention scores move to the archive.

Retention score = weighted mix of:

| Signal | Meaning |
|---|---|
| recency | halves every `half_life_turns` without access |
| frequency | how often it gets retrieved |
| used_rate | how often it's used *when* retrieved (Laplace-smoothed) |

**Fact extraction** (on by default in the API). Instead of storing every raw message, an LLM rewrites durable facts as standalone first-person sentences and drops chatter and questions: "started a new book, the three body problem" becomes "I'm currently reading The Three-Body Problem." That fixes two problems the eval exposed: cold start (at save time a fact and "lol ok" looked identical) and phrasing mismatch (the newest fact didn't match how the question was asked). Messages are batched, one extraction call per 8 messages or right before an answer is needed, and each fact keeps the turn it was said. Before extracting a batch, related facts already in memory are looked up (a local embedding search with no side effects) and shown to the extractor, so an update that spans batches ("now reading project hail mary" in one, "started a new book" in the next) is written as an update rather than an unrelated new fact. If the extractor returns unparseable output, the raw messages are stored instead so nothing is lost.

Recalled memories are shown to the model oldest first, tagged with the turn they were said, so when a fact changes ("moved to indore" at turn 7, "now in hyderabad" at turn 13) the model can tell which one is current. The held-out eval showed similarity-ranked lists made RAG answer update chains backwards.

Memories retrieved often but never used sink fastest. Anything accessed in the last `grace_turns` is protected. Every decision is logged with its score breakdown in `ops`.

## Stack

FastAPI · Groq (Llama) · ChromaDB · FastEmbed (ONNX, no torch) · SQLite

SQLite is the source of truth for text, tier, and stats; Chroma holds vectors with tier/session metadata for filtered search.

## Run

```bash
pip install -e ".[dev]"
cp .env.example .env   # add your GROQ_API_KEY; it's loaded automatically
uvicorn lethe.api:build_default_app --factory --reload
```

Config via env vars or `.env`: `LETHE_EXTRACT` (1 = store extracted facts, 0 = raw messages), `LETHE_BUDGET` (active memory cap, default 40), `LETHE_GRACE` (default 3), `GROQ_MODEL` (default `openai/gpt-oss-20b`), `GROQ_MIN_INTERVAL` (seconds between calls, default 2.2 for the free tier).

```bash
curl -X POST localhost:8000/sessions/demo/chat -H "Content-Type: application/json" \
  -d '{"message": "my sister lives in pune"}'
curl localhost:8000/sessions/demo/memories   # tiers + score breakdowns
curl localhost:8000/sessions/demo/ops        # add / retrieve / used / ignored / evict / reload log
```

## Eval

Scripted multi-turn conversations: facts planted early, buried under filler chatter, then asked about later. Four conditions on the same conversations:

| condition | what the model sees |
|---|---|
| `no_memory` | last 4 messages only |
| `full_history` | the entire conversation (accuracy ceiling, most tokens) |
| `naive_rag` | similarity retrieval over every message, nothing ever evicted |
| `lethe` | same retrieval, but active memory capped by the retain/evict/reload policy |

Task categories: single recall, multi-fact recall, distractors (similar but wrong facts), updates (a fact changes), long gaps. Scoring is deterministic regex matching against expected answers.

Two task sets:

- **dev** (`evals/tasks.json`, 12 conversations, 16 questions): where settings get tuned.
- **heldout** (`evals/heldout.json`, 16 conversations, 34 questions): harder, with unguessable answers, lookalike distractors, 3-step update chains and gaps up to 45 turns. Fresh facts and filler, checked by a test. Run once per version of lethe; never tune on it.

```bash
python -m evals.run --dry-run                  # offline pipeline check
python -m evals.run                            # dev set, ~64 Groq calls
python -m evals.run --split heldout            # held-out set, ~136 Groq calls
python -m evals.run --reload-threshold 0.55    # try other settings (dev only)
python -m evals.run --conditions lethe lethe_extract --judge   # raw messages vs extracted facts
```

**Scoring.** Every question has a reference answer. Two scorers run side by side: a deterministic regex match (free, but it can't grade ordering: "before Farah it was Omkar" contains "farah" and passes) and an LLM judge (`--judge`, `openai/gpt-oss-120b` by default) that compares each answer with the reference. When the judge runs it is the primary score, and the report shows how often the two agree. `python -m evals.rescore --split heldout --judge` re-grades a saved run without regenerating answers.

Long runs checkpoint after every finished conversation. If the provider's daily quota runs out, the run stops immediately (no pointless retries) and keeps its progress; rerun the same command with `--resume` after the reset. Per-minute limits are still waited out automatically.

Only question turns call the LLM; filler turns get a canned reply so a full run fits Groq's free tier (`--llm-every-turn` for full fidelity).

## Tests

```bash
pytest -q
```

Tests run offline with a fake embedder and fake LLM.

## Known limits

- Lexical overlap misses paraphrased usage.
- Raw user messages are stored as memories; fact extraction is a planned upgrade.
- If every active memory is inside the grace window, the budget can temporarily overflow.
