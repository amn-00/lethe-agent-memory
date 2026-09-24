# lethe

Long-horizon memory for LLM agents. Decides what to **retain**, **evict**, and **reload** across a conversation, and tracks which retrieved memories the answer actually **used**.

> Weekend 1 build: core library + API. Eval harness, UI, and benchmark numbers come next.

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

Memories retrieved often but never used sink fastest. Anything accessed in the last `grace_turns` is protected. Every decision is logged with its score breakdown in `ops`.

## Stack

FastAPI · Groq (Llama) · ChromaDB · FastEmbed (ONNX, no torch) · SQLite

SQLite is the source of truth for text, tier, and stats; Chroma holds vectors with tier/session metadata for filtered search.

## Run

```bash
pip install -e ".[dev]"
cp .env.example .env   # add your GROQ_API_KEY
export $(cat .env | xargs)
uvicorn lethe.api:build_default_app --factory --reload
```

```bash
curl -X POST localhost:8000/sessions/demo/chat -H "Content-Type: application/json" \
  -d '{"message": "my sister lives in pune"}'
curl localhost:8000/sessions/demo/memories   # tiers + score breakdowns
curl localhost:8000/sessions/demo/ops        # add / retrieve / used / ignored / evict / reload log
```

## Tests

```bash
pytest -q
```

Tests run offline with a fake embedder and fake LLM.

## Known limits

- Lexical overlap misses paraphrased usage.
- Raw user messages are stored as memories; fact extraction is a planned upgrade.
- If every active memory is inside the grace window, the budget can temporarily overflow.
