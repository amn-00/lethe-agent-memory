# lethe scale test: 2026-10-23 20:46

model `openai/gpt-oss-20b` · budget 6 · grace 2 · 2 chats × 400 messages · 4 questions per chat · 1 run

Graded by `openai/gpt-oss-120b` against reference answers (agreement with regex 100%).

Accuracy is measured on the question turns only (8 per setup, one run): a sanity check, not a comparison. Prompt sizes are computed for every message as if each one got a reply (offline estimate plus the calibrated per-message overhead below); completions aren't counted.

| setup | correct | prompt tokens per answer at msg 100 | 200 | 300 | 400 | prompt tokens over the whole chat | memories held at msg 400 |
|---|---|---|---|---|---|---|---|
| No memory | 0/8 | 118 | 118 | 118 | 122 | 47,246 | - |
| Full history | 8/8 | 1,968 | 3,883 | 5,760 | 7,655 | 1,551,224 | - |
| Search everything | 8/8 | 224 | 254 | 240 | 275 | 99,994 | 400.0 |
| lethe | 8/8 | 224 | 232 | 226 | 257 | 80,260 | 6.5 |

Extraction (lethe): measured 52.0 calls and 43,470 tokens per chat, batched every 8 messages as in this eval. In a live chat the agent extracts before every answer, so once per message: about 215,622 prompt tokens per chat (computed offline and calibrated, prompts only).

| setup | msg 100 | msg 200 | msg 300 | msg 400 |
|---|---|---|---|---|
| No memory | ✗ ✗ | ✗ ✗ | ✗ ✗ | ✗ ✗ |
| Full history | ✓ ✓ | ✓ ✓ | ✓ ✓ | ✓ ✓ |
| Search everything | ✓ ✓ | ✓ ✓ | ✓ ✓ | ✓ ✓ |
| lethe | ✓ ✓ | ✓ ✓ | ✓ ✓ | ✓ ✓ |

Calibration: Groq's prompt tokens ≈ chars / 4 + 4.79 per message + 53.7 (fitted on 32 answer calls of this run, worst miss 17.9 tokens).

Groq `tool_use_failed` glitch: 0 retries, 0 answers recovered from `failed_generation`.
