# lethe eval (heldout2): 2026-10-07 21:48

model `openai/gpt-oss-20b` · budget 6 · grace 2 · 20 tasks · 39 questions per condition · memories ordered

Scored by LLM judge `openai/gpt-oss-120b` against reference answers; regex scorer shown alongside (agreement 99%).

| condition | accuracy | 95% CI | spread across runs | regex accuracy | avg prompt tokens | avg memories injected | avg active memories | extraction calls / tokens per conversation |
|---|---|---|---|---|---|---|---|---|
| no_memory | 2% | 0%–4% | ±1.5 pts (3 runs) | 3% | 140.8 | 0.0 | - | - |
| full_history | 92% | 79%–100% | ±2.6 pts (3 runs) | 92% | 514.7 | 0.0 | - | - |
| naive_rag | 97% | 92%–100% | ±0.0 pts (3 runs) | 97% | 255.3 | 3.41 | 25.0 | - |
| lethe_extract | 97% | 93%–100% | ±4.4 pts (3 runs) | 99% | 245.1 | 2.38 | 3.0 | 4.7 / 3150 |

| category | no_memory | full_history | naive_rag | lethe_extract |
|---|---|---|---|---|
| distractor | 0% | 70% | 100% | 93% |
| long_gap | 0% | 100% | 100% | 100% |
| multi_recall | 2% | 100% | 100% | 100% |
| update_chain | 4% | 100% | 88% | 96% |

Note: Groq `tool_use_failed` glitch (gpt-oss): answering + extraction hit 3 retries and 0 answers recovered from `failed_generation` after every retry failed (not counted for 21 task runs checkpointed before the counter existed); judging hit 0 retries and 0 recoveries.
