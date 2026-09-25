# lethe eval (heldout): 2026-09-25 21:36

model `openai/gpt-oss-20b` · budget 6 · grace 2 · 16 tasks · 34 questions per condition

Scored by LLM judge `openai/gpt-oss-120b` against reference answers; regex scorer shown alongside (agreement 98%).

| condition | accuracy | regex accuracy | avg prompt tokens | avg memories injected | avg active memories |
|---|---|---|---|---|---|
| no_memory | 0% | 0% | 137.4 | 0.0 | - |
| full_history | 94% | 94% | 568.0 | 0.0 | - |
| naive_rag | 85% | 91% | 200.4 | 3.12 | 27.5 |
| lethe | 82% | 85% | 197.3 | 2.79 | 6.1 |

| category | no_memory | full_history | naive_rag | lethe |
|---|---|---|---|---|
| distractor | 0% | 67% | 100% | 100% |
| long_gap | 0% | 100% | 100% | 100% |
| multi_recall | 0% | 100% | 100% | 100% |
| update_chain | 0% | 100% | 38% | 25% |
