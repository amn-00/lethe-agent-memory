# lethe eval (heldout): 2026-09-25 22:17

model `openai/gpt-oss-20b` · budget 6 · grace 2 · 16 tasks · 34 questions per condition · memories ordered

Scored by LLM judge `openai/gpt-oss-120b` against reference answers; regex scorer shown alongside (agreement 100%).

| condition | accuracy | regex accuracy | avg prompt tokens | avg memories injected | avg active memories |
|---|---|---|---|---|---|
| no_memory | 0% | 0% | 134.2 | 0.0 | - |
| full_history | 100% | 100% | 567.4 | 0.0 | - |
| naive_rag | 100% | 100% | 255.1 | 3.12 | 27.5 |
| lethe | 94% | 94% | 250.1 | 2.79 | 6.1 |

| category | no_memory | full_history | naive_rag | lethe |
|---|---|---|---|---|
| distractor | 0% | 100% | 100% | 100% |
| long_gap | 0% | 100% | 100% | 100% |
| multi_recall | 0% | 100% | 100% | 100% |
| update_chain | 0% | 100% | 100% | 75% |
