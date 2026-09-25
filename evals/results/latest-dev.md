# lethe eval (dev): 2026-09-25 22:02

model `openai/gpt-oss-20b` · budget 6 · grace 2 · 16 tasks · 22 questions per condition · memories ordered

Scored by LLM judge `openai/gpt-oss-120b` against reference answers; regex scorer shown alongside (agreement 100%).

| condition | accuracy | regex accuracy | avg prompt tokens | avg memories injected | avg active memories |
|---|---|---|---|---|---|
| naive_rag | 96% | 96% | 237.0 | 2.41 | 20.3 |
| lethe | 96% | 96% | 233.5 | 2.14 | 6.1 |

| category | naive_rag | lethe |
|---|---|---|
| distractor | 100% | 100% |
| long_gap | 100% | 100% |
| multi_recall | 100% | 100% |
| single_recall | 100% | 100% |
| update | 100% | 100% |
| update_chain | 83% | 83% |
