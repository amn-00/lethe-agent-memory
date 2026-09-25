# lethe eval (dev): 2026-09-25 22:33

model `openai/gpt-oss-20b` · budget 6 · grace 2 · 16 tasks · 22 questions per condition · memories ordered

Scored by LLM judge `openai/gpt-oss-120b` against reference answers; regex scorer shown alongside (agreement 100%).

| condition | accuracy | regex accuracy | avg prompt tokens | avg memories injected | avg active memories | extraction calls / tokens per conversation |
|---|---|---|---|---|---|---|
| lethe | 96% | 96% | 233.2 | 2.14 | 6.1 | - |
| lethe_extract | 96% | 96% | 230.0 | 1.73 | 4.2 | 3.2 / 1550 |

| category | lethe | lethe_extract |
|---|---|---|
| distractor | 100% | 100% |
| long_gap | 100% | 100% |
| multi_recall | 100% | 100% |
| single_recall | 100% | 100% |
| update | 100% | 100% |
| update_chain | 83% | 83% |
