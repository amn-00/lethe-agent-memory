# lethe eval (dev): 2026-10-05 20:01

model `openai/gpt-oss-20b` · budget 6 · grace 2 · 16 tasks · 22 questions per condition · memories ordered

Scored by LLM judge `openai/gpt-oss-120b` against reference answers; regex scorer shown alongside (agreement 100%).

| condition | accuracy | 95% CI | spread across runs | regex accuracy | avg prompt tokens | avg memories injected | avg active memories | extraction calls / tokens per conversation |
|---|---|---|---|---|---|---|---|---|
| lethe_extract | 96% | 86%–100% | 1 run | 96% | 229.6 | 1.68 | 2.6 | 3.2 / 2040 |

| category | lethe_extract |
|---|---|
| distractor | 100% |
| long_gap | 100% |
| multi_recall | 100% |
| single_recall | 100% |
| update | 100% |
| update_chain | 83% |
