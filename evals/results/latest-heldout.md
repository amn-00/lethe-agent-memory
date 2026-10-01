# lethe eval (heldout): 2026-10-01 21:50

model `openai/gpt-oss-20b` · budget 6 · grace 2 · 16 tasks · 34 questions per condition · memories ordered

Scored by LLM judge `openai/gpt-oss-120b` against reference answers; regex scorer shown alongside (agreement 98%).

| condition | accuracy | regex accuracy | avg prompt tokens | avg memories injected | avg active memories | extraction calls / tokens per conversation |
|---|---|---|---|---|---|---|
| naive_extract | 97% | 100% | 249.7 | 2.65 | 4.8 | 5.0 / 2649 (1 parse failures) |
| lethe_extract | 97% | 97% | 249.8 | 2.68 | 4.4 | 5.0 / 2544 |

| category | naive_extract | lethe_extract |
|---|---|---|
| distractor | 100% | 100% |
| long_gap | 100% | 100% |
| multi_recall | 100% | 100% |
| update_chain | 88% | 88% |
