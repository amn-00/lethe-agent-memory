# lethe eval: 2026-09-25 02:06

model `openai/gpt-oss-20b` · budget 6 · grace 2 · 12 tasks · 16 questions per condition

| condition | accuracy | avg prompt tokens | avg memories injected | avg active memories |
|---|---|---|---|---|
| no_memory | 6% | 129.1 | 0.0 | - |
| full_history | 94% | 425.8 | 0.0 | - |
| naive_rag | 100% | 179.3 | 2.0 | 20.2 |
| lethe | 100% | 177.8 | 1.75 | 6.0 |

| category | no_memory | full_history | naive_rag | lethe |
|---|---|---|---|---|
| distractor | 0% | 50% | 100% | 100% |
| long_gap | 0% | 100% | 100% | 100% |
| multi_recall | 0% | 100% | 100% | 100% |
| single_recall | 25% | 100% | 100% | 100% |
| update | 0% | 100% | 100% | 100% |
