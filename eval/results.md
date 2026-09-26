| configuration | R@1 | R@3 | R@5 | R@10 | MRR | no-answer on negatives | speaker acc. | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|---|
| keyword (FTS) | 0.72 | 0.75 | 0.76 | 0.76 | 0.73 | 1.00 | 0.96 | 11 | 24 |
| fuzzy (trigram) | 0.63 | 0.63 | 0.63 | 0.63 | 0.63 | 1.00 | 0.98 | 57 | 153 |
| vector (gemini) | 0.90 | 0.97 | 0.97 | 1.00 | 0.94 | 0.00 | 0.97 | 32 | 53 |
| hybrid, children ungrouped (gemini) | 0.94 | 0.97 | 0.99 | 1.00 | 0.96 | 0.00 | 0.96 | 25 | 103 |
| hybrid (gemini) | 0.94 | 0.97 | 1.00 | 1.00 | 0.96 | 0.00 | 0.96 | 39 | 129 |
| hybrid+rerank (gemini) | 0.96 | 1.00 | 1.00 | 1.00 | 0.98 | 0.80 | 0.97 | 40 | 101 |
| hybrid+rerank+mmr (gemini) | 0.96 | 1.00 | 1.00 | 1.00 | 0.98 | 0.80 | 0.97 | 78 | 168 |

Recall@5 by query type:

| configuration | entity | keyword | phrase | role | semantic | speaker | typo |
|---|---|---|---|---|---|---|---|
| keyword (FTS) | 1.00 | 1.00 | 1.00 | 1.00 | 0.50 | 0.83 | 0.40 |
| fuzzy (trigram) | 1.00 | 1.00 | 1.00 | 0.86 | 0.08 | 0.83 | 1.00 |
| vector (gemini) | 1.00 | 0.89 | 1.00 | 0.86 | 1.00 | 1.00 | 1.00 |
| hybrid, children ungrouped (gemini) | 1.00 | 1.00 | 1.00 | 1.00 | 0.96 | 1.00 | 1.00 |
| hybrid (gemini) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| hybrid+rerank (gemini) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| hybrid+rerank+mmr (gemini) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
