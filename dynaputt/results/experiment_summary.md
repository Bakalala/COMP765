# Primary experiment summary

Generated from `episode_results.csv`. Threshold is the first episode ending
a 20-episode window with at least 80% training sinks. Greedy success is the
mean over the checkpoints after episodes 301-350 and five agent seeds.
Real shots are training shots up to the threshold episode. Greedy-stable is
the first episode after which every checkpoint sinks, for at least the last
50 checkpoints; the column shows its mean and how many seeds qualify. The game
and greedy evaluation policy are deterministic. Action exploration and model
planning use separate random-number streams in this run. The earlier coupled
stream run is preserved in `results/rng_coupled_baseline/`.

| Course | Planning updates | Episodes to threshold (mean ± SD) | Real shots to threshold (mean ± SD) | First 50 training success | Greedy success, episodes 301-350 | Greedy-stable episode (seeds) |
|---|---:|---:|---:|---:|---:|---:|
| easy | 0 | 79.0 ± 9.9 | 1136 ± 115 | 0.448 | 1.000 | 169.8 (5/5) |
| easy | 5 | 61.0 ± 4.6 | 827 ± 54 | 0.528 | 1.000 | 57.6 (5/5) |
| easy | 20 | 49.2 ± 7.9 | 713 ± 132 | 0.572 | 1.000 | 48.6 (5/5) |
| wall_gate | 0 | 284.8 ± 18.4 | 5063 ± 231 | 0.028 | 0.400 | none (0/5) |
| wall_gate | 5 | 103.8 ± 8.1 | 1848 ± 112 | 0.036 | 1.000 | 129.2 (5/5) |
| wall_gate | 20 | 89.8 ± 4.7 | 1607 ± 59 | 0.044 | 1.000 | 74.6 (5/5) |

These results come from five seeds on a small deterministic toy environment
and support conclusions about this implementation only. See
`results/extended/` for the 100-seed robustness and initial-value studies.
