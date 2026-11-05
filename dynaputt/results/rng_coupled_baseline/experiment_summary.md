# Preliminary experiment summary

Generated from `episode_results.csv`. Threshold is the first episode ending
a 20-episode window with at least 80% training sinks. Final greedy success is
the mean over the last 50 episode checkpoints and five agent seeds. The game
and greedy evaluation policy are deterministic.

| Course | Planning updates | Episodes to threshold (mean ± SD) | First 50 training success | Final greedy success |
|---|---:|---:|---:|---:|
| easy | 0 | 79.0 ± 9.9 | 0.448 | 1.000 |
| easy | 5 | 50.4 ± 11.0 | 0.552 | 1.000 |
| easy | 20 | 54.4 ± 7.1 | 0.516 | 1.000 |
| wall_gate | 0 | 284.8 ± 18.4 | 0.028 | 0.400 |
| wall_gate | 5 | 100.0 ± 6.3 | 0.008 | 1.000 |
| wall_gate | 20 | 86.4 ± 3.2 | 0.040 | 1.000 |

These are preliminary results for a small deterministic toy environment.
They support conclusions about this implementation only. The 5-versus-20
planning-update comparison is not monotonic across the two courses.
