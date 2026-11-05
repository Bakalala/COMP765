# Archived exploratory run

This is the original 350-episode, five-seed experiment before separating the
agent's action-selection and Dyna planning random-number streams. It is
preserved for transparency and should not be mixed with the current primary
results in `results/`.

In this archived code path, planning-pair draws consumed the same RNG used by
epsilon-greedy actions. The extra draws changed later exploration choices as
planning count increased. The primary run now uses independent streams; see the
current source and `results/experiment_summary.md`.

This archive can be regenerated exactly: set `self.planning_rng = self.rng` in
`DynaQAgent` and rerun `run_experiments.py`. The 0-update rows match the primary
run because that setting makes no planning draws. The easy-course reversal here
(5 updates ahead of 20) is best explained by five-seed noise, not the shared
stream: see the stream comparison in `results/extended/summary.md`.
