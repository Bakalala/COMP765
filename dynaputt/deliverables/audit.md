# Implementation and demo audit

Line references are to the current source. The primary and archived CSVs were
regenerated and compared byte for byte, and the browser demo was exercised in a
browser. The manual check on the student's Mac is still outstanding.

## Environment

- Confirmed: states are traversable cells, excluding walls and the bumper
  (`golf_env.py`, `valid_cells` in `__init__`). Easy has 54 and Wall-gate 56, but only
  15 and 23 are reachable from the start, because shots move two or four cells.
- Confirmed: 16 actions, eight directions at 2 or 4 ticks; reward -1 per shot.
- Confirmed: shortest routes by breadth-first search are 2 shots on Easy and 5 on
  Wall-gate. Stable greedy policies reach these lengths.
- Confirmed: the 20-shot cap is a truncation. `run_experiments.py` passes
  `info["sank"]`, not `done`, to `observe`, so truncations bootstrap.

## Learners

- Confirmed: every real shot gets one Q update; Dyna-Q then samples k stored pairs
  uniformly with replacement (`agents.py`, `DynaQAgent.observe`).
- Confirmed: exploration and planning use separate generators (`self.rng` and
  `self.planning_rng`). Exploration can still diverge across planning settings once
  behavior differs, because the number of draws depends on the states visited.
- Confirmed consequential issue, not a bug: `self.q` starts at zero while every
  return is negative, so the table is optimistic for every pair. Neither the code
  comments nor the earlier report mentioned this. It strongly changes the measured
  planning benefit (see `analysis_notes.md`). The primary run keeps this setting;
  `run_extended.py` varies it.
- Confirmed: ties break randomly in training and by lowest index in evaluation.

## Experiment and metrics

- Confirmed: seeds, episode counts, schedule, and evaluation match across settings.
- Confirmed: `results/episode_results.csv` regenerates byte for byte from current
  code. The archive regenerates byte for byte with `planning_rng = rng`, so the
  documented protocol difference is real. 0-update rows are identical in both.
- Fixed: `summarize_results.py` hard-coded the sentence that the 5-versus-20
  comparison "is not monotonic", which is false for the primary run.
- Fixed: the research question concerns real shots, but only episodes were reported.
  The summarizer now also reports shots to threshold and a greedy-stable episode.

## Browser demo

- Confirmed: all 1,728 non-terminal transitions of Easy and Wall-gate match Python
  (state, sink flag, and path), tested by running both implementations.
- Confirmed: the demo agent uses one shared random stream (`DynaQ.rng`), 700 training
  episodes, and seed 7. It is illustrative and is not an experiment checkpoint.
- Confirmed: trained greedy play sinks in 2 shots on Easy and 5 on Wall-gate.
- Fixed bug: `train()` returned a finished environment, so the page loaded showing a
  nonzero stroke count and "Take shot" did nothing until Reset.
- Fixed bug: the start path was stored as a flat `[x, y]`, so `draw()` threw "number 1
  is not iterable" on load and after Reset, and the ball was not drawn.
- Fixed: clicking after a round ended was silently ignored; it now shows a message.
- After the fixes, load, human shot, agent shots, reset, and course switching ran with
  no console errors in Claude's built-in browser. This is not the Mac check.

## Interpretation limits

1. Hand-built deterministic courses; exact state; no model error.
2. Fixed hyperparameters. The -20 failures depend on the epsilon decay and budget.
3. Five primary seeds cannot resolve small effects.
4. Threshold is measured under exploration and can be floored by epsilon.
5. In the ladder family, route length moves together with size and wall count.
