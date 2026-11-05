# Dyna Putt

A small, deterministic mini-golf game for studying Sutton's Dyna architecture.
The agent sees the ball's resting grid cell and chooses one of 16 discrete
shots. Q-learning updates from real shots; Dyna-Q also updates from simulated
transitions stored in its learned lookup-table model.

## Setup

From this directory, create and activate a virtual environment, then install:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Play the showcase

```bash
python play.py
```

The script opens the standalone `index.html` demo in your default browser. The
game and small Dyna-Q agents run locally in the browser; it needs no server, UI
package, or network connection. If the browser does not open automatically,
open `index.html` directly. Click the direction and power controls, take a shot,
or let Dyna-Q choose. Keyboard shortcuts are `1`-`8` for direction, `P` to
toggle power, and `Space` to shoot. Human shots also update the agent's learned
model. The showcase trains its own illustrative agent with 20 planning updates
per shot, for 700 episodes with one random stream. It is not a checkpoint from
the measured experiments.

## Run the comparison

```bash
python run_experiments.py
```

The runner compares 0, 5, and 20 Dyna planning updates on an open course and a
wall-gate course with a forced route around an interior barrier, across five
random seeds. Action selection and planning-pair sampling use separate random
streams. It writes episode-level metrics and a learning-curve plot to
`results/`. The original exploratory run, which used one shared random stream,
is preserved in `results/rng_coupled_baseline/` for comparison and must not be
pooled with the primary data.

Summarize the saved data with:

```bash
python summarize_results.py
```

This writes `results/summary.csv` and `results/experiment_summary.md`.

Run the robustness studies with:

```bash
python run_extended.py
```

This takes about two minutes on eight cores and writes `results/extended/`. It uses
100 seeds that are separate from the primary five, and covers planning budget,
shared versus separate random streams, initial Q values, and a route-length
family of courses. See [`results/extended/README.md`](results/extended/README.md).

## Research question

Does Dyna-Q learn a course in fewer real shots than Q-learning, and how does the
number of imagined planning updates affect learning speed?

The core comparison is deliberately narrow. The game uses discrete states and
actions so the world model remains inspectable.

## Main findings

- In the five-seed primary run, more planning means fewer real shots to the
  threshold on both courses, for every paired seed.
- With 100 seeds the same ordering holds on average, with sharply diminishing
  returns, and the Easy 5-versus-20 gap is much smaller than five seeds suggested.
- An ordering reversal in the archived run is explained by five-seed noise, not by
  its random-number protocol.
- The zero Q table is optimistic because every shot costs -1. With non-optimistic
  values the planning advantage disappears on Easy and shrinks on Wall-gate.
- That optimism also finds long routes: at initial Q=-20 most agents never solve
  the 8-shot course.

## Method and related work

The project adapts the learned-model-and-planning loop in Sutton's 1990 Dyna
architecture to a small mini-golf task. It is a focused adaptation, not a full
reproduction of Sutton's experiments or of later methods such as PETS or TD-MPC.

Status and remaining work are in [LLM_TASKS.md](LLM_TASKS.md). The report, its
source, the station slides, and the demo plan are in
[`deliverables/`](deliverables/).

## Implementation notes

The current environment is deterministic and state-based. Each stored
state-action pair has one observed outcome. The 20-shot episode limit is a
truncation; only sinking the ball is a terminal transition for Q-value updates.
Results should be interpreted as a small controlled demonstration of planning
from an imperfectly learned model, not evidence about real-world robotics or
image-based world models.
