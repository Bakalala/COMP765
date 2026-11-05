# Report outline — Dyna Putt

## Proposed title

**Dyna Putt: Does Imagined Practice Help an Agent Learn a Mini-Golf Course?**

## Research question and hypothesis

Does Dyna-Q learn a small mini-golf course in fewer real shots than Q-learning?
The hypothesis is that Dyna-Q's additional updates from its learned transition
table improve sample efficiency. The experiment varies the number of planning
updates (0, 5, or 20) after each real shot. The stronger prediction from the
project proposal is that the advantage grows with more planning updates, so the
20-update agent should reach reliable success sooner than the 5-update agent.
The current primary run supports both predictions on both courses, and 100 further
seeds support them on average. Two exploratory hypotheses were added later: part
of the gain comes from the optimistic zero Q table, and the remaining gain depends
on route length. See `results/extended/summary.md`.

## Method

Describe Sutton's Dyna architecture in your own words. Explain the discrete
course state, 16 shot actions, reward, Q-learning update, transition table, and
imagined planning updates. State clearly that this project is a small adaptation
of the paper's navigation setting, not a reproduction of its full experiments.

## Experimental setup

- Two deterministic course layouts: an open course and a wall-gate course.
- Three planning settings: 0, 5, and 20 model updates per observed shot.
- Five agent random seeds, with the same episode limit and learning settings.
- Extended studies on 100 separate seeds, never pooled with the primary five.
- Episode-level training success, greedy evaluation success, and strokes per
  successful evaluation episode.
- Report means and variability across seeds. Preserve unsuccessful runs rather
  than excluding them.

Before writing results, run the experiment runner and inspect the saved CSV and
plot. Do not infer or invent findings from the hypothesis. Preliminary outcomes
from the current configuration are recorded in `results/experiment_summary.md`.
The current report in `deliverables/report/` follows this outline.

## Related work

Start with Sutton's 1990 Dyna paper, then compare the tabular learned transition
model with later approaches such as PETS (probabilistic neural dynamics models
and trajectory sampling), TD-MPC (learned latent dynamics, short-horizon
planning, and a learned value estimate), and TD-MPC2 (a later scalable
continuous-control world model). A recent discrete-codebook world-model paper
can provide a brief contrast: its learned latent codes are discrete, but differ
from Dyna Putt's directly indexed grid cells. Focus on model representation and
planning; do not imply this project implements any of those later methods.

- Sutton, R. S. (1990). *Integrated Architectures for Learning, Planning, and
  Reacting Based on Approximating Dynamic Programming*.
  https://mlanthology.org/icml/1990/sutton1990icml-integrated/
- Chua, K. et al. (2018). *Deep Reinforcement Learning in a Handful of Trials
  using Probabilistic Dynamics Models*. https://arxiv.org/abs/1805.12114
- Hansen, N., Su, H., & Wang, X. (2022). *Temporal Difference Learning for Model
  Predictive Control*. ICML 2022. https://proceedings.mlr.press/v162/hansen22a.html
- Hansen, N. et al. (2024). *TD-MPC2: Scalable, Robust World Models for
  Continuous Control*.
  https://proceedings.iclr.cc/paper_files/paper/2024/hash/cf73d57b6dcda32b293df7c2d5341f49-Abstract-Conference.html
- Scannell, A. et al. (2025). *Discrete Codebook World Models for Continuous
  Control*.
  https://proceedings.iclr.cc/paper_files/paper/2025/hash/1dd85d064697ec1a258cd2b8755b8c6d-Abstract-Conference.html

The full verified reference list is in `deliverables/literature_notes.md`.

Relate the project to the course's model-based RL and modern world-model
lectures and to papers your peers presented. Write the discussion in your own
words and cite the papers directly.

## Findings and limitations

Present the learning curves and evaluation metrics, including any case where
more imagined updates fail to help. Discuss the small discrete state/action
space, deterministic physics, hand-built courses, and fixed hyperparameters.
The results support conclusions about this toy environment only.

## Conclusion and showcase

Summarize whether imagined updates changed learning speed, what the result
suggests about planning from a learned model, and one next question. For the
interactive session, show a course, let a person take a shot, and compare a
trained agent's selected shot and path.
