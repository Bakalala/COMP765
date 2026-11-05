# Related-work notes for Dyna Putt

Bibliographic details below were checked online against publisher, proceedings, or
arXiv pages in November 2026. Sutton (1990) and Koenig and Simmons (1996) were read
directly. Exceptions are listed under "Verification limits".

## Target paper

Sutton (1990) presents Dyna agents that act, learn a world model, and run k
hypothetical steps between real steps, applying the same learning rule to remembered
experience. Dyna-PI builds on policy iteration and Dyna-Q on Q-learning. Experiments
use small grid mazes: navigation with and without planning, then blocking and
shortcut tasks that compare Dyna-PI, Dyna-Q, and Dyna-Q+, which adds an exploration
bonus. The paper notes that search control was kept trivial because the tasks were
small, and that reward was non-zero only when a path ended. That last point matters
here: a zero table is not optimistic in Sutton's setting, but it is in Dyna Putt's,
where every shot costs -1.

## Where to spend planning

Moore and Atkeson (1993) use a priority queue to direct backups to where the largest
possible value change lies. Their priority is an upper bound on change, not an
expectation. Their paper does not frame this against uniform sampling; that contrast
comes from Sutton and Barto (Section 8.4) or from us. Peng and Williams (1993) study
making Dyna's computation more efficient.

## Replay and models

Lin (1992) introduces experience replay alongside learned action models and teaching.
van Seijen and Sutton (2015) show a model-based policy-evaluation method and a replay
method can produce the same sequence of value functions. van Hasselt et al. (2019)
argue that when a parametric model only produces imagined transitions from observed
states for model-free updates, replay should match or beat it. In Dyna Putt a planning
update is close to a replay update.

## Initial values and exploration

Sutton and Barto (Section 2.6) describe optimistic initial values. Koenig and Simmons
(1996) show that under an action-penalty reward, zero-initialized greedy Q-learning
reaches a goal in polynomially many actions in deterministic spaces, with learning
rate 1, while a goal-reward form with zero values can need exponentially many. Their
bounds concern reaching a goal, not convergence at other learning rates. This matches
what we see: optimism finds long routes, at a cost in settling on them.

## Learned models at scale

PETS (Chua et al., 2018) pairs probabilistic neural dynamics ensembles with trajectory
sampling. MBPO (Janner et al., 2019) runs short model rollouts branched from real
states; the authors relate this to Dyna. MuZero (Schrittwieser et al., 2020) uses tree
search over a learned model without being given the rules. TD-MPC (Hansen et al., 2022)
combines short-horizon latent trajectory optimization with a learned terminal value.
TD-MPC2 (Hansen et al., 2024) reports 104 online tasks with one set of hyperparameters;
its single 317M-parameter agent covers 80 tasks, so do not merge those two figures.
DreamerV3 (Hafner et al., 2025) learns actor and critic from imagined latent
trajectories without lookahead planning at action time. DCWM (Scannell et al., 2025)
learns discrete latent codes and uses decision-time MPC; those codes are learned,
while Dyna Putt's cells are directly observed. Dyna Putt implements none of these.

## Connection to COMP 765

The schedule names "Early model-based RL and OC with learned models" (Oct. 19) and
"Modern WM Survey" (Nov. 2). The lecture notes are not in this repository, so do not
attribute specific claims to them without checking. **The project requirements also
ask for comparison with peers' papers. Only the student can supply this.**

## Verification limits

- Sutton (1991): metadata only. Peng and Williams (1993): summaries only.
- TD-MPC author order follows PMLR (Hansen, Su, Wang); arXiv lists Hansen, Wang, Su.
- Page numbers for NeurIPS papers are omitted because sources conflict or are unofficial.
- DreamerV3's description comes from the arXiv version, not the Nature text.

## References

- Chua, K., Calandra, R., McAllister, R., & Levine, S. (2018). Deep Reinforcement
  Learning in a Handful of Trials using Probabilistic Dynamics Models. NeurIPS 31.
- Hafner, D., Pasukonis, J., Ba, J., & Lillicrap, T. (2025). Mastering diverse control
  tasks through world models. Nature 640(8059):647-653.
- Hansen, N., Su, H., & Wang, X. (2022). Temporal Difference Learning for Model
  Predictive Control. ICML, PMLR 162:8387-8406.
- Hansen, N., Su, H., & Wang, X. (2024). TD-MPC2: Scalable, Robust World Models for
  Continuous Control. ICLR 2024.
- Janner, M., Fu, J., Zhang, M., & Levine, S. (2019). When to Trust Your Model:
  Model-Based Policy Optimization. NeurIPS 32.
- Koenig, S., & Simmons, R. G. (1996). The effect of representation and knowledge on
  goal-directed exploration with reinforcement-learning algorithms. Machine Learning
  22(1-3):227-250.
- Lin, L.-J. (1992). Self-improving reactive agents based on reinforcement learning,
  planning and teaching. Machine Learning 8(3-4):293-321.
- Moore, A. W., & Atkeson, C. G. (1993). Prioritized sweeping: Reinforcement learning
  with less data and less time. Machine Learning 13(1):103-130.
- Peng, J., & Williams, R. J. (1993). Efficient learning and planning within the Dyna
  framework. Adaptive Behavior 1(4):437-454.
- Scannell, A., Nakhaei, M., Kujanpää, K., Zhao, Y., Luck, K. S., Solin, A., & Pajarinen,
  J. (2025). Discrete Codebook World Models for Continuous Control. ICLR 2025.
- Schrittwieser, J., et al. (2020). Mastering Atari, Go, chess and shogi by planning
  with a learned model. Nature 588(7839):604-609.
- Sutton, R. S. (1990). Integrated Architectures for Learning, Planning, and Reacting
  Based on Approximating Dynamic Programming. ICML, pp. 216-224.
- Sutton, R. S. (1991). Dyna, an integrated architecture for learning, planning, and
  reacting. SIGART Bulletin 2(4):160-163.
- Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction, 2nd
  ed. MIT Press.
- van Hasselt, H., Hessel, M., & Aslanides, J. (2019). When to use parametric models in
  reinforcement learning? NeurIPS 32.
- van Seijen, H., & Sutton, R. S. (2015). A Deeper Look at Planning as Learning from
  Replay. ICML, PMLR 37:2314-2322.
