# Related-work notes

Bibkeys refer to `report/references.bib`. Metadata was checked on 2026-11-05/06. Depth of reading: LeWM v3 in full, plus its code at commit tree `8edfeb3`; NGC, TRM, SCALE, Temporal Straightening and EpicWorldModel in part (HTML); the rest at abstract and metadata level, so check any detail in the paper before citing it.

## Read this first

- **Novelty risk.** `you2026ngc` (arXiv, Oct 2026) already replaces LeWM's L2 cost with shortest paths on a graph of latents, and reports near-solving TwoRoom. The report must cite it. The Dyna-style contribution has to be argued against it, ideally with NGC as a baseline. NGC's edges are mutual 15-nearest neighbours weighted by latent Euclidean distance, with no actions, value backups or model-imagined transitions, so that is where this project can differ.
- **The research question is partly answered.** `you2026ngc`, `li2026trm` and `hu2026scale` all measure L2 failing as a reachability proxy on TwoRoom. The mini-golf result is best framed as a replication in a new domain, with shots and bounces, not as a first finding.
- **Paper and code disagree on SIGReg and lambda.** See "Paper versus code". State which one this project follows.

## LeWorldModel implementation details (`maes2026lewm`, arXiv v3 unless noted)

**Objective.** L = L_pred + lambda * SIGReg(Z). L_pred is the squared L2 error between predicted and encoded next embedding, with teacher forcing. No stop-gradient or EMA; everything trains jointly end to end.

**SIGReg (Sec. 3.1, App. A).**
- Z has shape N x B x d (history, batch, embedding). Take M random unit directions u^(m) sampled uniformly on the hypersphere, and set h^(m) = Z u^(m).
- SIGReg(Z) = (1/M) sum_m T(h^(m)).
- Epps-Pulley: T = integral over R of w(t) |phi_N(t; h) - phi_0(t)|^2 dt. phi_N(t; h) = (1/N) sum_n exp(i t h_n) is the empirical characteristic function, and phi_0 is that of N(0,1). The weight is, for example, w(t) = exp(-t^2 / (2 lambda^2)). Notation clashes: this lambda is not the loss weight, and in App. A N counts samples, not history.
- Cramer-Wold justifies the slicing: as M grows, SIGReg -> 0 iff the embedding law tends to N(0, I).
- Step-wise: the pseudocode applies SIGReg to `emb.transpose(0, 1)`, so statistics run over the batch at each time step, not across time. App. H suggests this leaves temporal structure free and may explain emergent latent straightening.

**Constants.**
- Projections: M = 1024. Fig. 15 sweeps 64 to 1024 with little effect.
- Knots: the text gives no default. Fig. 15 sweeps 4, 8, 12, 17, 32, and accuracy falls at 4; the code uses 17.
- Integration: trapezoid with nodes uniform on [0.2, 4] (App. A).
- lambda = 0.1 in the text; Fig. 16 peaks near 0.09 and stays above 80% on [0.01, 0.2].

**Encoder and projection.** ViT-Tiny, patch 14, 12 layers, 3 heads, width 192 (about 5M parameters). z_t is the last-layer [CLS] token passed through a "1-layer MLP with Batch Normalization". The stated reason: the ViT's final LayerNorm stops the anti-collapse objective from being optimised effectively. The predictor (6 layers, 16 heads, 10% dropout, about 10M parameters, AdaLN action conditioning initialised to zero, causal masking) has a projector of the same form.

**History length (App. D).** 3 for PushT and OGBench-Cube, 1 for TwoRoom. Reacher is not stated.

**CEM planning (Sec. 3.2, App. B, App. D).**
- Cost C = ||z_hat_H - z_g||_2^2, terminal only, with z_g the encoded goal image.
- 300 sampled action sequences per iteration; top 30 are elites; initial mu = 0, variance 1; final plan is the distribution mean.
- Iterations: App. D says at most 30 on PushT and 10 elsewhere; App. B says 30 flat.
- Horizon 5, which is 25 environment steps with frame skip 5. All 5 planned actions run before replanning.
- Evaluation: 50-step budget, goal 25 steps later on the same dataset trajectory.

**Two-Room versus DINO-WM and PLDM (Fig. 6, values read from bar labels).**
- Success: LeWM 87, PLDM 97, DINO-WM 100, DINO-WM with proprioception 100, GCBC / GCIQL / GCIVL 100, random 0.
- Stated explanation (hedged as "possible"): low diversity and low intrinsic dimension make it hard to match a high-dimensional isotropic Gaussian, which may give a less structured latent. The conclusion repeats this as a limitation of SIGReg in simple environments.
- Tension inside the paper: the Table 3 caption says linear probes match PLDM (MSE 0.008, r 0.996), so the gap is "not due to a less informative representation" but to other factors such as dynamics or planning. Neither claim is tested. Neither mentions walls or L2 distance; that reading comes from later papers.
- App. E describes TwoRoom targets as in the other room, but App. F samples goals 25 steps ahead on one trajectory, so many test goals may be short range.
- TwoRoom data: 10,000 episodes, mean length 92, noisy door-seeking heuristic, 10 epochs.

**Paper versus code (`github.com/lucas-maes/le-wm`, main).**

| Item | Paper v3 | Code |
|---|---|---|
| Integration range | [0.2, 4] | `linspace(0, 3, 17)`, doubled trapezoid weights for symmetry |
| Knots | not stated | 17 |
| Weight window | "e.g." Gaussian, scale lambda | `exp(-t^2/2)` |
| Scaling | none shown | statistic multiplied by batch size |
| SIGReg weight | 0.1 | 0.09 |
| Projector | "1-layer MLP" with BN | Linear(192, 2048), BatchNorm1d, GELU, Linear(2048, 192) |
| History | 1 on TwoRoom | default 3; no TwoRoom override seen in the configs read |
| CEM iterations | 10 outside PushT | `n_steps: 30` in the shared solver config |

Directions are resampled every forward pass. `balestriero2025lejepa` recommends 17 points on [-5, 5] with 1024 slices, and its listing defaults to 256 slices, so three ranges are in circulation.

**This repo as of now.** `model.py` uses 256 projections, 17 knots, range [0, 5], and no BatchNorm before SIGReg. `plan.py` uses 6 CEM iterations, std 0.7, and a min-over-time cost rather than LeWM's terminal cost. Report these as deliberate changes or align them.

## Core JEPA and SIGReg

- `balestriero2025lejepa` (arXiv only). Argues that the isotropic Gaussian is the embedding distribution to aim for, and introduces SIGReg to reach it with one main hyperparameter and no teacher-student tricks. *Relation:* direct source of the regulariser; its recommended settings differ from LeWM's code.
- `assran2023ijepa` (CVPR 2023). Predicts representations of masked image blocks from a context block, avoiding pixel reconstruction and hand-made augmentations. *Relation:* the image-level ancestor; it uses an EMA target encoder, which LeWM removes.
- `assran2025vjepa2` (arXiv). Scales JEPA pretraining to internet video, then adds an action-conditioned stage used for robot planning toward image goals. *Relation:* large-scale version of goal-image latent planning; its energy is also a latent distance.
- `zhou2025dinowm` (ICML 2025). Learns dynamics on frozen DINOv2 patch features and optimises actions toward a goal's features at test time. *Relation:* LeWM baseline and the source of its planning setup; contrast with learning the encoder from pixels.
- `sobal2025pldm` (NeurIPS 2025). Compares offline reward-free RL with planning on a JEPA latent dynamics model, finding planning stronger on new layouts and stitching. *Relation:* origin of TwoRoom and closest end-to-end baseline. LeWM's reference list gives a workshop title for it; cite this version.
- `terver2026jepawms` (TMLR, May 2026). Ablates architecture, objective and planner choices for JEPA world models on simulated and real robot data, then combines the best. *Relation:* a source for planner settings; code is `facebookresearch/jepa-wms`.
- `hu2026scale` (arXiv; this is the "SCALE" paper). Adds a LeWM loss correlating latent pair distances with task-state distances, using privileged state only in training. *Relation:* same diagnosis as ours; its Table 1 gives LeWM a Spearman of .41 against .85 for DINO-WM on Two-Room. It needs ground-truth state, which our graph method does not.
- `feng2026epicwm` (arXiv). Trains a stochastic JEPA with flow matching and uses predictive variance as an exploration bonus inside CEM. *Relation:* an alternative LeWM extension aimed at partial observability; the arXiv comment says NeurIPS 2026, not confirmed elsewhere.

## Latent distance versus reachability (2026 search)

- `you2026ngc` (arXiv). Shows the LeWM L2 cost tracks path length only near the goal (Spearman .91 within 12 px, -.17 overall on TwoRoom) and creates false basins. Its graph shortest-path cost needs no retraining and lifts the farthest TwoRoom offset from 27% to about 95%. *Relation:* the nearest prior work; see "Read this first".
- `li2026trm` (arXiv). Trains a small temporal cost head on logged trajectories to rerank endpoints from a frozen LeWM, with TwoRoom cases controlled so wall crossing is not confounded with Euclidean distance. Success on its hard set rises from 7.0% to 96.7%. *Relation:* a learned-cost alternative; its same-room versus across-wall design is worth copying for the courses.
- `wang2026straightening` (ICML 2026). Adds a curvature penalty to JEPA training and argues that straighter latent paths make Euclidean distance closer to geodesic distance and planning better conditioned. *Relation:* the only peer-reviewed paper of the three; tests Wall and maze tasks, and fixes geometry at training time rather than planning time.

Also seen, arXiv record and abstract checked, left out of the .bib to keep to the requested number: 2608.12959 (single-author CPU study; L2 charges 4% less to cross the TwoRoom wall, a learned head 24% more), 2605.07278 (RC-aux), 2609.05461 (ARC-Bench, ranking audit), 2607.25337 (Temporal-Distance JEPA), 2609.37441 (AnisoWM), 2608.14125 (Traj-LeWM). Baseline numbers vary by protocol: LeWM on Two-Room is 87 in its paper, about 90 in NGC, 97.4 in 2607.25337 and 7.0 on TRM's hard set. Compare only within one protocol.

## World models and model-based RL

- `ha2018worldmodels` and `ha2018recurrent` (arXiv; NeurIPS 2018). Compress frames with a VAE, predict with a recurrent mixture model, and train a tiny controller, even inside the dream. *Relation:* the framing origin; cite the NeurIPS paper for the peer-reviewed version.
- `hafner2019planet` (ICML 2019). Learns a latent state-space model from pixels and runs CEM on it online. *Relation:* the earliest close match to latent CEM from pixels, but with reconstruction and rewards.
- `hafner2025dreamerv3` (Nature 2025). Trains an actor-critic purely on imagined latent rollouts, with one configuration across many domains. *Relation:* value learning in imagination is the neural analogue of our Dyna step. The Nature title differs from the arXiv title.
- `hansen2024tdmpc2` (ICLR 2024). Combines a decoder-free latent model with a learned value function, and plans with short-horizon sampling. *Relation:* shows a value estimate extending short planning horizons, which is our argument.
- `chua2018pets` (NeurIPS 2018). Uses an ensemble of probabilistic dynamics networks with CEM-based MPC. *Relation:* standard reference for CEM MPC and for handling model uncertainty.
- `schrittwieser2020muzero` (Nature 2020). Learns a model that predicts only value, policy and reward, and searches it with MCTS. *Relation:* planning in a learned latent where only decision-relevant quantities matter.
- `janner2019mbpo` (NeurIPS 2019). Justifies short model rollouts branched from real states to limit compounding error. *Relation:* grounds keeping imagined edges short and anchored to real latents.

## Dyna and RL foundations

- `sutton1990dyna` (ICML 1990 proceedings); `sutton1991dyna` is the short SIGART overview. An agent learns a model from experience and updates values from simulated experience alongside real experience. *Relation:* the named inspiration; cite 1990 for the idea.
- `sutton2018rl` (MIT Press 2018). The standard RL textbook. *Relation:* reference for value iteration and tabular Dyna-Q.
- `barkley2025stealing` (ICML 2025). Finds Dyna-style off-policy methods that work on Gym often degrade on DeepMind Control, with synthetic rollouts hurting most tasks. *Relation:* the honest caveat; imagined data can damage value learning, so ablate it.

## Graph planning and goal-reaching distances

- `savinov2018sptm` (ICLR 2018). Stores observations as nodes of a topological map and uses a learned reachability network for retrieval and edges. *Relation:* the classic case for connectivity over metric distance.
- `eysenbach2019sorb` (NeurIPS 2019). Uses a goal-conditioned value as an edge distance on replay-buffer states, then runs graph search. *Relation:* the closest classical design to latent graph planning, and sharper than NGC as a comparison because edges carry learned value.
- `wang2023qrl` (ICML 2023). Learns goal-reaching values in a quasimetric model, which enforces the triangle inequality while allowing asymmetry. *Relation:* a symmetric L2 cannot express one-way moves; relevant if the course has slopes or one-way geometry.
- `eysenbach2022contrastive` (NeurIPS 2022). Shows contrastive representation learning on trajectories acts as goal-conditioned value learning. *Relation:* a reward-free way to learn a reachability-aware metric. Author order differs between arXiv and NeurIPS; the .bib uses NeurIPS.

## Statistics and optimisation

- `rubinstein1997rare` (EJOR). Introduces the idea for rare-event simulation. `rubinstein1999cem` (MCAP) extends it to combinatorial and continuous optimisation. *Relation:* cite 1999 for CEM as an optimiser. LeWM cites the 2004 Rubinstein and Kroese book, which was not checked.
- `epps1983test` (Biometrika). A normality test comparing empirical and Gaussian characteristic functions under a weight. *Relation:* the statistic inside SIGReg.
- `cramer1936theorems` (J. London Math. Soc.). The source of the Cramer-Wold device: a multivariate law is fixed by its one-dimensional projections. *Relation:* the reason random slices suffice. Good primary source, also cited by LeWM.

## UNVERIFIED and caveats

**UNVERIFIED, left out of the .bib**
- LeCun 2022, "A Path Towards Autonomous Machine Intelligence". OpenReview returned 403 to a metadata request, so it was not retried. The likely Meta publication page gave 404, and a 2022 Meta post only announces an upcoming paper. Seen only second-hand: LeWM's reference list gives "version 0.9.2, 2022-06-27", "Open Review 62 (1), pp. 1-62"; a forum ID `BZ5a1r-kVsf` appears on third-party pages. Please open OpenReview by hand, then add it as `lecun2022path`.

**In the .bib, with gaps**
- `sobal2025pldm`: NeurIPS 2025 confirmed on neurips.cc, but the proceedings page was not found. Volume and pages are omitted because they came only from a third-party citation.
- `wang2026straightening`: ICML 2026 confirmed by icml.cc and the arXiv comment. No PMLR volume or pages checked.
- `terver2026jepawms`: confirmed on the jmlr.org/tmlr listing (May 2026); volume style not added.
- `hansen2024tdmpc2`, `savinov2018sptm`: confirmed on ICLR virtual pages, not OpenReview.
- `sutton2018rl`: MIT Press returned 403; confirmed on the author's official book page. ISBN omitted.
- NeurIPS 2018 and 2019 entries: official pages show volume but no page numbers, so pages are omitted.
- `hu2026scale`: the arXiv metadata title misspells "Reperesentation"; the paper itself and the .bib use "Representation".
- `assran2023ijepa`: the arXiv comment says ICCV, but CVF and Crossref confirm CVPR 2023.
- `assran2025vjepa2`: the arXiv API splits one author name; the paper shows Mojtaba Komeili.
- Only `wang2026straightening` among the three search additions is peer reviewed; `you2026ngc` and `li2026trm` are recent preprints, and their numbers are the authors' own.
