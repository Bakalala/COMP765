# Independent review of `report.tex`

Line numbers refer to `report/report.tex`. Each paragraph is one source line, so I quote the sentence as well.

## Verdict

**Not ready to submit. One interpretive conclusion is wrong, and the rest is fixable by editing.**

The numbers are clean: every figure I recomputed matches the CSVs. The code also matches the report, apart from some omissions. The serious problem is Pocket. The report says three times that the cost is the bottleneck there, and that latent distance "is not a route planner". The report never ran the cell that tests this. I ran it, without training anything: **the same latent cost with the learned predictor swapped for the true physics sinks 60 of 60 Pocket episodes.** The failure on Pocket is therefore mainly the predictor, not the cost. That changes Section 5.4 (Limits), the H2 interpretation, the Discussion and the Conclusion.

The second problem is that the causal story (H3, "bounces") is stated more strongly than the controls allow. The metric's "validation" is also weaker than claimed.

H1's rejection on Gate and Slalom survives every check I tried, and my additional runs strengthen it.

## What I ran

All runs were CPU only, with no training. Scripts stayed in the scratchpad, and this file is the only one written in the project.

| Check | Result |
|---|---|
| Recomputed Table 2 and all prose numbers from `episodes_seed*.csv`, `geometry_all.csv`, `ablations.csv`, `ghost_control.csv` | All match |
| Re-ran the aim and random planners with current code | 480 of 480 episodes match the CSVs exactly, so `golf.py` and `plan.py` are not stale for these deterministic planners |
| Re-ran `geometry()` on the 12 main checkpoints and the 3 ghost checkpoints | Main: max difference 8e-7. Ghost: wall gaps identical |
| Parameter count | 1,867,072 (encoder 1,307,424, predictor 559,648), so "1.9M" is correct |
| Missing planner cells (below) | Changes the Pocket conclusion |

Missing cells: H=1, the same `starts()`, three seeds, 20 episodes per seed. These are my runs and are not in `results/`.

| Dynamics / cost | Gate | Slalom | Pocket | Pocket from course start |
|---|---|---|---|---|
| Predictor / latent L2 (Table 2) | 1.00 | 0.97 | 0.48 | 0/3 |
| Predictor / latent L2 (my control, same harness) | not run | not run | 0.50 | 0/3 |
| **True physics / latent L2 (new)** | 0.97 ± 0.06 | 0.93 ± 0.12 | **1.00 ± 0.00** | 3/3 |
| **Predictor / true path length via linear probe (new)** | 0.95 ± 0.09 | 0.95 ± 0.09 | **0.53 ± 0.28** | 2/3 |
| True physics / true path length (oracle, Table 2) | 1.00 | 1.00 | 1.00 | 3/3 |

The control row reproduces the published value, so the harness is sound. The probe row adds some decoding noise, so rely on the true-physics row. It is the clean one.

One-shot prediction error supports this. Decoded through the probe, the median error on shots that touch an interior wall is 0.074 on Pocket, against 0.051 on Gate and 0.058 on Slalom. The hole radius is 0.04, and the corridor beside the Pocket walls leaves the ball centre about 0.034 of clearance.

To reproduce the key cell, add this to `plan.py` or `evaluate.py`:

```python
def physics_latent_planner(model, course, goal_pos, generator):
    goal = model.encode(goal_pos[None])[0]
    def choose(pos):
        def cost(actions):
            shots = actions[:, 0].numpy().astype(np.float64)
            final, _ = simulate(course, np.repeat(pos[None], len(shots), 0), shots)
            return (model.encode(final) - goal).pow(2).sum(-1)
        return cem(cost, 1, generator=generator)[0].numpy()
    return choose
```

## Blocking errors

### B1. Pocket: "the cost is the bottleneck" is wrong

The oracle differs from latent L2 in both dynamics and cost, so comparing them cannot isolate the cost. The missing cell shows the cost is sufficient there. Run the cell above, save it to `results/`, add it as a row in Table 2, then make these edits.

**Line 114.** "and the pair separates errors of the model from errors of the cost."
Replace with: "and the pair shows how much the cost matters when the model is perfect. To separate model error from cost error for the learned planner, we also run latent L2 through the true physics."

**Line 178, whole paragraph.** Replace with:
"\paragraph{Pocket.} Latent L2 sinks only 48\% here, and 0 of 3 from the course start, while the oracle reaches 100\%. The oracle changes both the dynamics and the cost, so we swap them one at a time. The same latent cost through the true physics sinks 100\% of episodes, while the true path-length cost through the learned predictor, decoded with a linear probe, sinks only 53\%. The latent distance is therefore good enough here, as its wall gap suggests ($6.99\pm0.51$), and the bottleneck is one-shot prediction: shots that touch a wall are predicted worse on Pocket than on Gate or Slalom (median error 0.074 against 0.051 and 0.058). Correlation near the hole, meaning path length under 0.2, is also lowest here ($0.41\pm0.13$)."

Delete "Figure~\ref{fig:maps} shows no clear route to the opening." The Pocket panel does darken towards the right-hand side, and a cost with no route could not reach 60 of 60.

**Line 185.** "With the oracle at 100\%, a better cost clearly exists, but ours did not find it."
Replace with: "The graph cost is also evaluated on the predictor's imagined outcome, so these results do not show that the graph itself is at fault: on Pocket a predictor error limits any cost, as the 53\% for the true path-length cost shows."

H2 "not supported" still stands as a result. Its explanation does not. Running the shot-graph cost through the true physics on Pocket would settle it, and needs no training.

**Line 190.** "Why did the graph fail on Pocket? We can offer candidates but have not tested them."
Replace with: "Why did the graph fail on Pocket? The main limit is probably shared with latent L2: the imagined outcome of each candidate shot is inaccurate there. Beyond that we offer candidates that we have not tested."

**Line 196.** "Latent distance is still not a route planner, as Pocket shows, and our Dyna-style graph did not fix that."
Replace with: "On Pocket the planner still fails, but the fault lies mainly with the predictor: the same latent cost succeeds with true dynamics, and our Dyna-style graph did not fix it."

Also on line 196, change "whether a learned cost can close the gap to the oracle" to "whether better prediction of wall-contact shots can close the gap to the oracle".

**Line 22.** "A pocket course still defeats latent planning (48\%), and a Dyna-style planner ... did not clearly fix it."
Add after it: "On that course the limit is the learned predictor rather than the latent cost, which reaches 100\% with true dynamics."

### B2. H3 is stated beyond what the controls isolate

The ghost control changes three things at once, not one:

1. Shots can cross the strip.
2. Bounces disappear.
3. The set of states changes. On the ghost course, `sample_free` places balls on the wall strip, so the training states are connected. On the real course the strip is absent from the data.

No experiment separates blocking from bouncing, so "bounces" in the title and H3 is a guess about mechanism (restitution 0 would test it). The shot-power sweep does help with the third item, because it keeps the same start distribution. The report does not say so, and it should. All of this is one course, three seeds, post hoc.

**Line 22.** "We conclude that walls enter the latent space through the transitions rather than the pixels: shots that span and bounce off obstacles make them visible to the predictor."
Replace with: "In this exploratory, post hoc analysis, the evidence favours walls entering the latent space through the transitions rather than the pixels, though we did not separate blocking from bouncing."

**Line 159.** "What the wall \emph{does} matters, and how it looks does not."
Replace with: "On Gate, then, appearance alone is not enough. The ghost also differs from the real wall in the states it allows, since balls can rest on the strip, so this control alone cannot separate blocked transitions, bounces, and missing states. The shot-power sweep below keeps the start distribution fixed, which argues against missing states as the whole explanation."

**Line 188.** "A wall is not an object in this model. It is a discontinuity in what shots can do, so it appears exactly when the data contains shots that reveal it."
Replace with: "Our results suggest that a wall enters this model as a discontinuity in what shots can do, rather than as a visual object, and that it appears more strongly when more of the data reveals it."

**Line 196.** "Controls show why: a model trained on shots that cross and bounce off walls builds them into its geometry, whereas..."
Replace "Controls show why:" with "Exploratory controls suggest why:".

**Line 198.** "because a simple idea settled a question that was otherwise only a story."
Replace with: "because a simple idea turned a story into something I could test."

### B3. The metric's validation is weaker than claimed

- **The stand-in is a tautology.** If the latent is the position, every pair in `wall_gap` has length exactly 0.09, so the ratio is 1 whatever pairs are selected. I get 1.0000 on every wall course and seed. Line 109's "reads 1.00" therefore cannot detect a selection bug in the final metric.
- **The ghost is circular.** The metric was revised because of the ghost result, then the ghost's near-1 value is reported as a finding and as validation.
- **"Indistinguishable" is not accurate.** Ghost seeds are 0.975, 0.958, 0.961, all below 1 by more than their SD (0.965 ± 0.009).

**Line 109.** "and the final metric reads 1.00 for a stand-in model whose latent is simply the ball position, at both real and ghost walls."
Replace with: "The final metric reads 1.00 for a stand-in whose latent is the ball position, but this is only a sanity check, since any isometric latent gives exactly 1. We revised the metric after seeing the ghost result, so the ghost value is not an independent test of it."

**Line 159.** "indistinguishable from no wall (Figure~\ref{fig:profile})"
Replace with: "close to the no-wall value of 1 (Figure~\ref{fig:profile})".

**Line 193.** "the metric was validated on a stand-in and on the ghost, but independent replication is needed."
Replace with: "the metric was revised after seeing the ghost result and has no independent validation."

### B4. Lines 102 and 182: the ablation text hides results and mislabels a failure

Three problems:

- At λ=0.01 the "failed" seed is not the constant collapse of line 102. Its latents have SD about 0.6 but effective rank 1.00 (my check, 4,000 positions). That is dimensional collapse, with BatchNorm present.
- The other seed gives the best geometry in the whole project (Spearman 0.90, wall gap 25.2, success 100%), also at effective rank 1.13. The report omits this. The baseline effective rank is 4.7.
- Wall gap does not track success. λ=1 has gap 2.57 and 95%. Half power has gap 4.04 and 37%. Pocket has 6.99 and 48%. Readers of Figure 4 will assume the opposite.

**Line 182.** "At $\lambda{=}1$ success stays at 95\% but the wall gap drops to $2.57\pm0.11$, because a stronger Gaussian target flattens the geometry. At $\lambda{=}0.01$ the two seeds split: one model failed entirely, giving 50\% on average. Two seeds cannot characterize that instability, but it is consistent with the collapse described above."
Replace with:
"At $\lambda{=}1$ success stays at 95\% but the wall gap drops to $2.57\pm0.11$ and path-length correlation to $0.54\pm0.03$, possibly because a stronger Gaussian target flattens the geometry. Wall gap therefore does not predict planning success on its own: half-power models have a larger gap (4.04) and far lower success (37\%). At $\lambda{=}0.01$ the two seeds split. One failed entirely (0\%, gap 0.50) and the other gave the largest gap and correlation of any Gate model (25.2 and 0.90, 100\%). Both have nearly one-dimensional latents, so this is dimensional collapse rather than the constant encoder above, and BatchNorm does not prevent it. Two seeds cannot characterize this."

**Line 102.** "Normalizing the projection head amplifies the small signal and removed the collapse."
Replace with: "Normalizing the projection head removed this constant collapse, though not the low-rank failure seen at small $\lambda$ below."

Also on line 102, "BatchNorm proved essential" rests on one unrecorded run: nothing in `results/` shows it. The zero gradient at an all-zero embedding is correct (I checked: exactly 0, against about 0.06 for a constant 0.5 embedding). "The cause is visible" is still an inference. Change it to "A likely cause is visible". "About twelve of 4,096 pixels" is the disc area; the rendered ball changes 15 to 20 pixels, so write "about 15".

### B5. Lines 109 and 198: the two accounts of the metric bug disagree

Line 109 says "The ghost-wall control exposed both". Line 198 says the ghost "also exposed two mistakes in my own metric: an open course with no walls scored 10". An open course is not the ghost control, so one of these is wrong. No project file I could read records the score of 10 or the earlier metric versions: not `results/`, the tests, or the station script. I did not run git, so it may be in history. Current code returns NaN on Open.

**Line 109.** "The ghost-wall control exposed both, and"
Replace with: "Sanity checks on the open course and the ghost wall exposed these, and".

**Line 198.** "It also exposed two mistakes in my own metric: an open course with no walls scored 10, which was the warning I needed."
Replace with: "Checking the metric where it should read nothing also exposed two mistakes in it: an open course with no walls scored about 10, which was the warning I needed."

**Line 198.** "how much a single BatchNorm layer matters"
Replace with: "how much BatchNorm matters". `model.py` has two BatchNorm layers in the head, and no run isolates one.

### B6. Line 94 and Table 1: the "Shared" row overstates, and deviations are missing

All listed values match the code. The word "Shared" is too strong, and real differences are omitted:

- Per the notes, LeWM's code doubles the trapezoid weights. `model.py` does not.
- The statistic is multiplied by batch size, which here is 512 because `now` and `after` are concatenated. LeWM applies SIGReg per time step. Equation 2 shows neither the multiplier nor the [0, 3] range.
- Per the notes, LeWM runs all 5 planned actions before replanning. This project replans after every shot.
- CEM initial std is 0.7 here, against variance 1 in LeWM.

Add table rows: "Replanning & after 5 actions & every shot" and "CEM initial spread & variance 1 & std 0.7".

**Line 53**, after "and $w$ a Gaussian window.", add: "In code the integral runs over $t\in[0,3]$ using symmetry, and the result is multiplied by the number of embeddings."

**Line 78**, add: "We apply SIGReg to the current and next embeddings of a batch together, and we do not double the trapezoid weights, so $\lambda$ is not strictly comparable with LeWM's."

**Line 94.** Change "Shared" to "Same values".

### B7. Lines 116 and 175: the power sweep's premises are unmeasured and the budget is unmatched

Halving velocity quarters the range, because stopping distance scales with v². Maximum roll is 0.853, 0.213, 0.053. So shot limits of 20 and 40 are not matched budgets; matching range would need 40 and 160.

From the Slalom start, 27 perfect maximum-range shots are needed at quarter power. 26 of 60 Slalom starts need more than half the budget even with perfect shots. A quarter-power shot also cannot span the wall at all, since the free gap is 0.066, and the wall-gap pair separation (0.09) exceeds the maximum roll. "How much of the data involves walls" is asserted but never reported. I measured it.

**Line 175.** "Shot power then varies how much of the data involves walls."
Replace with: "Shot power then varies how much of the data involves walls: an interior wall changes the outcome of 23\%, 7\%, and 2\% of random shots on Gate at full, half, and quarter power, and 42\%, 14\%, and 4\% on Slalom. It also shrinks range by 4 and 16 times, so the two effects are confounded."

**Line 175.** "and success (17\% and 12\%) partly reflects a one-shot planner that can only see a short distance ahead."
Replace with: "and success (17\% and 12\%) partly reflects a one-shot planner that can only see a short distance ahead, and a shot limit that grew by 4 while range fell by 16."

Add the same point to the Figure 4 caption (line 171).

### B8. Line 28: stronger than the notes

"so a planner can press against the wall dividing the rooms \cite{you2026ngc,hu2026scale,li2026trm}". The notes support "tracks path length only near the goal" and "false basins" for `you2026ngc` only. For `hu2026scale` and `li2026trm` they support only L2 failing as a reachability proxy. Nothing mentions pressing against a wall.

Replace the sentence's second half with: "on a two-room navigation task, latent distance is a poor proxy for reachability \cite{you2026ngc,hu2026scale,li2026trm}: it tracks path length only near the goal and creates false basins \cite{you2026ngc}."

### B9. Line 22: two misleading phrases in the abstract

- "test that claim": the report does not test it, since it uses a different domain and never runs the two-room task. Replace with "ask whether it holds there".
- "one-shot planning ... sinks 100\% and 97\%": this reads as holing out in one shot. It is one-step lookahead with replanning, taking 2.2 and 3.4 shots on average. Replace with "one-step-lookahead planning" here and on line 156.

### B10. Line 200: the link does not match the branch

The URL points to `tree/main/putt_world`, but this work is on branch `putt-world`. `report.tex`, `report.pdf`, and some edits were uncommitted at session start. Merge before submitting, or the link may not show this work. "Code, data" also overstates slightly: checkpoints are git-ignored, so write "Code, result files, and the interactive demo".

## Findings by requested category

**1. Numbers: no mismatches.** Everything in the abstract, Table 2, prose and captions matches the recomputation, including derived values (0.925 prints as 93%). Three small items:

- Slalom H=1 is 0.97 in Table 2 and 0.98 in Figure 4, because `run_ablations.py` re-evaluates the same models with a different random stream. Add a note to the caption.
- Line 156 quotes straight-line correlations of 0.77 and 0.77 correctly.
- Line 175's "about 0.2" holds for all four values (0.17 to 0.23).

**2. Implementation: confirmed, apart from B6.**

- Confirmed: encoder, predictor, M=1024, 17 knots, [0,3], λ=0.09, CEM 300/30/6, 10/20/40 shot limits, starts, 300 clusters, 20,000 k-means points, restitution, course descriptions, metric definitions, 60,000 shots, 4,000 steps, batch 256.
- **Please confirm one thing.** `docs/related_work_notes.md` (saved 11:01) records 256 projections, range [0,5], and no BatchNorm "as of now". The first checkpoint finished at 11:08. The checkpoints do have BatchNorm, but the logged SIGReg value (17.98) fits both settings, so they cannot prove M=1024 and [0,3] were used. `model.py` is also newer than the results.
- Training time: checkpoint timestamps give 4.3 to 5.3 minutes for main models and about 6 for ghost models, so "four to five minutes" is more accurate.
- Line 114: "takes the best step along the rollout" is wrong. The anytime planner scores a sequence by its best imagined step, then executes the first action. Write "scores a rollout by its best step".
- Line 105: the graph cost is a softmax over the 4 nearest clusters, not $V$ at one cluster. Worth a clause.
- Table 2 and line 114: call the nearest-neighbour planner "our reimplementation" and state 1,500 nodes.

**3. Citations.**

- Every `\cite` key exists. Four bib entries are unused and harmless, since BibTeX omits them: `feng2026epicwm`, `rubinstein1997rare`, `ha2018recurrent`, `sutton1991dyna`.
- Beyond B8, three stretches:
  - Line 41 calls Barkley "relevant to our negative result", but replay and model edges did not differ, so synthetic experience did not visibly hurt. Soften it, or delete the matching clause on line 188.
  - Line 175's "where actions are small steps" is not in the notes. Give a source or hedge it.
  - Table 1's $224^2$ is not in the notes either. Check it against the paper.
- Line 39 quotes -0.17 alone. The notes also give 0.41 from SCALE and warn against comparing across protocols. Add a clause.

**4. Overclaims.** Beyond B1 to B4:

- Line 156: "Spearman ... slightly above" is presented as evidence, but straight-line and path-length distance correlate at 0.92 on Gate and 0.93 on Slalom, so these correlations can barely separate them. The Gate margin is 0.014. Say so and let wall gap carry the argument.
- Line 156: "The learned \emph{cost} is therefore doing the work" is now supported for Gate and Slalom by the new cell. Cite it.
- Line 180: the "sinks early" explanation is hedged but untested. It also understates the data: on Gate only 36% of shots starting in the hole stay there, so hole-leaving is in the data, not merely absent.
- Line 32: "evidence that action scale shapes latent geometry" should read "suggestive evidence", given B7.
- Negative results are otherwise reported fairly. The post hoc finishing planner is labelled.
- **Hypothesis order is honest.** The notes predate the first checkpoint and already frame L2 failure and the graph, which supports H1 and H2 being planned. Line 30 flags H3 as post hoc. Only the abstract and title do not. B2 fixes the abstract.

**5. Contradictions, references, requirements.** Contradictions are B1 and B5. Other items:

- Figure 1 (`fig:pipeline`) is never referenced in the text.
- `sec:gap` is unused.
- "Equation~2" is hard-coded (currently correct).
- `algorithm` and `algpseudocode` are loaded but unused.
- No broken references in the PDF.
- Line 122: "where a ball almost never rests" is imprecise. Starts are uniform there, so these positions appear in training, but a shot never ends there. Write "where a shot never ends".
- Line 178: "near the hole" was undefined; the B1 text fixes that.

| Requirement | Status |
|---|---|
| Goal or hypothesis | Met |
| Related work, own words, current, compared with class work | Met; the class comparison is brief (Assignment 1, Dyna) |
| Method recap with added detail and technical support | Met, subject to B6 |
| Data and findings | Met, subject to B1 |
| Conclusion, lessons, surprises, changed assumptions, open questions | Met; open questions need the B1 update |
| 6 to 8 pages, two columns | Met: 7 pages, two columns |

**6. Reflection against project files.**

| Event | Finding |
|---|---|
| Tabular Dyna project | Supported. `../dynaputt/deliverables/analysis_notes.md` says "most of the gain on short routes depends on the optimistic table", and five-seed noise is recorded. Those notes call the study exploratory, so line 188 should too |
| BatchNorm collapse | Partly supported. The notes and a code comment show a no-BatchNorm version and the collapse rationale. "Prediction loss exactly zero and SIGReg frozen" has no record. "Single layer" is wrong (B5) |
| Open course scored about 10 | Not in any file I could read, and the attribution conflicts with line 109 (B5) |
| Recent close paper | Supported by the notes (`you2026ngc`). They predate all training, so "it turned out" is fine, but the lesson about checking early should not imply it was found late |
| "The first distance maps showed the opposite" | No saved artefact, plausible given Figure 2 |
| Assignment 1 claims (line 35) | Supported: DreamerV3 profile and cart-pole model with a short-horizon lookahead controller. It was a bonus experiment |

## Non-blocking suggestions

1. **Add blocked-start results to line 156.** The 100% and 97% include starts with a clear line to the hole, 23 of 60 on Gate and 15 of 60 on Slalom. From blocked starts only, latent L2 sinks 37/37 and 43/45, against 21/37 and 6/45 for true physics with a straight-line cost, and 5/37 and 1/45 for aim. This strengthens your main claim.
2. **Report the course-start result.** Only 3 episodes per course start there, and on Pocket latent L2 gets 0/3 while the oracle gets 3/3. The headline 48% comes entirely from random starts.
3. **Use the near-hole correlations at reduced power.** Gate near-hole Spearman rises to 0.95 and 0.97 at half and quarter power while global correlation collapses. That resembles You et al.'s near-goal pattern and is better support for the reconciliation than anything now in the text. Slalom at half power (0.67) is an exception to mention.
4. **Add a Limitations line** that the graph was never tested with true dynamics, if you do not run that cell.
5. **Cite arXiv 2608.12959** from the notes (L2 charges 4% less to cross the TwoRoom wall). It is the most direct comparison to wall gap, though it needs a bib entry and a check.
6. **Mention LeWM's own 87% on two rooms,** which gives context for "fails around walls".
7. **Note that the probe is in-sample** (line 153): fitted and scored on the same grid.
8. **"Dyna-style" is generous** for value iteration on a fixed graph with no acting-learning loop. Line 190 half admits this; say so once in Section 3.4.
