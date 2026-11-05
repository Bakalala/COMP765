# Dyna Putt: status and remaining work

Course deadline: **December 3, 9pm**, report PDF to the My Courses assignment
"Final Project", 6-8 pages. The interactive session is the same day: seated station,
1-2 slides or videos, then Q&A.

The original seven-task plan was reviewed and carried out in one pass. Findings are in
[`deliverables/plan_review.md`](deliverables/plan_review.md). What remains is below.

## Project in one paragraph

The project adapts Sutton's Dyna architecture to deterministic tabular mini-golf. It
asks whether Dyna-Q learns a course in fewer real shots than Q-learning, and whether
more planning helps. The primary comparison is 0, 5, and 20 planning updates on `easy`
and `wall_gate`, with five seeds and 350 episodes. Extended studies on 100 separate
seeds cover planning budget, random streams, initial Q values, and route length.

## Source of truth

| Data | Location | Use |
|---|---|---|
| Primary | `results/` | The planned experiment |
| Extended | `results/extended/` | Robustness and exploratory studies |
| Archive | `results/rng_coupled_baseline/` | Earlier shared-stream run, for reference |

Never pool these or plot them as one procedure. Numbers and their definitions are in
`deliverables/analysis_notes.md`. Always verify against the CSV before quoting.

Findings that must not be lost:

- The archived Easy reversal is five-seed noise. Do not call it protocol sensitivity.
- "Final greedy success" means checkpoints after episodes 301-350.
- The zero Q table is optimistic, and that strongly affects the planning benefit.
- The initial-value and route-length studies are exploratory.

## Done

| Original task | Result |
|---|---|
| 1. Implementation audit | `deliverables/audit.md`; two demo bugs fixed in `index.html` |
| 2. Data analysis | `deliverables/analysis_notes.md`; extended studies added |
| 3. Related work | `deliverables/literature_notes.md`, citations checked |
| 4. Reconcile notes | Notes, READMEs, outline, and summarizer now agree |
| 5. Report | `deliverables/report/report.pdf`, 6 pages, rebuilt and page-checked |
| 6B. Station plan | `deliverables/demo_plan.md`, `deliverables/station_slides.pdf` |

The separate refresh files from the original plan were not created; findings went
straight into the existing notes.

## Student actions (do not delegate)

1. Read the report and confirm you can explain the method, data, and caveats, including
   why the later hypotheses are labeled exploratory.
2. Check that the reflection in Section 8 matches your own view. It was drafted by an
   assistant from the project history at your request.
3. Add peers' papers to Section 2.4 if you have them. The requirements ask for this, and
   they are not recorded in this repository.
4. Check the course's rules on AI assistance, and disclose it if they require that.
5. Run `python play.py` on the Mac, using the checklist in `deliverables/demo_plan.md`.
6. Submit the PDF, and bring the laptop to the session.

## Optional LLM tasks

Run these one at a time.

### Task A: independent report review

**Inputs:** `deliverables/report/report.pdf`, `build_report.py`, `results/`,
`deliverables/analysis_notes.md`.
**Output:** `deliverables/report_review.md`. Edit nothing else.

```text
Review the Dyna Putt report against the code and result files in this folder. Do not
edit the report. Check every number in the prose against results/summary.csv,
results/extended/summary.md, and the CSVs. Check that primary, extended, and archived
data are never pooled, that exploratory hypotheses are labeled, that every in-text
citation has a reference, and that the PDF is 6-8 pages with no clipping. Do not change
or comment on the student's own reflection except to flag factual conflicts with the
data. For each failure give the page and the exact correction.
```

**Done when:** every prose number is marked verified or wrong, and each failure has a
page and correction. A person then decides which corrections to apply in
`build_report.py`. Rerun this task after any change to the numbers.

### Task B: optional extension, prioritized sweeping

Only if time allows. **Output:** a new `run_prioritized.py` and a new
`results/prioritized/`. Do not change existing files or data.

```text
Implement tabular prioritized sweeping as a new agent class, without changing existing
agents. Compare it with uniform Dyna-Q on easy, wall_gate, and the ladder courses at
matched updates per shot, on seeds 1000-1099. Report threshold, real shots, greedy-stable
episode, and total updates, with paired bootstrap intervals. Label this as a test of the
prediction in Section 6 of the report. Report the result whichever way it comes out.
```

**Done when:** results reproduce from one command and the comparison is matched on
updates. Add to the report only if the page limit allows.

## Rules for any assistant

- Read the relevant files before making claims. Never invent measurements, citations,
  implementation details, or student reflections.
- Do not invent peers' papers or claims about them.
- Do not claim the demo passed the Mac check until the student says so.
- Do not rewrite `results/episode_results.csv` or the archive.
- Keep the project scale: an inspectable tabular adaptation, not a neural world model.
