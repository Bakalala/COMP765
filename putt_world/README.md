# Putt World

A small LeWorldModel for continuous mini-golf. A JEPA world model learns the game from
pixels of random shots, with no rewards, rules, or wall positions, then plans shots by
searching in its latent space. This is the COMP 765 course project. The earlier tabular
Dyna project is in [`../dynaputt/`](../dynaputt/).

**Main finding:** the model's latent distance respects walls, and the evidence points to
transitions rather than pixels: shots blocked by a wall reveal it, and bouncing strengthens
the effect. A wall that is drawn but has no physics leaves almost no trace.

- Report: [`report/report.pdf`](report/report.pdf)
- Results: [`results/summary.md`](results/summary.md)
- Station materials: [`station/`](station/)

## Play the demo

Open [`demo/index.html`](demo/index.html) in a current browser. It needs no server, no
packages, and no network. Drag back from the ball and release to putt, press **Let the
world model plan** to watch the agent, and turn on **Mind view** to see its distance map.

## Reproduce

```bash
uv venv --python 3.13 .venv
```

```bash
uv pip install --python .venv/bin/python -r requirements.txt
```

Then run these from this folder, in order. Training takes about an hour on an Apple M4,
and the ablations about another hour.

```bash
.venv/bin/python train_models.py
```

```bash
for seed in 0 1 2; do .venv/bin/python evaluate.py $seed; done
```

```bash
.venv/bin/python evaluate.py finish
```

```bash
.venv/bin/python evaluate.py geometry
```

```bash
.venv/bin/python run_ablations.py ghost
```

```bash
.venv/bin/python run_ablations.py
```

```bash
.venv/bin/python factorial.py
```

```bash
.venv/bin/python analyze.py
```

Build the report with `tectonic report.tex` in `report/`. Run the tests with:

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests
```

Checkpoints are not stored in git. `export_demo.py` rebuilds `demo/model_data.js` from
them, and `node demo/parity_test.mjs` checks the browser physics against Python.

## Files

| File | Purpose |
|---|---|
| `golf.py` | Physics, courses, ground-truth path lengths |
| `model.py` | Encoder, predictor, SIGReg, training |
| `plan.py` | CEM, baselines, graph planners, episode loop |
| `evaluate.py` | Planner evaluation, latent geometry, wall gap |
| `run_ablations.py` | Shot power, SIGReg weight, dataset size, ghost and dead walls |
| `factorial.py` | Dynamics and cost swapped, prediction error, wall involvement |
| `analyze.py` | Summary tables and figures |
| `docs/related_work_notes.md` | Verified notes on related work |
