# Assignment 1 - World Models and Cart-Pole Control

## Reproduce the experiments

```bash
cd TeachingCartpole
python3 -m pip install -r requirements.txt
python3 lqr_starter.py
python3 experiment.py
python3 double_experiment.py
python3 world_model_bonus.py
```

The GUI simulation defaults to the hybrid controller (LQR near upright and
energy shaping away from upright):

```bash
python3 cartpole_sim.py --initial-offset 0.1
python3 cartpole_sim.py --initial-offset 3.141592653589793
python3 cartpole_sim.py --env DoubleCartpole --initial-offset 0.1
```

The experiment harness is headless and saves its summary to
`TeachingCartpole/results/experiment_results.csv`.

The DoubleCartpole bonus uses its own six-state LQR controller in
`TeachingCartpole/double_cartpole_control.py`. The GUI selects it automatically
for `--env DoubleCartpole`; `double_experiment.py` runs the headless 20-second
offset trials and writes `results/double_cartpole_results.csv`.

The second bonus experiment in `world_model_bonus.py` collects simulator
transitions in `(next state, state, force)` order, fits a compact dynamics
model, tests held-out one-step and open-loop prediction, and evaluates a
short-horizon lookahead controller. It writes the dataset, fitted coefficients,
prediction trace, and metrics under `TeachingCartpole/results/`. Run
`python3 build_report.py` from this folder to rebuild the PDF report.
