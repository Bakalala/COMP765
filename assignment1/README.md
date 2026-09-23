# Assignment 1 - World Models and Cart-Pole Control

## Reproduce the experiments

```bash
cd TeachingCartpole
python3 -m pip install -r requirements.txt
python3 lqr_starter.py
python3 experiment.py
```

The GUI simulation defaults to the hybrid controller (LQR near upright and
energy shaping away from upright):

```bash
python3 cartpole_sim.py --initial-offset 0.1
python3 cartpole_sim.py --initial-offset 3.141592653589793
```

The experiment harness is headless and saves its summary to
`TeachingCartpole/results/experiment_results.csv`.
