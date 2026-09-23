"""Headless, deterministic evaluation for Assignment 1.

Run with ``python3 experiment.py``. The script writes the exact trial summary
used in the report to ``results/experiment_results.csv``.
"""

from pathlib import Path
import csv
import math

import numpy as np

from cartpole_control import Controller, wrap_to_pi
from cartpole_envs import CartPole


DT = 0.005
DURATION = 20.0


def run_trial(offset, hybrid=False, duration=DURATION):
    env = CartPole(initial_offset=float(offset))
    controller = Controller(hybrid=hybrid)
    samples = []
    captured_at = None

    for step in range(int(duration / DT)):
        state = env.get_state().copy()
        force = controller.compute_control(state)
        next_state = env.step(force, dt=DT).copy()
        angle_error = abs(wrap_to_pi(next_state[3] - math.pi))
        samples.append((step * DT, *next_state, force, angle_error))
        if captured_at is None and angle_error < 0.10 and abs(next_state[2]) < 0.25:
            captured_at = step * DT

    data = np.asarray(samples)
    tail = data[int(0.75 * len(data)) :]
    return {
        "offset_rad": float(offset),
        "controller": "hybrid" if hybrid else "lqr",
        "success": bool(np.max(tail[:, 6]) < 0.10 and np.max(np.abs(tail[:, 1])) < 0.50),
        "capture_time_s": captured_at if captured_at is not None else "",
        "max_angle_error_rad": float(np.max(data[:, 6])),
        "tail_max_angle_error_rad": float(np.max(tail[:, 6])),
        "tail_max_cart_position_m": float(np.max(np.abs(tail[:, 1]))),
        "max_force_N": float(np.max(np.abs(data[:, 5]))),
    }


def main():
    offsets = [0.01, 0.1, math.pi / 8.0, math.pi / 4.0]
    rows = [run_trial(offset, hybrid=False) for offset in offsets]
    rows.append(run_trial(math.pi, hybrid=False))
    rows.append(run_trial(math.pi, hybrid=True))

    output = Path(__file__).parent / "results" / "experiment_results.csv"
    output.parent.mkdir(exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(row)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
