"""Deterministic nonlinear DoubleCartpole bonus experiments.

Run ``python3 double_experiment.py`` from this directory. Output is written
to results/double_cartpole_results.csv.
"""

from pathlib import Path
import csv
import math

import numpy as np

from cartpole_control import Controller
from cartpole_envs import DoubleCartPole
from double_cartpole_control import A, B, K, wrap_to_pi


DT = 0.005
DURATION = 20.0


def run_trial(offset, force_limit=40.0, duration=DURATION):
    # Pass a new list each time. The upstream constructor mutates x_init,
    # including its default argument, when applying initial_offset.
    initial = [0.0, 0.0, 0.0, 0.0, math.pi, math.pi]
    env = DoubleCartPole(x_init=initial, initial_offset=float(offset))
    controller = Controller(hybrid=False, double_force_limit=force_limit)
    samples = []
    capture_time = None
    for step in range(int(duration / DT)):
        state = env.get_state().copy()
        force = controller.compute_control(state)
        next_state = env.step(force, DT).copy()
        angle1 = abs(wrap_to_pi(next_state[4] - math.pi))
        angle2 = abs(wrap_to_pi(next_state[5] - math.pi))
        samples.append((step * DT, next_state[0], next_state[2], next_state[3],
                        angle1, angle2, force))
        if (capture_time is None and max(angle1, angle2) < 0.1
                and max(abs(next_state[2]), abs(next_state[3])) < 0.25):
            capture_time = step * DT

    data = np.asarray(samples)
    tail = data[-int(5.0 / DT):]
    return {
        "initial_offset_rad": float(offset),
        "force_limit_N": float(force_limit),
        "success": bool(np.max(tail[:, 4:6]) < 0.1
                        and np.max(np.abs(tail[:, 1])) < 0.5),
        "capture_time_s": "" if capture_time is None else capture_time,
        "tail_max_pole1_error_rad": float(np.max(tail[:, 4])),
        "tail_max_pole2_error_rad": float(np.max(tail[:, 5])),
        "tail_max_cart_position_m": float(np.max(np.abs(tail[:, 1]))),
        "max_cart_position_m": float(np.max(np.abs(data[:, 1]))),
        "peak_force_N": float(np.max(np.abs(data[:, 6]))),
    }


def main():
    controllability = np.column_stack([np.linalg.matrix_power(A, k) @ B
                                       for k in range(6)])
    if np.linalg.matrix_rank(controllability) != 6:
        raise RuntimeError("DoubleCartpole linearization is not controllable")
    equilibrium = np.array([0.0, 0.0, 0.0, 0.0, math.pi, math.pi])
    check_env = DoubleCartPole(x_init=equilibrium.tolist())
    step = 1e-6
    finite_a = np.column_stack([
        (check_env.dynamics(0, equilibrium + np.eye(6)[j] * step).ravel()
         - check_env.dynamics(0, equilibrium - np.eye(6)[j] * step).ravel())
        / (2 * step) for j in range(6)
    ])
    check_env.u = step
    plus = check_env.dynamics(0, equilibrium).ravel()
    check_env.u = -step
    minus = check_env.dynamics(0, equilibrium).ravel()
    finite_b = (plus - minus) / (2 * step)
    jacobian_error = max(np.max(np.abs(finite_a - A)),
                         np.max(np.abs(finite_b - B.ravel())))
    if jacobian_error > 2e-7:
        raise RuntimeError(f"DoubleCartpole Jacobian mismatch: {jacobian_error}")
    rows = [run_trial(offset) for offset in
            (math.pi / 40, 0.01, 0.1, math.pi / 8, 0.42,
             math.pi / 4, math.pi)]
    output = Path(__file__).parent / "results" / "double_cartpole_results.csv"
    output.parent.mkdir(exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys(),
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)
    print("K =", K)
    print("max finite-difference Jacobian error =", jacobian_error)
    print("closed-loop poles =", np.linalg.eigvals(A - B @ K))
    print("Wrote", output)


if __name__ == "__main__":
    main()
