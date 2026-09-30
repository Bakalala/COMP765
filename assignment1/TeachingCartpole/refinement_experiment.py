"""Additional range and tuning experiments, without changing the final controller.

Run with ``python3 refinement_experiment.py``. Every trial uses the supplied
nonlinear simulator, a 30 N cap, dt=0.005 s, and a 20 s duration. The range scan
changes only the initial angle; it is not a proof of a region of attraction.
"""

import csv
import math
from pathlib import Path

import numpy as np

from cartpole_control import Controller, wrap_to_pi
from cartpole_envs import CartPole
from lqr_starter import A, B, lqr


DT = 0.005
DURATION = 20.0
RESULTS = Path(__file__).parent / "results"


def run_case(offset, controller=None):
    env = CartPole(initial_offset=float(offset))
    controller = Controller(hybrid=False) if controller is None else controller
    initial = env.get_state()
    initially_captured = (abs(wrap_to_pi(initial[3] - math.pi)) < 0.10
                          and abs(initial[2]) < 0.25)
    samples = []
    for step in range(round(DURATION / DT)):
        force = controller.compute_control(env.get_state().copy())
        state = env.step(force, dt=DT).copy()
        if not np.all(np.isfinite(state)) or not env.solver.successful():
            raise RuntimeError(f"Integrator failed for offset {offset}")
        samples.append(((step + 1) * DT, *state, force,
                        abs(wrap_to_pi(state[3] - math.pi))))
    data = np.asarray(samples)
    tail = data[-round(5.0 / DT):]
    angular_capture = (data[:, 6] < 0.10) & (np.abs(data[:, 3]) < 0.25)
    capture = np.flatnonzero(angular_capture)
    # Unlike first capture, settling requires the criteria to remain satisfied
    # through the end of the trial. Include cart position in this criterion.
    settled = angular_capture & (np.abs(data[:, 1]) < 0.50)
    last_bad = np.flatnonzero(~settled)
    settling_index = int(last_bad[-1]) + 1 if last_bad.size else 0
    return {
        "offset_rad": float(offset),
        "success": bool(np.max(tail[:, 6]) < 0.10
                        and np.max(np.abs(tail[:, 1])) < 0.50),
        "capture_time_s": (0.0 if initially_captured else
                           float(data[capture[0], 0]) if capture.size else ""),
        "settling_time_s": (float(data[settling_index, 0])
                            if settling_index < len(data) else ""),
        "peak_cart_position_m": float(np.max(np.abs(data[:, 1]))),
        "tail_max_angle_error_rad": float(np.max(tail[:, 6])),
        "tail_max_cart_position_m": float(np.max(np.abs(tail[:, 1]))),
        "max_force_N": float(np.max(np.abs(data[:, 5]))),
        "force_squared_integral_N2s": float(DT * np.sum(data[:, 5] ** 2)),
        "saturation_fraction": float(np.mean(np.abs(data[:, 5]) >= 30.0 - 1e-9)),
    }


def write_rows(name, rows):
    RESULTS.mkdir(exist_ok=True)
    with (RESULTS / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def scan_range():
    magnitudes = [round(k * 0.05, 10) for k in range(1, 63)] + [math.pi]
    rows = [{"phase": "grid", **run_case(0.0)}]
    for sign in [1, -1]:
        grid = []
        for magnitude in magnitudes:
            row = {"phase": "grid", **run_case(sign * magnitude)}
            grid.append(row)
            rows.append(row)
        first_failure = next(i for i, row in enumerate(grid) if not row["success"])
        low = 0.0 if first_failure == 0 else abs(grid[first_failure - 1]["offset_rad"])
        high = abs(grid[first_failure]["offset_rad"])
        # Refine the first adjacent pass/fail pair, not an assumed global bound.
        while high - low > 0.001:
            midpoint = (low + high) / 2
            row = {"phase": "refinement", **run_case(sign * midpoint)}
            rows.append(row)
            if row["success"]:
                low = midpoint
            else:
                high = midpoint
        print(f"sign={sign:+d}: first transition bracket [{low:.6f}, {high:.6f}] rad",
              f"; successes beyond first failure={sum(r['success'] for r in grid[first_failure:])}",
              flush=True)
    write_rows("stability_sweep.csv", rows)


def compare_lqr():
    profiles = [("lower_angle", 30.0, 0.2), ("selected", 120.0, 0.2),
                ("higher_input_penalty", 120.0, 1.0),
                ("lower_input_penalty", 120.0, 0.05)]
    rows = []
    offsets = [math.pi / 40, 0.01, 0.1, math.pi / 8, math.pi / 4, math.pi]
    for name, angle_weight, input_weight in profiles:
        gain = lqr(A, B, np.diag([2.0, 1.0, 2.0, angle_weight]),
                   np.array([[input_weight]]))
        for offset in offsets:
            controller = Controller(hybrid=False)
            controller.K = gain.copy()
            rows.append({"profile": name, "q_angle": angle_weight, "r": input_weight,
                         **run_case(offset, controller)})
        print(f"LQR profile {name}: {rows[-2]}", flush=True)
    write_rows("lqr_tuning.csv", rows)


def compare_swingup():
    rows = []
    for gain in [20.0, 40.0, 60.0]:
        controller = Controller(hybrid=True)
        controller.energy_gain = gain
        row = {"energy_gain": gain, **run_case(math.pi, controller)}
        rows.append(row)
        print(f"Swing-up tuning: {row}", flush=True)
    write_rows("swingup_tuning.csv", rows)


def main():
    compare_lqr()
    compare_swingup()
    scan_range()


if __name__ == "__main__":
    main()
