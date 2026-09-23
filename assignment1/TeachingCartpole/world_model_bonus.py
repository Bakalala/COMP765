"""Bonus: learn cart-pole dynamics from (next state, state, force) records.

The fitted model is deliberately small and interpretable. It learns the
coefficients of trigonometric features from data, without using the simulator's
mass, length, friction, or gravity values. A local lookahead controller then
uses model rollouts to select a force.

Run: python3 world_model_bonus.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import math

import numpy as np
from scipy.linalg import solve_continuous_are

from cartpole_envs import CartPole
from cartpole_control import Controller, wrap_to_pi
from lqr_starter import A, B, Q, R


DT = 0.005
BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"


def collect(seed: int, episodes: int, steps: int) -> np.ndarray:
    """Create records in the teaching simulator's (s', s, u) column order."""
    rng = np.random.default_rng(seed)
    records = []
    for _ in range(episodes):
        angle = rng.uniform(-math.pi, math.pi)
        start = [rng.uniform(-1, 1), rng.uniform(-1, 1),
                 rng.uniform(-3, 3), angle]
        env = CartPole(x0=start, initial_offset=0.0)
        force = 0.0
        for t in range(steps):
            if t % 10 == 0:
                force = rng.uniform(-25.0, 25.0)
            state = env.get_state().copy()
            next_state = env.step(force, DT).copy()
            records.append(np.r_[next_state, state, force])
    return np.asarray(records)


def _feature_blocks(records: np.ndarray):
    next_state = records[:, :4]
    state = records[:, 4:8]
    force = records[:, 8]
    middle = 0.5 * (state + next_state)
    velocity = middle[:, 1]
    angular_velocity = middle[:, 2]
    angle = middle[:, 3]
    sin = np.sin(angle)
    cos = np.cos(angle)
    cart_accel = (next_state[:, 1] - state[:, 1]) / DT
    pole_accel = (next_state[:, 2] - state[:, 2]) / DT
    cart_features = np.column_stack([
        angular_velocity**2 * sin,
        sin * cos,
        force,
        velocity,
        cart_accel * cos**2,
    ])
    pole_features = np.column_stack([
        angular_velocity**2 * sin * cos,
        sin,
        force * cos,
        velocity * cos,
        pole_accel * cos**2,
    ])
    return cart_features, pole_features, cart_accel, pole_accel


@dataclass
class LearnedWorldModel:
    cart: np.ndarray
    pole: np.ndarray

    @classmethod
    def fit(cls, records: np.ndarray):
        xf, pf, xa, pa = _feature_blocks(records)
        cart = np.linalg.lstsq(xf, xa, rcond=None)[0]
        pole = np.linalg.lstsq(pf, pa, rcond=None)[0]
        return cls(cart, pole)

    def acceleration(self, state: np.ndarray, force: np.ndarray):
        """Return cart and pole angular acceleration; supports batched states."""
        s = np.asarray(state)
        u = np.asarray(force)
        velocity, omega, theta = s[..., 1], s[..., 2], s[..., 3]
        sin, cos = np.sin(theta), np.cos(theta)
        cf = np.stack([omega**2 * sin, sin * cos,
                       np.broadcast_to(u, velocity.shape), velocity], axis=-1)
        pf = np.stack([omega**2 * sin * cos, sin,
                       np.broadcast_to(u, velocity.shape) * cos,
                       velocity * cos], axis=-1)
        cart = (cf @ self.cart[:4]) / (1.0 - self.cart[4] * cos**2)
        pole = (pf @ self.pole[:4]) / (1.0 - self.pole[4] * cos**2)
        return cart, pole

    def derivative(self, state: np.ndarray, force: np.ndarray):
        cart, pole = self.acceleration(state, force)
        return np.stack([state[..., 1], cart, pole, state[..., 2]], axis=-1)

    def step(self, state: np.ndarray, force: np.ndarray, dt: float = DT):
        """Fourth order integration of the learned continuous dynamics."""
        s = np.asarray(state)
        k1 = self.derivative(s, force)
        k2 = self.derivative(s + 0.5 * dt * k1, force)
        k3 = self.derivative(s + 0.5 * dt * k2, force)
        k4 = self.derivative(s + dt * k3, force)
        return s + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0


class ModelLookaheadController:
    """Use learned model rollouts to refine the local LQR force."""

    def __init__(self, model: LearnedWorldModel):
        self.model = model
        self.baseline = Controller(hybrid=False)
        self.candidates = np.linspace(-30.0, 30.0, 31)
        self.terminal_value = solve_continuous_are(A, B, Q, R)

    def compute_control(self, state):
        state = np.asarray(state, dtype=float)
        # Keep short horizon near the upright equilibrium where this simple
        # quadratic cost is meaningful. Replan every environment step.
        baseline = self.baseline.compute_control(state)
        candidates = np.unique(np.r_[self.candidates, baseline])
        batch = np.repeat(state[None, :], len(candidates), axis=0)
        cost = np.zeros(len(candidates))
        for _ in range(10):
            batch = self.model.step(batch, candidates, DT)
            angle_error = (batch[:, 3] - math.pi + math.pi) % (2 * math.pi) - math.pi
            delta = batch.copy()
            delta[:, 3] = angle_error
            cost += DT * (np.einsum("bi,ij,bj->b", delta, Q, delta)
                          + 0.2 * candidates**2)
        cost += np.einsum("bi,ij,bj->b", delta, self.terminal_value, delta)
        return float(candidates[np.argmin(cost)])


def evaluate_model(model: LearnedWorldModel, records: np.ndarray):
    actual = records[:, :4]
    prediction = model.step(records[:, 4:8], records[:, 8])
    errors = prediction - actual
    errors[:, 3] = (errors[:, 3] + math.pi) % (2 * math.pi) - math.pi
    return np.sqrt(np.mean(errors**2, axis=0))


def evaluate_rollouts(model: LearnedWorldModel, records: np.ndarray,
                      episode_steps: int):
    """Open-loop prediction on held-out episodes using recorded forces."""
    errors = []
    example = None
    for start in range(0, len(records), episode_steps):
        episode = records[start:start + episode_steps]
        predicted = episode[0, 4:8].copy()
        trace = []
        for row in episode:
            predicted = model.step(predicted, row[8])
            truth = row[:4]
            error = predicted - truth
            error[3] = wrap_to_pi(error[3])
            errors.append(error)
            trace.append(np.r_[truth, predicted])
        if example is None:
            example = np.asarray(trace)
    return np.sqrt(np.mean(np.asarray(errors)**2, axis=0)), example


def evaluate_control(model: LearnedWorldModel, initial_offset: float,
                     duration: float = 8.0):
    env = CartPole(initial_offset=initial_offset)
    controller = ModelLookaheadController(model)
    data = []
    for _ in range(int(duration / DT)):
        state = env.get_state().copy()
        force = controller.compute_control(state)
        next_state = env.step(force, DT).copy()
        data.append([next_state[0], wrap_to_pi(next_state[3] - math.pi), force])
    data = np.asarray(data)
    tail = data[-int(2.0 / DT):]
    return {
        "initial_offset_rad": initial_offset,
        "success": bool(np.max(np.abs(tail[:, 1])) < 0.1
                        and np.max(np.abs(tail[:, 0])) < 0.5),
        "tail_max_angle_error_rad": float(np.max(np.abs(tail[:, 1]))),
        "tail_max_cart_position_m": float(np.max(np.abs(tail[:, 0]))),
        "peak_force_N": float(np.max(np.abs(data[:, 2]))),
    }


def main():
    RESULTS.mkdir(exist_ok=True)
    train = collect(seed=765, episodes=48, steps=160)
    test = collect(seed=1765, episodes=12, steps=160)
    np.savetxt(RESULTS / "world_model_train.csv", train, delimiter=",",
               header="next_x,next_xdot,next_thetadot,next_theta,x,xdot,thetadot,theta,u",
               comments="")
    model = LearnedWorldModel.fit(train)
    np.savez(RESULTS / "learned_world_model.npz", cart=model.cart, pole=model.pole)
    one_step = evaluate_model(model, test)
    rollout, example = evaluate_rollouts(model, test, episode_steps=160)
    np.savetxt(RESULTS / "world_model_heldout_rollout.csv", example,
               delimiter=",", header="true_x,true_xdot,true_thetadot,true_theta,"
               "pred_x,pred_xdot,pred_thetadot,pred_theta", comments="")
    trials = [evaluate_control(model, offset) for offset in (0.1, math.pi / 8, math.pi / 4)]
    summary = RESULTS / "world_model_bonus_summary.csv"
    with summary.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["metric", "value"])
        writer.writerow(["train_transitions", len(train)])
        writer.writerow(["test_transitions", len(test)])
        for name, value in zip(("x", "xdot", "thetadot", "theta"), one_step):
            writer.writerow([f"heldout_one_step_rmse_{name}", value])
        for name, value in zip(("x", "xdot", "thetadot", "theta"), rollout):
            writer.writerow([f"heldout_open_loop_0p8s_rmse_{name}", value])
        for trial in trials:
            key = f"lookahead_offset_{trial['initial_offset_rad']:.6f}"
            for name, value in trial.items():
                if name != "initial_offset_rad":
                    writer.writerow([f"{key}_{name}", value])
    print("train transitions:", len(train), "test transitions:", len(test))
    print("held-out one-step RMSE [x, xdot, thetadot, theta]:", one_step)
    print("held-out 0.8-s open-loop RMSE [x, xdot, thetadot, theta]:", rollout)
    print("fitted cart coefficients:", model.cart)
    print("fitted pole coefficients:", model.pole)
    for trial in trials:
        print(trial)
    print("Wrote", summary)


if __name__ == "__main__":
    main()
