"""LQR balance controller with an optional energy-shaping swing-up mode."""

import math
import numpy as np

from lqr_starter import K


def wrap_to_pi(angle):
    """Map an angle to [-pi, pi)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class Controller:
    """Hybrid energy-shaping and LQR controller.

    Near upright, the force is exactly u = K(g - x), as requested. Away from
    upright, hybrid mode uses bounded energy shaping and hands control to LQR
    inside the capture region. Set hybrid=False for pure-LQR experiments.
    """

    def __init__(self, hybrid=True, force_limit=30.0, double_force_limit=40.0):
        self.hybrid = hybrid
        self.force_limit = float(force_limit)
        self.double_force_limit = float(double_force_limit)
        self.K = K.copy()
        self.captured = False

        self.m = 0.5
        self.l = 0.5
        self.g = 9.82

        self.energy_gain = 40.0
        self.cart_position_gain = 1.0
        self.cart_velocity_gain = 2.0
        self.capture_angle = 0.42
        self.release_angle = 0.65
        self.capture_rate = 3.5
        self.double_controller = None

    def lqr_control(self, state):
        state = np.asarray(state, dtype=float).reshape(4)
        goal_minus_state = np.array(
            [-state[0], -state[1], -state[2], wrap_to_pi(math.pi - state[3])]
        )
        return float((self.K @ goal_minus_state)[0])

    def swingup_control(self, state):
        x, x_dot, theta_dot, theta = np.asarray(state, dtype=float).reshape(4)

        # Uniform rod about one end: I = m*l^2/3. With theta measured from
        # downward, upright energy relative to downward is m*g*l.
        energy = (
            self.m * self.l**2 * theta_dot**2 / 6.0
            + self.m * self.g * self.l * (1.0 - math.cos(theta)) / 2.0
        )
        target_energy = self.m * self.g * self.l
        phase = theta_dot * math.cos(theta)
        direction = 1.0 if abs(phase) < 1e-8 else math.copysign(1.0, phase)
        pump = -self.energy_gain * (target_energy - energy) * direction
        centre_cart = -self.cart_position_gain * x - self.cart_velocity_gain * x_dot
        return float(pump + centre_cart)

    def compute_control(self, state):
        state = np.asarray(state, dtype=float).reshape(-1)
        if state.size == 6:
            if self.double_controller is None:
                from double_cartpole_control import DoubleController
                self.double_controller = DoubleController(force_limit=self.double_force_limit)
            return self.double_controller.compute_control(state)
        if state.size != 4:
            raise ValueError(f"Expected four or six state values, got {state.size}")
        upright_error = abs(wrap_to_pi(state[3] - math.pi))

        if not self.hybrid:
            force = self.lqr_control(state)
        else:
            if upright_error < self.capture_angle and abs(state[2]) < self.capture_rate:
                self.captured = True
            elif upright_error > self.release_angle:
                self.captured = False
            force = self.lqr_control(state) if self.captured else self.swingup_control(state)

        return float(np.clip(force, -self.force_limit, self.force_limit))
