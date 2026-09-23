"""Upright LQR controller for the supplied six-state DoubleCartPole.

State order: cart position, cart velocity, pole-1 angular velocity,
pole-2 angular velocity, pole-1 angle, pole-2 angle.
"""

import math
import numpy as np
from scipy.linalg import solve_continuous_are


def linear_model():
    """Linearize the simulator's implicit dynamics about both poles upright.

    At theta1=theta2=pi, the acceleration equations are H a = r. Velocity
    squared terms vanish to first order, so only cart drag, force, and the
    two angle displacements contribute to r.
    """
    cart_mass = pole1_mass = pole2_mass = 0.5
    length1 = length2 = 0.6
    friction = 0.1
    gravity = -9.82  # Sign convention in DoubleCartPole.dynamics.

    h = np.array([
        [2 * (cart_mass + pole1_mass + pole2_mass),
         (pole1_mass + 2 * pole2_mass) * length1,
         pole2_mass * length2],
        [3 * pole1_mass + 6 * pole2_mass,
         (2 * pole1_mass + 6 * pole2_mass) * length1,
         3 * pole2_mass * length2],
        [3.0, 3 * length1, 2 * length2],
    ])
    rhs_jacobian = np.array([
        [-2 * friction, 0.0, 0.0],
        [0.0, -(3 * pole1_mass + 6 * pole2_mass) * gravity, 0.0],
        [0.0, 0.0, -3 * gravity],
    ])
    acceleration_jacobian = np.linalg.solve(h, rhs_jacobian)
    force_jacobian = np.linalg.solve(h, np.array([2.0, 0.0, 0.0]))

    a = np.zeros((6, 6))
    a[0, 1] = 1.0
    a[1:4, 1] = acceleration_jacobian[:, 0]
    a[1:4, 4:6] = acceleration_jacobian[:, 1:]
    a[4, 2] = 1.0
    a[5, 3] = 1.0
    b = np.zeros((6, 1))
    b[1:4, 0] = force_jacobian
    return h, a, b


H, A, B = linear_model()
Q = np.diag([2.0, 1.0, 2.0, 2.0, 120.0, 120.0])
R = np.array([[0.2]])
P = solve_continuous_are(A, B, Q, R)
K = np.linalg.solve(R, B.T @ P)


def wrap_to_pi(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


class DoubleController:
    def __init__(self, force_limit=40.0):
        self.force_limit = float(force_limit)

    def compute_control(self, state):
        state = np.asarray(state, dtype=float).reshape(6)
        error = state.copy()
        error[4] = wrap_to_pi(state[4] - math.pi)
        error[5] = wrap_to_pi(state[5] - math.pi)
        force = float((-K @ error)[0])
        return float(np.clip(force, -self.force_limit, self.force_limit))


if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True)
    print("H =\n", H)
    print("A =\n", A)
    print("B =\n", B)
    print("Q =\n", Q)
    print("R =\n", R)
    print("K =\n", K)
    print("closed-loop poles =\n", np.linalg.eigvals(A - B @ K))
