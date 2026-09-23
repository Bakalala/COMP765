"""Continuous-time LQR design for the supplied single cart-pole.

State order: [cart position, cart velocity, pole angular velocity, pole angle].
The linear model is expressed in error coordinates about theta = pi.
"""

import numpy as np
import scipy.linalg


def lqr(A, B, Q, R):
    """Return the stabilizing continuous-time LQR gain."""
    P = scipy.linalg.solve_continuous_are(A, B, Q, R)
    return np.linalg.solve(R, B.T @ P)


def cartpole_matrices(M=0.5, m=0.5, l=0.5, b=1.0, g=9.82):
    """Linearization about [0, 0, 0, pi] for the supplied dynamics.

    The handout lists b=0.1, whereas cartpole_envs.py uses b=1.0 for
    CartPole. The default here matches the executable simulator. Passing
    b=0.1 reproduces the handout's numerical matrices.
    """
    denominator = 4.0 * (M + m) - 3.0 * m
    A = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, -4.0 * b / denominator, 0.0, 3.0 * m * g / denominator],
            [0.0, -6.0 * b / (l * denominator), 0.0,
             6.0 * (M + m) * g / (l * denominator)],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    B = np.array(
        [[0.0], [4.0 / denominator], [6.0 / (l * denominator)], [0.0]],
        dtype=float,
    )
    return A, B


# Tuned once and then kept fixed for every experiment in experiment.py.
A, B = cartpole_matrices()
Q = np.diag([2.0, 1.0, 2.0, 120.0])
R = np.array([[0.2]])
K = lqr(A, B, Q, R)


if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True)
    print("A =\n", A)
    print("B =\n", B)
    print("Q =\n", Q)
    print("R =\n", R)
    print("K =\n", K)
    print("closed-loop poles =\n", np.linalg.eigvals(A - B @ K))
