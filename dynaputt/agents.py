"""Tabular Q-learning and Dyna-Q agents."""

from __future__ import annotations

import numpy as np


class QLearningAgent:
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        *,
        alpha: float = 0.25,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.03,
        epsilon_decay: float = 0.992,
        seed: int | None = None,
    ):
        self.q = np.zeros((n_states, n_actions), dtype=np.float64)
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = np.random.default_rng(seed)
        self.update_count = 0

    def act(self, state: int, *, explore: bool = True) -> int:
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        best = np.flatnonzero(self.q[state] == self.q[state].max())
        if not explore:
            return int(best[0])
        return int(self.rng.choice(best))

    def _update(self, state: int, action: int, reward: float, next_state: int, done: bool) -> None:
        target = reward if done else reward + self.gamma * float(self.q[next_state].max())
        self.q[state, action] += self.alpha * (target - self.q[state, action])
        self.update_count += 1

    def observe(self, state: int, action: int, reward: float, next_state: int, done: bool) -> None:
        self._update(state, action, reward, next_state, done)

    def end_episode(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


class DynaQAgent(QLearningAgent):
    """Dyna-Q with planning randomness separate from action exploration."""

    def __init__(self, *args, planning_steps: int = 5, **kwargs):
        super().__init__(*args, **kwargs)
        if planning_steps < 0:
            raise ValueError("planning_steps must be non-negative")
        self.planning_steps = planning_steps
        seed = kwargs.get("seed")
        if seed is None:
            self.planning_rng = np.random.default_rng()
        else:
            self.planning_rng = np.random.default_rng(
                np.random.SeedSequence([int(seed), 1])
            )
        self.model: dict[tuple[int, int], tuple[float, int, bool]] = {}

    def observe(self, state: int, action: int, reward: float, next_state: int, done: bool) -> None:
        self._update(state, action, reward, next_state, done)
        self.model[(state, action)] = (reward, next_state, done)
        if not self.model:
            return
        known_pairs = tuple(self.model)
        for _ in range(self.planning_steps):
            index = int(self.planning_rng.integers(len(known_pairs)))
            imagined_state, imagined_action = known_pairs[index]
            imagined_reward, imagined_next, imagined_done = self.model[(imagined_state, imagined_action)]
            self._update(
                imagined_state,
                imagined_action,
                imagined_reward,
                imagined_next,
                imagined_done,
            )

