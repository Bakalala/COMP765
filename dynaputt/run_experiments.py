"""Compare Q-learning and Dyna-Q on the bundled mini-golf courses."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from agents import DynaQAgent
from golf_env import MiniGolfEnv


EPISODES = 350
SEEDS = (11, 23, 37, 41, 59)
PLANNING_SETTINGS = (0, 5, 20)
COURSES = ("easy", "wall_gate")
# The game and greedy evaluation policy are deterministic, so one rollout
# gives the exact policy outcome at each checkpoint. Action exploration and
# Dyna model sampling use separate random-number streams in the current run.
EVALUATION_EPISODES = 1
OUT_DIR = Path(__file__).parent / "results"


def evaluate(env: MiniGolfEnv, agent: DynaQAgent) -> tuple[float, float]:
    successes = 0
    successful_strokes: list[int] = []
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0
    for _ in range(EVALUATION_EPISODES):
        state = env.reset()
        done = False
        strokes = 0
        while not done:
            action = agent.act(state, explore=False)
            state, _, done, info = env.step(action)
            strokes += 1
        if info["sank"]:
            successes += 1
            successful_strokes.append(strokes)
    agent.epsilon = old_epsilon
    mean_strokes = float(np.mean(successful_strokes)) if successful_strokes else float("nan")
    return successes / EVALUATION_EPISODES, mean_strokes


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    curves: dict[tuple[str, int], list[float]] = {}

    for course in COURSES:
        for planning_steps in PLANNING_SETTINGS:
            per_seed_curves: list[list[float]] = []
            for seed in SEEDS:
                env = MiniGolfEnv(course=course)
                agent = DynaQAgent(
                    env.n_states,
                    env.n_actions,
                    planning_steps=planning_steps,
                    seed=seed,
                )
                seed_successes: list[float] = []
                for episode in range(1, EPISODES + 1):
                    state = env.reset(seed=seed + episode)
                    done = False
                    strokes = 0
                    sank = False
                    while not done:
                        action = agent.act(state)
                        next_state, reward, done, info = env.step(action)
                        # A stroke-limit cutoff ends the episode but is not a
                        # terminal state in the learned transition model.
                        agent.observe(state, action, reward, next_state, info["sank"])
                        state = next_state
                        strokes += 1
                        sank = info["sank"]
                    agent.end_episode()
                    seed_successes.append(float(sank))
                    eval_success, eval_strokes = evaluate(env, agent)
                    rows.append({
                        "course": course,
                        "seed": seed,
                        "planning_steps": planning_steps,
                        "episode": episode,
                        "training_sank": int(sank),
                        "training_strokes": strokes,
                        "evaluation_success_rate": eval_success,
                        "evaluation_mean_strokes_if_sunk": eval_strokes,
                        "epsilon": agent.epsilon,
                    })
                per_seed_curves.append(seed_successes)
            curves[(course, planning_steps)] = np.mean(per_seed_curves, axis=0).tolist()

    csv_path = OUT_DIR / "episode_results.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(1, len(COURSES), figsize=(10, 4), sharey=True)
    if len(COURSES) == 1:
        axes = [axes]
    for ax, course in zip(axes, COURSES):
        for planning_steps in PLANNING_SETTINGS:
            raw = np.asarray(curves[(course, planning_steps)])
            window = min(20, len(raw))
            smooth = np.convolve(raw, np.ones(window) / window, mode="valid")
            ax.plot(np.arange(window, len(raw) + 1), smooth, label=f"{planning_steps} planning updates")
        ax.set_title(course.capitalize())
        ax.set_xlabel("Training episode")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Training sink rate (20-episode moving average)")
    axes[-1].legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "learning_curves.png", dpi=160)
    print(f"Wrote {csv_path}")
    print(f"Wrote {OUT_DIR / 'learning_curves.png'}")


if __name__ == "__main__":
    main()

