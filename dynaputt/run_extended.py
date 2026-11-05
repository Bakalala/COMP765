"""Robustness studies that extend, and never pool with, the primary experiment.

Four questions, each on 100 seeds that are disjoint from the primary five:

* budget: how does learning speed change across planning budgets?
* rng: does sharing one random stream change the result on average?
* init: how much of the Dyna-Q advantage depends on the zero initial Q table?
* route: how do planning and initial value interact as the best route gets longer?
"""

from __future__ import annotations

import csv
from collections import deque
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from agents import DynaQAgent
from golf_env import MiniGolfEnv
from run_experiments import COURSES, EPISODES, evaluate
from summarize_results import first_stable_checkpoint, first_window_threshold


SEEDS = tuple(range(1000, 1100))
BUDGETS = (0, 1, 2, 5, 10, 20, 50)
CORE_BUDGETS = (0, 5, 20)
# Every shot costs 1, so 0 is optimistic for all pairs and -20 = -1 / (1 - gamma)
# is the lowest possible value. -5 is close to the wall-gate start value (-4.52).
INITIAL_VALUES = (0.0, -5.0, -20.0)
LADDER_COURSES = ("ladder_1", "ladder_2", "ladder_3", "ladder_4")
# The longest ladder needs a longer run for zero planning to reach the threshold.
LADDER_EPISODES = 800
FIVE_SEED_RESAMPLES = 20000
BOOTSTRAP_RESAMPLES = 10000
OUT_DIR = Path(__file__).parent / "results" / "extended"


def shortest_route_length(course: str) -> int:
    env = MiniGolfEnv(course=course)
    start = env.reset()
    distance = {start: 0}
    queue = deque([start])
    while queue:
        state = queue.popleft()
        if state == env.hole_state:
            continue
        for action in range(env.n_actions):
            env.reset()
            env.state = state
            next_state = env.step(action)[0]
            if next_state not in distance:
                distance[next_state] = distance[state] + 1
                queue.append(next_state)
    return distance[env.hole_state]


def run_once(course: str, planning_steps: int, seed: int, shared_rng: bool, q_init: float,
             episodes: int = EPISODES) -> dict:
    env = MiniGolfEnv(course=course)
    agent = DynaQAgent(env.n_states, env.n_actions, planning_steps=planning_steps, seed=seed)
    agent.q.fill(q_init)
    if shared_rng:
        agent.planning_rng = agent.rng

    sinks: list[int] = []
    shots: list[int] = []
    checkpoints: list[float] = []
    for episode in range(1, episodes + 1):
        state = env.reset()
        done = False
        strokes = 0
        sank = False
        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            agent.observe(state, action, reward, next_state, info["sank"])
            state = next_state
            strokes += 1
            sank = info["sank"]
        agent.end_episode()
        sinks.append(int(sank))
        shots.append(strokes)
        checkpoints.append(evaluate(env, agent)[0])

    threshold = first_window_threshold(sinks)
    return {
        "course": course,
        "planning_steps": planning_steps,
        "rng": "shared" if shared_rng else "separate",
        "q_init": q_init,
        "seed": seed,
        "threshold_episode": threshold if threshold is not None else "",
        "real_shots_to_threshold": sum(shots[:threshold]) if threshold is not None else "",
        "greedy_stable_episode": first_stable_checkpoint(checkpoints) or "",
        "first50_training_success": sum(sinks[:50]) / 50,
        "final50_greedy_success": sum(checkpoints[-50:]) / 50,
    }


def conditions() -> list[tuple[str, int, bool, float]]:
    wanted = [(course, k, False, 0.0) for course in COURSES for k in BUDGETS]
    wanted += [(course, k, True, 0.0) for course in COURSES for k in CORE_BUDGETS if k]
    wanted += [
        (course, k, False, q_init)
        for course in COURSES for k in CORE_BUDGETS for q_init in INITIAL_VALUES[1:]
    ]
    return wanted


def ladder_conditions() -> list[tuple[str, int, bool, float]]:
    return [
        (course, k, False, q_init)
        for course in LADDER_COURSES for q_init in INITIAL_VALUES for k in CORE_BUDGETS
    ]


def values(rows: list[dict], key: str) -> np.ndarray:
    return np.array([float(row[key]) for row in rows if row[key] != ""])


def select(rows: list[dict], course: str, k: int, rng: str = "separate", q_init: float = 0.0) -> list[dict]:
    found = [row for row in rows if row["course"] == course and row["planning_steps"] == k
             and row["rng"] == rng and row["q_init"] == q_init]
    return sorted(found, key=lambda row: row["seed"])


def paired_ci(first: list[dict], second: list[dict], generator: np.random.Generator):
    """Mean of first-minus-second threshold, with a 95% paired bootstrap interval."""
    difference = values(first, "threshold_episode") - values(second, "threshold_episode")
    picks = generator.integers(0, len(difference), (BOOTSTRAP_RESAMPLES, len(difference)))
    low, high = np.percentile(difference[picks].mean(axis=1), [2.5, 97.5])
    return float(difference.mean()), float(low), float(high), difference


def summarize(rows: list[dict]) -> tuple[list[dict], list[str]]:
    generator = np.random.default_rng(765)
    summary: list[dict] = []
    for course, k, shared_rng, q_init in conditions():
        group = select(rows, course, k, "shared" if shared_rng else "separate", q_init)
        thresholds = values(group, "threshold_episode")
        shots = values(group, "real_shots_to_threshold")
        stable = values(group, "greedy_stable_episode")
        summary.append({
            "course": course,
            "planning_steps": k,
            "rng": "shared" if shared_rng else "separate",
            "q_init": q_init,
            "seeds": len(group),
            "seeds_reaching_threshold": len(thresholds),
            "threshold_mean": thresholds.mean(),
            "threshold_se": thresholds.std(ddof=1) / np.sqrt(len(thresholds)),
            "real_shots_mean": shots.mean(),
            "real_shots_se": shots.std(ddof=1) / np.sqrt(len(shots)),
            "seeds_greedy_stable": len(stable),
            "greedy_stable_mean": stable.mean() if len(stable) else float("nan"),
            "first50_training_success_mean": values(group, "first50_training_success").mean(),
            "final50_greedy_success_mean": values(group, "final50_greedy_success").mean(),
        })

    lines = [
        "# Extended study summary",
        "",
        "Generated by `run_extended.py`. These 100 seeds (1000-1099) are separate from",
        "the primary five and are never pooled with them. Intervals are 95% paired",
        "bootstrap intervals over seeds. They describe this implementation and these two",
        "courses, not a population of courses.",
        "",
        "## Planning budget (separate streams, zero initial Q)",
        "",
        "| Course | Updates | Threshold episode (mean ± SE) | Real shots | Greedy-stable episode (seeds) | Greedy success, last 50 |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    def table_row(row: dict, label: str) -> str:
        return (
            f"| {row['course']} | {label} | {row['threshold_mean']:.1f} ± {row['threshold_se']:.1f} | "
            f"{row['real_shots_mean']:.0f} | {row['greedy_stable_mean']:.1f} "
            f"({row['seeds_greedy_stable']}/{row['seeds']}) | {row['final50_greedy_success_mean']:.3f} |"
        ).replace("nan", "none")

    for row in summary:
        if row["rng"] == "separate" and row["q_init"] == 0.0:
            lines.append(table_row(row, str(row["planning_steps"])))

    lines += [
        "",
        "## Shared versus separate random streams",
        "",
        "| Course | Stream | 5-update threshold | 20-update threshold | 5 minus 20 (95% CI) | Seeds where 20 is faster / tie / slower | Five-seed samples with 5 ahead of 20 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for course in COURSES:
        for rng in ("separate", "shared"):
            five = select(rows, course, 5, rng)
            twenty = select(rows, course, 20, rng)
            mean, low, high, difference = paired_ci(five, twenty, generator)
            a = values(five, "threshold_episode")
            b = values(twenty, "threshold_episode")
            picks = generator.integers(0, len(a), (FIVE_SEED_RESAMPLES, 5))
            flips = float((a[picks].mean(axis=1) < b[picks].mean(axis=1)).mean())
            lines.append(
                f"| {course} | {rng} | {a.mean():.1f} | {b.mean():.1f} | "
                f"{mean:.1f} ({low:.1f}, {high:.1f}) | {(difference > 0).sum()} / "
                f"{(difference == 0).sum()} / {(difference < 0).sum()} | {flips:.1%} |"
            )

    lines += [
        "",
        "## Initial Q value (separate streams)",
        "",
        "| Course | Initial Q | Updates | Threshold episode (mean ± SE) | Real shots | Greedy-stable episode (seeds) | Greedy success, last 50 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for course in COURSES:
        for q_init in INITIAL_VALUES:
            for k in CORE_BUDGETS:
                row = next(item for item in summary if item["course"] == course
                           and item["planning_steps"] == k and item["rng"] == "separate"
                           and item["q_init"] == q_init)
                body = table_row(row, str(k)).split("|", 2)[2]
                lines.append(f"| {course} | {q_init:.0f} |{body}")

    lines += [
        "",
        "## Planning effect by initial Q (0 updates minus 20 updates)",
        "",
        "| Course | Initial Q | Threshold difference (95% CI) |",
        "|---|---:|---:|",
    ]
    for course in COURSES:
        for q_init in INITIAL_VALUES:
            mean, low, high, _ = paired_ci(
                select(rows, course, 0, q_init=q_init),
                select(rows, course, 20, q_init=q_init),
                generator,
            )
            lines.append(f"| {course} | {q_init:.0f} | {mean:.1f} ({low:.1f}, {high:.1f}) |")
    return summary, lines


def summarize_ladder(rows: list[dict]) -> tuple[list[dict], list[str]]:
    summary: list[dict] = []
    lines = [
        "",
        "## Route length (separate streams, 800 episodes, serpentine courses)",
        "",
        "Threshold and greedy-stable means use only the seeds that reached them, so",
        "read them with the seed counts.",
        "",
        "| Course | Best route (shots) | Initial Q | Updates | Seeds reaching threshold | Threshold episode (mean ± SE) | Greedy-stable episode (seeds) | Greedy success, last 50 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for course, k, _, q_init in ladder_conditions():
        group = select(rows, course, k, q_init=q_init)
        thresholds = values(group, "threshold_episode")
        stable = values(group, "greedy_stable_episode")
        route = shortest_route_length(course)
        row = {
            "course": course,
            "best_route_shots": route,
            "planning_steps": k,
            "q_init": q_init,
            "seeds": len(group),
            "seeds_reaching_threshold": len(thresholds),
            "threshold_mean": thresholds.mean() if len(thresholds) else float("nan"),
            "threshold_se": (thresholds.std(ddof=1) / np.sqrt(len(thresholds))
                             if len(thresholds) > 1 else float("nan")),
            "seeds_greedy_stable": len(stable),
            "greedy_stable_mean": stable.mean() if len(stable) else float("nan"),
            "final50_greedy_success_mean": values(group, "final50_greedy_success").mean(),
        }
        summary.append(row)
        lines.append((
            f"| {course} | {route} | {q_init:.0f} | {k} | {len(thresholds)}/{len(group)} | "
            f"{row['threshold_mean']:.1f} ± {row['threshold_se']:.1f} | "
            f"{row['greedy_stable_mean']:.1f} ({len(stable)}/{len(group)}) | "
            f"{row['final50_greedy_success_mean']:.3f} |"
        ).replace("nan ± nan", "none").replace("nan", "none"))
    return summary, lines


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jobs = [(*condition[:2], seed, *condition[2:]) for condition in conditions() for seed in SEEDS]
    ladder_jobs = [
        (*condition[:2], seed, *condition[2:], LADDER_EPISODES)
        for condition in ladder_conditions() for seed in SEEDS
    ]
    with Pool() as pool:
        rows = pool.starmap(run_once, jobs, chunksize=20)
        ladder_rows = pool.starmap(run_once, ladder_jobs, chunksize=20)

    write_csv(OUT_DIR / "runs.csv", rows)
    write_csv(OUT_DIR / "route_length_runs.csv", ladder_rows)
    summary, lines = summarize(rows)
    ladder_summary, ladder_lines = summarize_ladder(ladder_rows)
    write_csv(OUT_DIR / "summary.csv", summary)
    write_csv(OUT_DIR / "route_length_summary.csv", ladder_summary)
    (OUT_DIR / "summary.md").write_text("\n".join(lines + ladder_lines) + "\n")
    print(f"Wrote {len(rows) + len(ladder_rows)} runs to {OUT_DIR}")


if __name__ == "__main__":
    main()
