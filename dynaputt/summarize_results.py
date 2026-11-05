"""Summarize the episode CSV into per-course research metrics."""

from __future__ import annotations

import csv
import statistics
from pathlib import Path


RESULTS = Path(__file__).parent / "results"
INPUT = RESULTS / "episode_results.csv"


def first_window_threshold(successes: list[int], window: int = 20, threshold: float = 0.8):
    for end in range(window, len(successes) + 1):
        if sum(successes[end - window:end]) / window >= threshold:
            return end
    return None


def first_stable_checkpoint(evaluations: list[float], min_run: int = 50):
    """First episode after which every greedy checkpoint sinks, for >= min_run."""
    for start in range(len(evaluations) - min_run + 1):
        if all(value == 1.0 for value in evaluations[start:]):
            return start + 1
    return None


def mean_sd(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def main() -> None:
    with INPUT.open(newline="") as source:
        rows = list(csv.DictReader(source))
    courses = sorted({row["course"] for row in rows})
    settings = sorted({int(row["planning_steps"]) for row in rows})
    summaries: list[dict] = []

    for course in courses:
        for planning_steps in settings:
            group = [row for row in rows
                     if row["course"] == course and int(row["planning_steps"]) == planning_steps]
            seeds = sorted({row["seed"] for row in group}, key=int)
            seed_thresholds: list[float] = []
            seed_shots: list[float] = []
            seed_stable: list[float] = []
            first_50: list[float] = []
            for seed in seeds:
                seed_rows = sorted(
                    (row for row in group if row["seed"] == seed),
                    key=lambda row: int(row["episode"]),
                )
                successes = [int(row["training_sank"]) for row in seed_rows]
                threshold_episode = first_window_threshold(successes)
                if threshold_episode is not None:
                    seed_thresholds.append(float(threshold_episode))
                    seed_shots.append(float(sum(
                        int(row["training_strokes"]) for row in seed_rows[:threshold_episode]
                    )))
                stable_episode = first_stable_checkpoint(
                    [float(row["evaluation_success_rate"]) for row in seed_rows]
                )
                if stable_episode is not None:
                    seed_stable.append(float(stable_episode))
                first_50.append(sum(successes[:50]) / min(50, len(successes)))

            final_rows = [row for row in group if int(row["episode"]) > 300]
            final_success = [float(row["evaluation_success_rate"]) for row in final_rows]
            final_strokes = [
                float(row["evaluation_mean_strokes_if_sunk"])
                for row in final_rows
                if row["evaluation_mean_strokes_if_sunk"].lower() != "nan"
            ]
            threshold_mean, threshold_sd = mean_sd(seed_thresholds)
            shots_mean, shots_sd = mean_sd(seed_shots)
            first_mean, first_sd = mean_sd(first_50)
            final_mean, final_sd = mean_sd(final_success)
            summaries.append({
                "course": course,
                "planning_steps": planning_steps,
                "seeds_reaching_threshold": len(seed_thresholds),
                "episodes_to_80pct_rolling20_mean": threshold_mean,
                "episodes_to_80pct_rolling20_sd": threshold_sd,
                "first50_training_success_mean_across_seeds": first_mean,
                "first50_training_success_sd_across_seeds": first_sd,
                "final50_checkpoint_greedy_success_mean": final_mean,
                "final50_checkpoint_greedy_success_sd": final_sd,
                "final50_checkpoint_strokes_if_sunk_mean": (
                    statistics.mean(final_strokes) if final_strokes else float("nan")
                ),
                "real_shots_to_threshold_mean": shots_mean,
                "real_shots_to_threshold_sd": shots_sd,
                "seeds_greedy_stable": len(seed_stable),
                "greedy_stable_episode_mean": (
                    statistics.mean(seed_stable) if seed_stable else float("nan")
                ),
            })

    with (RESULTS / "summary.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    lines = [
        "# Primary experiment summary",
        "",
        "Generated from `episode_results.csv`. Threshold is the first episode ending",
        "a 20-episode window with at least 80% training sinks. Greedy success is the",
        "mean over the checkpoints after episodes 301-350 and five agent seeds.",
        "Real shots are training shots up to the threshold episode. Greedy-stable is",
        "the first episode after which every checkpoint sinks, for at least the last",
        "50 checkpoints; the column shows its mean and how many seeds qualify. The game",
        "and greedy evaluation policy are deterministic. Action exploration and model",
        "planning use separate random-number streams in this run. The earlier coupled",
        "stream run is preserved in `results/rng_coupled_baseline/`.",
        "",
        "| Course | Planning updates | Episodes to threshold (mean ± SD) | Real shots to threshold (mean ± SD) | First 50 training success | Greedy success, episodes 301-350 | Greedy-stable episode (seeds) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['course']} | {row['planning_steps']} | "
            f"{row['episodes_to_80pct_rolling20_mean']:.1f} ± "
            f"{row['episodes_to_80pct_rolling20_sd']:.1f} | "
            f"{row['real_shots_to_threshold_mean']:.0f} ± "
            f"{row['real_shots_to_threshold_sd']:.0f} | "
            f"{row['first50_training_success_mean_across_seeds']:.3f} | "
            f"{row['final50_checkpoint_greedy_success_mean']:.3f} | "
            f"{row['greedy_stable_episode_mean']:.1f} ({row['seeds_greedy_stable']}/{len(seeds)}) |".replace("nan", "none")
        )
    lines.extend([
        "",
        "These results come from five seeds on a small deterministic toy environment",
        "and support conclusions about this implementation only. See",
        "`results/extended/` for the 100-seed robustness and initial-value studies.",
    ])
    summary_path = RESULTS / "experiment_summary.md"
    summary_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {RESULTS / 'summary.csv'}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
