"""Turn saved results into summary tables and the report figures."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from golf import COURSES, MAIN_COURSES, geodesic_map, is_free, lookup
from model import load
from train_models import SEEDS, path_for

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "report" / "figures"
PLANNERS = (
    "random", "aim", "true physics, straight cost", "true physics, geodesic cost",
    "latent L2, H=1", "latent L2, H=2", "latent L2, H=3", "latent L2 anytime, H=3",
    "neighbour graph, H=1", "shot graph replay, H=1", "shot graph model, H=1",
    "shot graph + L2 finish (post hoc)",
)
INK, MUTED, GRID = "#2b2b29", "#6f6e69", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"

plt.rcParams.update({
    "font.family": "serif", "font.size": 7.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "savefig.dpi": 300,
})


def read(pattern: str) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(RESULTS.glob(pattern)):
        with path.open(newline="") as source:
            rows.extend(csv.DictReader(source))
    return rows


def mean_sd(values: list[float]) -> str:
    return f"{np.mean(values):.2f} ± {np.std(values, ddof=1):.2f}" if len(values) > 1 else f"{values[0]:.2f}"


def planner_table(episodes: list[dict]) -> tuple[list[dict], list[str]]:
    """Per-seed success first, then mean and SD over seeds, the replication unit."""
    by_seed: dict = defaultdict(list)
    for row in episodes:
        by_seed[row["course"], row["planner"], row["seed"]].append(row)
    rows = []
    for planner in PLANNERS:
        for course in MAIN_COURSES:
            seeds = [by_seed[course, planner, str(seed)] for seed in SEEDS if by_seed[course, planner, str(seed)]]
            success = [np.mean([int(r["sunk"]) for r in group]) for group in seeds]
            shots = [int(r["shots"]) for group in seeds for r in group if int(r["sunk"])]
            rows.append({"planner": planner, "course": course, "seeds": len(seeds),
                         "success_mean": float(np.mean(success)), "success_sd": float(np.std(success, ddof=1)),
                         "episodes": sum(len(group) for group in seeds),
                         "shots_when_sunk": float(np.mean(shots)) if shots else float("nan")})
    lines = ["## Planner success (mean ± SD over seeds; shots when sunk in brackets)", "",
             "| Planner | " + " | ".join(MAIN_COURSES) + " |", "|---|" + "---:|" * len(MAIN_COURSES)]
    for planner in PLANNERS:
        cells = []
        for course in MAIN_COURSES:
            row = next(r for r in rows if r["planner"] == planner and r["course"] == course)
            cells.append(f"{row['success_mean']:.2f} ± {row['success_sd']:.2f} ({row['shots_when_sunk']:.1f})")
        lines.append(f"| {planner} | " + " | ".join(cells) + " |")
    return rows, lines


def geometry_table(shapes: list[dict]) -> list[str]:
    lines = ["", "## Latent geometry", "",
             "| Course | Spearman vs path length | Spearman vs straight line | Near-hole Spearman | Wall gap | Probe RMSE |",
             "|---|---:|---:|---:|---:|---:|"]
    for course in MAIN_COURSES:
        group = [row for row in shapes if row["course"] == course]
        cell = lambda key: mean_sd([float(r[key]) for r in group if r[key] not in ("", "nan")]) if any(
            r[key] not in ("", "nan") for r in group) else "n/a"
        lines.append(f"| {course} | {cell('spearman_geodesic')} | {cell('spearman_straight')} | "
                     f"{cell('spearman_geodesic_near')} | {cell('wall_gap')} | {cell('probe_rmse')} |")
    return lines


def ablation_table(rows: list[dict]) -> list[str]:
    lines = ["", "## Ablations", "",
             "| Setting | Value | Course | Seeds | Wall gap | Spearman vs path length | Success | Probe RMSE |",
             "|---|---:|---|---:|---:|---:|---:|---:|"]
    groups: dict = defaultdict(list)
    for row in rows:
        groups[row["setting"], float(row["value"]), row["course"]].append(row)
    for (setting, value, course), group in sorted(groups.items()):
        cell = lambda key: mean_sd([float(r[key]) for r in group])
        lines.append(f"| {setting} | {value:g} | {course} | {len(group)} | {cell('wall_gap')} | "
                     f"{cell('spearman_geodesic')} | {cell('success')} | {cell('probe_rmse')} |")
    return lines


def ghost_table(rows: list[dict]) -> list[str]:
    lines = ["", "## Ghost-wall control", "", "| Variant | Seeds | Wall gap |", "|---|---:|---:|"]
    for variant in ("real wall", "ghost wall"):
        group = [float(r["wall_gap"]) for r in rows if r["variant"] == variant]
        lines.append(f"| {variant} | {len(group)} | {mean_sd(group)} |")
    return lines


def distance_maps() -> None:
    size = 64
    centers = (np.arange(size) + 0.5) / size
    xx, yy = np.meshgrid(centers, centers)
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1)
    fig, axes = plt.subplots(2, 4, figsize=(7.0, 3.55))
    for column, name in enumerate(MAIN_COURSES):
        course = COURSES[name]
        model = load(path_for(name, 0))
        free = is_free(course, grid)
        latent = model.encode(grid[free])
        goal = model.encode(np.array(course.hole)[None])[0]
        learned = np.full(size * size, np.nan)
        learned[free] = (latent - goal).norm(dim=1).numpy()
        true = np.full(size * size, np.nan)
        true[free] = lookup(geodesic_map(course), grid[free])
        for row, values in enumerate((learned, true)):
            ax = axes[row, column]
            limit = np.nanpercentile(values, 99)
            shown = np.ma.masked_invalid(values.reshape(size, size))
            colormap = plt.get_cmap("viridis").copy()
            colormap.set_bad("#3d3f46")
            ax.imshow(shown, cmap=colormap, vmin=0, vmax=limit, extent=(0, 1, 1, 0))
            ax.plot(*course.hole, marker="o", ms=3, mfc="white", mec=INK, mew=0.4)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        axes[0, column].set_title(name.capitalize(), fontsize=8, pad=3)
    axes[0, 0].set_ylabel("Learned latent distance", fontsize=7.5)
    axes[1, 0].set_ylabel("True path length", fontsize=7.5)
    fig.subplots_adjust(left=0.035, right=0.995, top=0.94, bottom=0.01, wspace=0.04, hspace=0.05)
    fig.savefig(FIGURES / "distance_maps.png")
    plt.close(fig)


def power_figure(rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.65))
    for course, color, marker in (("gate", BLUE, "o"), ("slalom", ORANGE, "s")):
        powers = (0.25, 0.5, 1.0)
        for ax, key in zip(axes, ("wall_gap", "success")):
            points = []
            for power in powers:
                group = [float(r[key]) for r in rows if r["course"] == course and (
                    (r["setting"] == "power_scale" and float(r["value"]) == power)
                    or (r["setting"] == "baseline" and power == 1.0))]
                points.append(group)
            means = [np.mean(group) for group in points]
            sds = [np.std(group, ddof=1) for group in points]
            ax.errorbar(powers, means, yerr=sds, color=color, marker=marker, ms=3, lw=1.1, capsize=1.2,
                        elinewidth=0.6, label=course.capitalize())
    axes[0].axhline(1.0, color=MUTED, lw=0.5, ls=(0, (3, 2)))
    axes[0].set_ylabel("Wall gap")
    axes[0].set_ylim(0, None)
    axes[1].set_ylabel("Success rate")
    axes[1].set_ylim(0, 1.05)
    axes[0].legend(loc="upper left", fontsize=6.5, handlelength=1.4, borderaxespad=0.1)
    for ax in axes:
        ax.set_xticks([0.25, 0.5, 1.0], ["1/4", "1/2", "Full"])
        ax.set_xlabel("Shot power")
        ax.grid(axis="y", color=GRID, lw=0.4)
        ax.set_axisbelow(True)
    fig.tight_layout(pad=0.3, w_pad=0.8)
    fig.savefig(FIGURES / "power.png")
    plt.close(fig)


def wall_profile() -> None:
    """Latent step across the wall row, for the real wall and the ghost wall."""
    from train_models import CHECKPOINTS

    positions = np.arange(0.08, 0.84, 0.02)
    start = np.stack([positions, np.full_like(positions, 0.6)], 1)
    end = start + [0.09, 0.0]
    fig, ax = plt.subplots(figsize=(3.4, 1.55))
    variants = (("Real wall", BLUE, lambda seed: path_for("gate", seed)),
                ("Ghost wall", ORANGE, lambda seed: CHECKPOINTS / "ablation" / f"gate_ghost_seed{seed}.pt"))
    for label, color, path in variants:
        curves = []
        for seed in SEEDS:
            model = load(path(seed))
            curves.append((model.encode(start) - model.encode(end)).norm(dim=1).numpy())
        curves = np.array(curves)
        # Pairs with an end inside the real wall are unreachable there, so both are masked.
        solid = ~(is_free(COURSES["gate"], start) & is_free(COURSES["gate"], end))
        shown = np.where(solid, np.nan, curves.mean(0))
        ax.plot(positions + 0.045, shown, color=color, lw=1.2, marker="o", ms=1.8, label=label)
    ax.axvspan(0.485, 0.515, color="#3d3f46", alpha=0.25, lw=0)
    ax.set_xlabel("Midpoint of a pair 0.09 apart, along y = 0.6")
    ax.set_ylabel("Latent distance")
    ax.set_ylim(0, None)
    ax.grid(axis="y", color=GRID, lw=0.4)
    ax.legend(loc="upper left", fontsize=6.5, handlelength=1.4, borderaxespad=0.1)
    fig.tight_layout(pad=0.3)
    fig.savefig(FIGURES / "wall_profile.png")
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    episodes, shapes = read("episodes_seed*.csv"), read("geometry_all.csv")
    ablations, ghosts = read("ablations.csv"), read("ghost_control.csv")
    summary, lines = planner_table(episodes)
    with (RESULTS / "planner_summary.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    lines = ["# Results summary", "", "Three model seeds per course and 20 episodes per seed. Seeds are the"
             " replication unit.", ""] + lines
    lines += geometry_table(shapes) + ablation_table(ablations) + ghost_table(ghosts)
    (RESULTS / "summary.md").write_text("\n".join(lines) + "\n")
    distance_maps()
    power_figure(ablations)
    wall_profile()
    print("Wrote results/summary.md and report figures")


if __name__ == "__main__":
    main()
