"""Draw the report figures at their printed size from the saved result files."""

from __future__ import annotations

import csv
import sys
from collections import deque
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from golf_env import MiniGolfEnv  # noqa: E402

OUT = Path(__file__).resolve().parent / "figures"
INK = "#2b2b29"
MUTED = "#6f6e69"
GRID = "#e4e3df"
PLANNING_COLORS = {0: "#2a78d6", 5: "#eb6834", 20: "#1baf7a"}
INIT_STYLES = {0.0: ("#6250d6", "o"), -5.0: ("#eda100", "s"), -20.0: ("#e87ba4", "^")}
COURSE_TITLES = {"easy": "Easy", "wall_gate": "Wall-gate"}
WIDTH = 3.4

plt.rcParams.update({
    "font.family": "serif", "font.size": 7, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.linewidth": 0.5, "xtick.major.width": 0.5,
    "ytick.major.width": 0.5, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "savefig.dpi": 300,
})


def shortest_route(env: MiniGolfEnv) -> tuple[list[list[tuple[int, int]]], set[int]]:
    """Breadth-first search over resting cells; returns one best route and all reachable cells."""
    start = env.reset()
    parents: dict[int, tuple[int, list[tuple[int, int]]] | None] = {start: None}
    queue = deque([start])
    while queue:
        state = queue.popleft()
        if state == env.hole_state:
            continue
        for action in range(env.n_actions):
            env.reset()
            env.state = state
            next_state, _, _, info = env.step(action)
            if next_state not in parents:
                parents[next_state] = (state, info["path"])
                queue.append(next_state)
    legs = []
    state = env.hole_state
    while parents[state] is not None:
        state, path = parents[state]
        legs.append(path)
    return legs[::-1], set(parents)


def draw_course(ax, course: str, title: str) -> None:
    env = MiniGolfEnv(course)
    legs, reachable = shortest_route(env)
    for y in range(env.height):
        for x in range(env.width):
            tile = env.tile_at(x, y)
            color = "#5b5a55" if tile == "#" else "#eaf3ea"
            ax.add_patch(plt.Rectangle((x, y), 1, 1, facecolor=color, edgecolor="white", linewidth=0.3))
            if tile == "B":
                ax.add_patch(plt.Circle((x + .5, y + .5), .3, facecolor="#eda100", edgecolor=INK, linewidth=0.4))
    for state in reachable:
        x, y = env.state_to_cell[state]
        ax.plot(x + .5, y + .5, "o", ms=1.6, color=MUTED)
    for leg in legs:
        xs, ys = zip(*[(x + .5, y + .5) for x, y in leg])
        ax.plot(xs, ys, color="#2a78d6", lw=1.2, solid_capstyle="round")
        ax.plot(xs[-1], ys[-1], "o", ms=2.6, color="#2a78d6")
    sx, sy = env.start_cell
    hx, hy = env.hole_cell
    ax.text(sx + .5, sy + .5, "S", ha="center", va="center", fontsize=6.5, weight="bold",
            bbox=dict(boxstyle="circle,pad=0.12", fc="white", ec=INK, lw=0.4))
    ax.text(hx + .5, hy + .5, "H", ha="center", va="center", fontsize=6.5, weight="bold", color="white",
            bbox=dict(boxstyle="circle,pad=0.12", fc=INK, ec=INK, lw=0.4))
    ax.set_xlim(0, env.width)
    ax.set_ylim(env.height, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title.format(shots=len(legs)), fontsize=7, pad=2)


def course_maps() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 1.62))
    for ax, course in zip(axes, COURSE_TITLES):
        draw_course(ax, course, COURSE_TITLES[course] + ": best route {shots} shots")
    fig.subplots_adjust(left=0.005, right=0.995, top=0.91, bottom=0.01, wspace=0.05)
    fig.savefig(OUT / "course_maps.png")
    plt.close(fig)


def learning_curves() -> None:
    with (ROOT / "results" / "episode_results.csv").open(newline="") as source:
        rows = list(csv.DictReader(source))
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 1.75), sharey=True)
    window = 20
    for ax, course in zip(axes, COURSE_TITLES):
        for k, color in PLANNING_COLORS.items():
            seeds = sorted({row["seed"] for row in rows if row["course"] == course})
            curves = []
            for seed in seeds:
                sinks = [int(row["training_sank"]) for row in rows
                         if row["course"] == course and row["seed"] == seed
                         and int(row["planning_steps"]) == k]
                curves.append(np.convolve(sinks, np.ones(window) / window, mode="valid"))
            curves = np.array(curves)
            episodes = np.arange(window, window + curves.shape[1])
            ax.fill_between(episodes, curves.min(axis=0), curves.max(axis=0), color=color, alpha=0.14, lw=0)
            ax.plot(episodes, curves.mean(axis=0), color=color, lw=1.1, label=f"{k} updates")
        ax.axhline(0.8, color=MUTED, lw=0.5, ls=(0, (3, 2)))
        ax.set_title(COURSE_TITLES[course], fontsize=7, pad=2)
        ax.set_xlabel("Training episode")
        ax.grid(axis="y", color=GRID, lw=0.4)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Sink rate (last 20 episodes)")
    axes[1].legend(loc="lower right", fontsize=6.3, handlelength=1.2, borderaxespad=0.1)
    fig.tight_layout(pad=0.3, w_pad=0.8)
    fig.savefig(OUT / "learning_curves.png")
    plt.close(fig)


def summary_rows() -> list[dict]:
    with (ROOT / "results" / "extended" / "summary.csv").open(newline="") as source:
        return list(csv.DictReader(source))


def planning_budget() -> None:
    rows = [row for row in summary_rows() if row["rng"] == "separate" and float(row["q_init"]) == 0.0]
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 1.55))
    for ax, course in zip(axes, COURSE_TITLES):
        group = [row for row in rows if row["course"] == course]
        positions = np.arange(len(group))
        means = np.array([float(row["threshold_mean"]) for row in group])
        errors = 1.96 * np.array([float(row["threshold_se"]) for row in group])
        ax.errorbar(positions, means, yerr=errors, color="#2a78d6", lw=1.1, marker="o", ms=2.8, capsize=1.2, elinewidth=0.6)
        ax.set_xticks(positions, [row["planning_steps"] for row in group])
        ax.set_ylim(0, means.max() * 1.12)
        ax.set_title(COURSE_TITLES[course], fontsize=7, pad=2)
        ax.set_xlabel("Planning updates per shot")
        ax.grid(axis="y", color=GRID, lw=0.4)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Episodes to threshold")
    fig.tight_layout(pad=0.3, w_pad=0.8)
    fig.savefig(OUT / "planning_budget.png")
    plt.close(fig)


def init_control() -> None:
    rows = [row for row in summary_rows() if row["rng"] == "separate"
            and row["planning_steps"] in {"0", "5", "20"}]
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 1.7))
    for ax, course in zip(axes, COURSE_TITLES):
        for q_init, (color, marker) in INIT_STYLES.items():
            group = [row for row in rows if row["course"] == course and float(row["q_init"]) == q_init]
            positions = np.arange(len(group))
            means = [float(row["threshold_mean"]) for row in group]
            errors = [1.96 * float(row["threshold_se"]) for row in group]
            ax.errorbar(positions, means, yerr=errors, color=color, lw=1.1, marker=marker, ms=3,
                        capsize=1.2, elinewidth=0.6, label=f"Initial Q = {q_init:.0f}")
        ax.set_xticks(np.arange(3), ["0", "5", "20"])
        ax.set_ylim(0, None)
        ax.set_title(COURSE_TITLES[course], fontsize=7, pad=2)
        ax.set_xlabel("Planning updates per shot")
        ax.grid(axis="y", color=GRID, lw=0.4)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Episodes to threshold")
    axes[1].legend(loc="upper right", fontsize=6.3, handlelength=1.4, borderaxespad=0.1)
    fig.tight_layout(pad=0.3, w_pad=0.8)
    fig.savefig(OUT / "init_control.png")
    plt.close(fig)


def ladder_maps() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 1.95))
    for ax, number in zip(axes.flat, range(1, 5)):
        draw_course(ax, f"ladder_{number}", f"Ladder {number}: {{shots}} shots")
    fig.subplots_adjust(left=0.005, right=0.995, top=0.94, bottom=0.01, wspace=0.05, hspace=0.2)
    fig.savefig(OUT / "ladder_maps.png")
    plt.close(fig)


def route_length() -> None:
    with (ROOT / "results" / "extended" / "route_length_summary.csv").open(newline="") as source:
        rows = list(csv.DictReader(source))

    def series(q_init: float, k: int, key: str) -> tuple[list[int], list[float]]:
        group = [row for row in rows if float(row["q_init"]) == q_init and int(row["planning_steps"]) == k]
        return [int(row["best_route_shots"]) for row in group], [float(row[key]) for row in group]

    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 1.8))
    for q_init in (0.0, -5.0):
        color, marker = INIT_STYLES[q_init]
        for k, style in ((0, (0, (3, 2))), (20, "-")):
            routes, means = series(q_init, k, "threshold_mean")
            axes[0].plot(routes, means, color=color, marker=marker, ms=3, lw=1.1, ls=style,
                         label=f"Q={q_init:.0f}, k={k}")
    for q_init, (color, marker) in INIT_STYLES.items():
        routes, solved = series(q_init, 20, "seeds_reaching_threshold")
        axes[1].plot(routes, solved, color=color, marker=marker, ms=3, lw=1.1, label=f"Q={q_init:.0f}, k=20")
    color, marker = INIT_STYLES[-20.0]
    routes, solved = series(-20.0, 0, "seeds_reaching_threshold")
    axes[1].plot(routes, solved, color=color, marker=marker, ms=3, lw=1.1, ls=(0, (3, 2)), label="Q=-20, k=0")
    axes[0].set_ylabel("Episodes to threshold")
    axes[0].set_ylim(0, None)
    axes[1].set_ylabel("Seeds reaching threshold")
    axes[1].set_ylim(0, 108)
    axes[0].legend(loc="upper left", fontsize=6, handlelength=2, borderaxespad=0.1)
    axes[1].legend(loc="lower left", fontsize=6, handlelength=2, borderaxespad=0.1)
    for ax in axes:
        ax.set_xticks([3, 4, 7, 8])
        ax.set_xlabel("Best route (shots)")
        ax.grid(axis="y", color=GRID, lw=0.4)
        ax.set_axisbelow(True)
    fig.tight_layout(pad=0.3, w_pad=0.8)
    fig.savefig(OUT / "route_length.png")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    course_maps()
    learning_curves()
    planning_budget()
    init_control()
    ladder_maps()
    route_length()
    print(f"Wrote figures to {OUT}")


if __name__ == "__main__":
    main()
