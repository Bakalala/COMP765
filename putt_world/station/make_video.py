"""Render a short station clip: the agent playing beside its learned distance map."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FFMpegWriter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from golf import COURSES, background, is_free, simulate  # noqa: E402
from model import load  # noqa: E402
from plan import latent_planner  # noqa: E402
from train_models import path_for  # noqa: E402

FPS = 30
STRIDE = 3
HOLD = 25
CLIPS = ("gate", "slalom")


def distance_map(model, size: int = 96) -> np.ndarray:
    centers = (np.arange(size) + 0.5) / size
    xx, yy = np.meshgrid(centers, centers)
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1)
    free = is_free(model.course, grid)
    goal = model.encode(np.array(model.course.hole)[None])[0]
    values = np.full(size * size, np.nan)
    values[free] = (model.encode(grid[free]) - goal).norm(dim=1).numpy()
    return values.reshape(size, size)


def play(model, seed: int = 0, max_shots: int = 10) -> tuple[list[np.ndarray], int, bool]:
    course = model.course
    generator = torch.Generator().manual_seed(seed)
    choose = latent_planner(model, np.array(course.hole), 1, generator)
    pos = np.array(course.start)
    legs = []
    for shot in range(1, max_shots + 1):
        final, sunk, path = simulate(course, pos[None], np.asarray(choose(pos))[None], return_paths=True)
        legs.append(path[0])
        pos = final[0]
        if sunk[0]:
            return legs, shot, True
    return legs, max_shots, False


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8, 4.4), dpi=150)
    fig.patch.set_facecolor("#14231a")
    writer = FFMpegWriter(fps=FPS, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
    with writer.saving(fig, str(Path(__file__).parent / "putt_world_clip.mp4"), dpi=150):
        for name in CLIPS:
            model = load(path_for(name, 0))
            course = COURSES[name]
            legs, shots, sunk = play(model)
            left, right = axes
            for ax in axes:
                ax.clear()
                ax.axis("off")
            left.imshow(background(course, 256), extent=(0, 1, 1, 0))
            colormap = plt.get_cmap("viridis").copy()
            colormap.set_bad("#3d3f46")
            right.imshow(np.ma.masked_invalid(distance_map(model)), cmap=colormap, extent=(0, 1, 1, 0))
            left.set_title(f"{name.capitalize()}: the agent plans from pixels", color="white", fontsize=11)
            right.set_title("Its learned distance to the hole", color="white", fontsize=11)
            marks = [ax.plot([], [], "o", color="white", mec="#222", ms=7, zorder=5)[0] for ax in axes]
            traces = [ax.plot([], [], color="#ffd34e", lw=1.2, alpha=0.9)[0] for ax in axes]
            seen = np.empty((0, 2))
            fig.tight_layout()
            for leg in legs:
                for index in list(range(0, len(leg), STRIDE)) + [len(leg) - 1]:
                    current = np.concatenate([seen, leg[: index + 1]])
                    for mark, trace in zip(marks, traces):
                        mark.set_data([leg[index, 0]], [leg[index, 1]])
                        trace.set_data(current[:, 0], current[:, 1])
                    writer.grab_frame()
                seen = np.concatenate([seen, leg])
                for _ in range(HOLD):
                    writer.grab_frame()
            print(name, "shots", shots, "sunk", sunk, flush=True)
            for _ in range(2 * HOLD):
                writer.grab_frame()


if __name__ == "__main__":
    main()
