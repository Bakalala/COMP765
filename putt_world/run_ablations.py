"""Train and measure ablations: shot power, SIGReg weight, and dataset size."""

from __future__ import annotations

import csv
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from evaluate import RESULTS, geometry, starts
from golf import COURSES
from model import WorldModel, load, save, train
from plan import latent_planner, run_episode
from train_models import CHECKPOINTS, path_for

SETTINGS = (
    [("power_scale", value, course, seed) for value in (0.25, 0.5) for course in ("gate", "slalom")
     for seed in (0, 1, 2)]
    + [("weight", value, "gate", seed) for value in (0.01, 1.0) for seed in (0, 1)]
    + [("transitions", value, "gate", seed) for value in (5000, 20000) for seed in (0, 1)]
)
BASELINE = {"power_scale": 1.0, "weight": 0.09, "transitions": 60000}


def measure(model: WorldModel, setting: str, value: float, seed: int) -> dict:
    row = {"setting": setting, "value": value, **geometry(model, seed)}
    max_shots = {1.0: 10, 0.5: 20, 0.25: 40}[model.power_scale]
    hole = np.array(model.course.hole)
    generator = torch.Generator().manual_seed(seed)
    outcomes = [
        run_episode(replace(model.course, start=tuple(start)), latent_planner(model, hole, 1, generator),
                    max_shots, model.power_scale)
        for start in starts(model.course, seed)
    ]
    row["success"] = float(np.mean([outcome["sunk"] for outcome in outcomes]))
    return row


def ghost_control() -> None:
    """A wall drawn in every image but absent from the physics, against the real gate."""
    rows = []
    for seed in (0, 1, 2):
        path = CHECKPOINTS / "ablation" / f"gate_ghost_seed{seed}.pt"
        if not path.exists():
            save(train("gate_ghost", seed=seed), path)
        rows.append({"variant": "ghost wall", **geometry(load(path), seed)})
        rows.append({"variant": "real wall", **geometry(load(path_for("gate", seed)), seed)})
        print(rows[-2:], flush=True)
    with (RESULTS / "ghost_control.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows = []
    for course in ("gate", "slalom"):
        for seed in (0, 1, 2):
            if path_for(course, seed).exists():
                rows.append(measure(load(path_for(course, seed)), "baseline", 1.0, seed))
    for setting, value, course, seed in SETTINGS:
        path = CHECKPOINTS / "ablation" / f"{course}_{setting}_{value}_seed{seed}.pt"
        if not path.exists():
            save(train(course, seed=seed, **{setting: value}), path)
        rows.append(measure(load(path), setting, value, seed))
        print(rows[-1], flush=True)
        with (RESULTS / "ablations.csv").open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    if sys.argv[1:] == ["ghost"]:
        ghost_control()
    else:
        main()
