"""Evaluate planners and latent geometry for every trained model."""

from __future__ import annotations

import csv
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from golf import COURSES, MAIN_COURSES, geodesic_map, in_rects, is_free, lookup, sample_free
from model import load, make_dataset
from plan import (NeighbourGraph, aim_planner, build_graph, finishing_planner, graph_planner, latent_planner,
                  random_planner, run_episode, simulator_planner)
from train_models import SEEDS, path_for

RESULTS = Path(__file__).parent / "results"
EPISODES = 20


def spearman(first: np.ndarray, second: np.ndarray) -> float:
    def rank(values: np.ndarray) -> np.ndarray:
        order = np.empty(len(values))
        order[values.argsort()] = np.arange(len(values))
        return order

    return float(np.corrcoef(rank(first), rank(second))[0, 1])


def starts(course, seed: int) -> np.ndarray:
    """The course start, then random resting positions shared by all planners."""
    rng = np.random.default_rng(1000 + seed)
    return np.concatenate([np.array([course.start]), sample_free(course, EPISODES - 1, rng)])


def wall_gap(model, rng: np.random.Generator, separation: float = 0.09, count: int = 8000) -> float:
    """How much larger the latent step is across a wall than beside it.

    Each pair that straddles a wall is compared with the same pair slid along its own
    axis, by one separation, so that it no longer straddles. Location and direction are
    therefore matched. A value of 1 means the latent space ignores the wall.
    """
    course = model.course
    ghosts = np.array(course.ghosts, dtype=np.float64).reshape(-1, 4)
    fractions = np.linspace(0, 1, 12)[None, :, None]

    def open_space(points: np.ndarray) -> np.ndarray:
        return is_free(course, points) & ~in_rects(points, ghosts, 0.018)

    def clear(first: np.ndarray, second: np.ndarray) -> np.ndarray:
        line = (first[:, None] * (1 - fractions) + second[:, None] * fractions).reshape(-1, 2)
        return open_space(line).reshape(len(first), -1).all(1)

    first = sample_free(course, count, rng)
    angle = rng.uniform(0, 2 * np.pi, count)
    step = separation * np.stack([np.cos(angle), np.sin(angle)], 1)
    second = first + step
    straddles = open_space(first) & open_space(second) & ~clear(first, second)
    if straddles.sum() < 10:
        return float("nan")
    first, second, step = first[straddles], second[straddles], step[straddles]
    before = clear(first - step, first)
    after = clear(second, second + step)
    keep = before & after
    if keep.sum() < 10:
        return float("nan")
    first, second, step = first[keep], second[keep], step[keep]

    def latent_step(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return (model.encode(a) - model.encode(b)).norm(dim=1).numpy()

    beside = 0.5 * (latent_step(first - step, first) + latent_step(second, second + step))
    return float(latent_step(first, second).mean() / beside.mean())


def geometry(model, seed: int) -> dict:
    course = model.course
    hole = np.array(course.hole)
    size = 64
    centers = (np.arange(size) + 0.5) / size
    xx, yy = np.meshgrid(centers, centers)
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1)
    pos = grid[is_free(course, grid)]
    latent = model.encode(pos)
    goal = model.encode(hole[None])[0]
    latent_distance = (latent - goal).norm(dim=1).numpy()
    straight = np.linalg.norm(pos - hole, axis=1)
    geodesic = lookup(geodesic_map(course), pos)
    keep = np.isfinite(geodesic)
    near = keep & (geodesic < 0.2)
    design = np.c_[latent.numpy(), np.ones(len(pos))]
    probe = np.linalg.lstsq(design, pos, rcond=None)[0]
    return {
        "course": course.name, "seed": seed,
        "spearman_straight": spearman(latent_distance[keep], straight[keep]),
        "spearman_geodesic": spearman(latent_distance[keep], geodesic[keep]),
        "spearman_geodesic_near": spearman(latent_distance[near], geodesic[near]),
        "wall_gap": wall_gap(model, np.random.default_rng(seed)),
        "detour_ratio": float((geodesic[keep] / np.maximum(straight[keep], 1e-6)).mean()),
        "probe_rmse": float(np.sqrt(((design @ probe - pos) ** 2).mean())),
    }


def planners(model, seed: int) -> dict:
    course = model.course
    hole = np.array(course.hole)
    data = make_dataset(course, 60000, seed)
    generator = torch.Generator().manual_seed(seed)
    geodesic = geodesic_map(course)
    nodes = model.encode(data["start"][:1500])
    goal = model.encode(hole[None])[0]
    replay = build_graph(model, data, hole, source="replay", seed=seed)
    imagined = build_graph(model, data, hole, source="model", seed=seed)
    neighbour = NeighbourGraph(nodes, goal)
    return {
        "random": random_planner(np.random.default_rng(seed)),
        "aim": aim_planner(course),
        "true physics, straight cost": simulator_planner(
            course, lambda p: np.linalg.norm(p - hole, axis=1), 1, generator),
        "true physics, geodesic cost": simulator_planner(course, lambda p: lookup(geodesic, p), 1, generator),
        "latent L2, H=1": latent_planner(model, hole, 1, generator),
        "latent L2, H=2": latent_planner(model, hole, 2, generator),
        "latent L2, H=3": latent_planner(model, hole, 3, generator),
        "latent L2 anytime, H=3": latent_planner(model, hole, 3, generator, terminal=False),
        "neighbour graph, H=1": graph_planner(model, neighbour, hole, 1, generator, tie_weight=0.0),
        "shot graph replay, H=1": graph_planner(model, replay, hole, 1, generator),
        "shot graph model, H=1": graph_planner(model, imagined, hole, 1, generator),
    }


def main() -> None:
    """Evaluate one seed, so seeds can run side by side: python evaluate.py 0"""
    RESULTS.mkdir(exist_ok=True)
    torch.set_num_threads(3)
    if sys.argv[1:] == ["geometry"]:
        rows = [geometry(load(path_for(name, seed)), seed) for seed in SEEDS for name in MAIN_COURSES]
        with (RESULTS / "geometry_all.csv").open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return
    if sys.argv[1:] == ["finish"]:
        rows = []
        for seed in SEEDS:
            for name in MAIN_COURSES:
                model = load(path_for(name, seed))
                hole = np.array(model.course.hole)
                graph = build_graph(model, make_dataset(model.course, 60000, seed), hole, source="replay", seed=seed)
                generator = torch.Generator().manual_seed(seed)
                choose = finishing_planner(model, graph, hole, generator)
                for index, start in enumerate(starts(model.course, seed)):
                    outcome = run_episode(replace(model.course, start=tuple(start)), choose)
                    rows.append({"course": name, "seed": seed, "planner": "shot graph + L2 finish (post hoc)",
                                 "episode": index, "sunk": int(outcome["sunk"]), "shots": outcome["shots"]})
                print(name, seed, flush=True)
        with (RESULTS / "episodes_seedfinish.csv").open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return
    seeds = [int(value) for value in sys.argv[1:]] or SEEDS
    episodes: list[dict] = []
    shapes: list[dict] = []
    for seed in seeds:
        for name in MAIN_COURSES:
            course = COURSES[name]
            path = path_for(name, seed)
            if not path.exists():
                continue
            began = time.time()
            model = load(path)
            shapes.append(geometry(model, seed))
            for planner, choose in planners(model, seed).items():
                for index, start in enumerate(starts(course, seed)):
                    outcome = run_episode(replace(course, start=tuple(start)), choose)
                    episodes.append({"course": name, "seed": seed, "planner": planner, "episode": index,
                                     "sunk": int(outcome["sunk"]), "shots": outcome["shots"]})
            print(f"{name} seed {seed} done in {time.time() - began:.0f}s", flush=True)
    tag = "_".join(str(seed) for seed in seeds)
    for filename, rows in ((f"episodes_seed{tag}.csv", episodes), (f"geometry_seed{tag}.csv", shapes)):
        with (RESULTS / filename).open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
