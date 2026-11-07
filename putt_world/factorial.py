"""Separate the learned dynamics from the cost, and record supporting measurements."""

from __future__ import annotations

import csv
from dataclasses import replace

import numpy as np
import torch

from evaluate import RESULTS, starts
from golf import COURSES, MAIN_COURSES, geodesic_map, in_rects, is_free, lookup, sample_free, simulate
from model import load, make_dataset, random_shots
from plan import build_graph, cem, latent_planner, run_episode
from train_models import CHECKPOINTS, SEEDS, path_for


def physics_planner(course, cost_of_positions, generator):
    """One-step CEM through the true physics, with any cost on resting positions."""

    def choose(pos: np.ndarray) -> np.ndarray:
        def cost(actions: torch.Tensor) -> torch.Tensor:
            shots = actions[:, 0].numpy().astype(np.float64)
            final, _ = simulate(course, np.repeat(pos[None], len(shots), 0), shots)
            return cost_of_positions(final)

        return cem(cost, 1, generator=generator)[0].numpy()

    return choose


def fit_probe(model, course, seed: int) -> np.ndarray:
    pos = sample_free(course, 4000, np.random.default_rng(seed + 50))
    design = np.c_[model.encode(pos).numpy(), np.ones(len(pos))]
    return np.linalg.lstsq(design, pos, rcond=None)[0]


def decode(latent: torch.Tensor, probe: np.ndarray) -> np.ndarray:
    return np.c_[latent.numpy(), np.ones(len(latent))] @ probe


def predictor_path_planner(model, geodesic, probe, generator):
    """One-step CEM through the predictor, with true path length of the decoded outcome."""

    def choose(pos: np.ndarray) -> np.ndarray:
        latent = model.encode(pos[None])

        def cost(actions: torch.Tensor) -> torch.Tensor:
            imagined = model.rollout(latent, actions)[:, -1]
            return torch.tensor(np.nan_to_num(lookup(geodesic, np.clip(decode(imagined, probe), 0, 1)), posinf=9.0))

        return cem(cost, 1, generator=generator)[0].numpy()

    return choose


def wall_contact(course, start: np.ndarray, action: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Whether interior walls change where a shot ends."""
    final, _ = simulate(course, start, scale * action)
    without = replace(course, walls=())
    free_final, _ = simulate(without, start, scale * action)
    return np.linalg.norm(final - free_final, axis=1) > 1e-6


def effective_rank(model, seed: int) -> float:
    pos = sample_free(model.course, 4000, np.random.default_rng(seed))
    latent = model.encode(pos)
    values = torch.linalg.svdvals(latent - latent.mean(0))
    weights = values / values.sum()
    return float(torch.exp(-(weights * torch.log(weights + 1e-12)).sum()))


def main() -> None:
    episodes, measures = [], []
    for seed in SEEDS:
        for name in MAIN_COURSES:
            model = load(path_for(name, seed))
            course = COURSES[name]
            hole = np.array(course.hole)
            goal = model.encode(hole[None])[0]
            geodesic = geodesic_map(course)
            probe = fit_probe(model, course, seed)
            graph = build_graph(model, make_dataset(course, 60000, seed), hole, source="replay", seed=seed)
            generator = torch.Generator().manual_seed(seed)
            planners = {
                "predictor, latent L2": latent_planner(model, hole, 1, generator),
                "physics, latent L2": physics_planner(
                    course, lambda p: (model.encode(p) - goal).pow(2).sum(-1), generator),
                "predictor, path length": predictor_path_planner(model, geodesic, probe, generator),
                "physics, shot graph": physics_planner(
                    course, lambda p: graph.cost(model.encode(p)) + 0.02 * (model.encode(p) - goal).pow(2).sum(-1),
                    generator),
            }
            for planner, choose in planners.items():
                for index, start in enumerate(starts(course, seed)):
                    outcome = run_episode(replace(course, start=tuple(start)), choose)
                    episodes.append({"course": name, "seed": seed, "planner": planner, "episode": index,
                                     "start_x": start[0], "start_y": start[1], "sunk": int(outcome["sunk"]),
                                     "shots": outcome["shots"]})
            rng = np.random.default_rng(seed + 90)
            start = sample_free(course, 6000, rng)
            action = random_shots(6000, rng)
            final, _ = simulate(course, start, action)
            touched = wall_contact(course, start, action) if course.walls else np.zeros(6000, dtype=bool)
            with torch.no_grad():
                imagined = model.predictor(model.encode(start), torch.tensor(action, dtype=torch.float32))
            error = np.linalg.norm(decode(imagined, probe) - final, axis=1)
            measures.append({"course": name, "seed": seed,
                             "error_wall_shots": float(np.median(error[touched])) if touched.any() else float("nan"),
                             "error_other_shots": float(np.median(error[~touched])),
                             "effective_rank": effective_rank(model, seed)})
            print(name, seed, flush=True)
    involvement = []
    for name in ("gate", "slalom"):
        rng = np.random.default_rng(7)
        start = sample_free(COURSES[name], 20000, rng)
        action = random_shots(20000, rng)
        for scale in (1.0, 0.5, 0.25):
            involvement.append({"course": name, "power": scale,
                                "wall_changes_outcome": float(wall_contact(COURSES[name], start, action, scale).mean())})
    ranks = []
    for value in (0.01, 1.0):
        for seed in (0, 1):
            model = load(CHECKPOINTS / "ablation" / f"gate_weight_{value}_seed{seed}.pt")
            ranks.append({"weight": value, "seed": seed, "effective_rank": effective_rank(model, seed)})
    for filename, rows in (("factorial.csv", episodes), ("predictor_error.csv", measures),
                           ("wall_involvement.csv", involvement), ("weight_rank.csv", ranks)):
        with (RESULTS / filename).open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
