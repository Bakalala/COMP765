"""Planners and an episode loop. Every planner sees the ball only as its model allows."""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from golf import FRICTION, MAX_SPEED, Course, simulate

MAX_SHOTS = 10


def project_disc(actions: torch.Tensor) -> torch.Tensor:
    return actions / actions.norm(dim=-1, keepdim=True).clamp(min=1.0)


def cem(cost: Callable[[torch.Tensor], torch.Tensor], horizon: int, *, samples: int = 300, elites: int = 30,
        iterations: int = 6, generator: torch.Generator | None = None) -> torch.Tensor:
    """Cross-entropy search over shot sequences; returns the best mean sequence."""
    mean = torch.zeros(horizon, 2)
    std = torch.full((horizon, 2), 0.7)
    for _ in range(iterations):
        noise = torch.randn(samples, horizon, 2, generator=generator)
        actions = project_disc(mean + std * noise)
        best = actions[cost(actions).topk(elites, largest=False).indices]
        mean = best.mean(0)
        std = best.std(0) + 0.02
    return project_disc(mean)


def latent_planner(model, goal_pos: np.ndarray, horizon: int, generator: torch.Generator, terminal: bool = True):
    """LeWorldModel planning: reach the goal image's latent at the end of the horizon."""
    goal = model.encode(goal_pos[None])[0]

    def choose(pos: np.ndarray) -> np.ndarray:
        latent = model.encode(pos[None])

        def cost(actions: torch.Tensor) -> torch.Tensor:
            imagined = model.rollout(latent, actions)
            distance = (imagined - goal).pow(2).sum(-1)
            return distance[:, -1] if terminal else distance.min(dim=1).values

        return cem(cost, horizon, generator=generator)[0].numpy()

    return choose


def run_episode(course: Course, choose: Callable[[np.ndarray], np.ndarray], max_shots: int = MAX_SHOTS,
                power_scale: float = 1.0) -> dict:
    pos = np.array(course.start, dtype=np.float64)
    trail = [pos.copy()]
    for shot in range(1, max_shots + 1):
        action = np.asarray(choose(pos), dtype=np.float64)
        final, sunk = simulate(course, pos[None], power_scale * action[None])
        pos = final[0]
        trail.append(pos.copy())
        if sunk[0]:
            return {"sunk": True, "shots": shot, "trail": trail}
    return {"sunk": False, "shots": max_shots, "trail": trail}


def random_planner(rng: np.random.Generator):
    def choose(pos: np.ndarray) -> np.ndarray:
        angle = rng.uniform(0, 2 * np.pi)
        return np.sqrt(rng.uniform()) * np.array([np.cos(angle), np.sin(angle)])

    return choose


def aim_planner(course: Course):
    """Shoot straight at the hole with just enough power, ignoring walls."""
    hole = np.array(course.hole)

    def choose(pos: np.ndarray) -> np.ndarray:
        offset = hole - pos
        distance = np.linalg.norm(offset)
        speed = np.sqrt(2 * FRICTION * distance) / MAX_SPEED
        return min(speed, 1.0) * offset / max(distance, 1e-9)

    return choose


def simulator_planner(course: Course, position_cost: Callable[[np.ndarray], np.ndarray], horizon: int,
                      generator: torch.Generator):
    """Oracle dynamics: CEM through the true physics with a chosen position cost."""

    def choose(pos: np.ndarray) -> np.ndarray:
        def cost(actions: torch.Tensor) -> torch.Tensor:
            batch = actions.numpy().astype(np.float64)
            current = np.repeat(pos[None], len(batch), axis=0)
            best = np.full(len(batch), np.inf)
            for step in range(batch.shape[1]):
                current, sunk = simulate(course, current, batch[:, step])
                best = np.minimum(best, np.where(sunk, -1.0, position_cost(current)))
            return torch.tensor(best)

        return cem(cost, horizon, generator=generator)[0].numpy()

    return choose


class LatentGraph:
    """Dyna-style planning in latent space.

    Latent states are clustered, each observed or imagined transition becomes an edge
    between clusters, and value iteration from the goal gives shots-to-go per cluster.
    """

    def __init__(self, centers: torch.Tensor, edges: np.ndarray, goal: torch.Tensor):
        self.centers = centers
        count = len(centers)
        goal_cluster = int(torch.cdist(goal[None], centers).argmin())
        value = np.full(count, np.inf)
        value[goal_cluster] = 0.0
        for _ in range(count):
            updated = value.copy()
            np.minimum.at(updated, edges[:, 0], value[edges[:, 1]] + 1.0)
            if np.array_equal(updated, value):
                break
            value = updated
        reachable = np.isfinite(value)
        self.value = torch.tensor(np.where(reachable, value, value[reachable].max() + 2.0), dtype=torch.float32)

    def cost(self, latent: torch.Tensor, neighbours: int = 4) -> torch.Tensor:
        distance = torch.cdist(latent, self.centers)
        near = distance.topk(neighbours, largest=False)
        weights = torch.softmax(-near.values / near.values.mean().clamp(min=1e-6) * 4.0, dim=-1)
        return (weights * self.value[near.indices]).sum(-1)


def kmeans(points: torch.Tensor, clusters: int, generator: torch.Generator, iterations: int = 25) -> torch.Tensor:
    centers = points[torch.randperm(len(points), generator=generator)[:clusters]].clone()
    for _ in range(iterations):
        labels = torch.cdist(points, centers).argmin(1)
        for index in range(clusters):
            members = points[labels == index]
            if len(members):
                centers[index] = members.mean(0)
    return centers


def build_graph(model, data: dict, goal_pos: np.ndarray, *, clusters: int = 300, source: str = "replay",
                seed: int = 0, min_count: int = 1, imagined_per_cluster: int = 200) -> LatentGraph:
    """Build the graph from stored transitions (replay) or from the predictor (model)."""
    generator = torch.Generator().manual_seed(seed)
    now = model.encode(data["start"])
    centers = kmeans(now[:20000], clusters, generator)
    if source == "replay":
        after = model.encode(data["final"])
        first = torch.cdist(now, centers).argmin(1).numpy()
        second = torch.cdist(after, centers).argmin(1).numpy()
    elif source == "model":
        rng = np.random.default_rng(seed)
        angle = rng.uniform(0, 2 * np.pi, (clusters, imagined_per_cluster))
        power = np.sqrt(rng.uniform(0, 1, (clusters, imagined_per_cluster)))
        actions = torch.tensor(np.stack([power * np.cos(angle), power * np.sin(angle)], -1), dtype=torch.float32)
        with torch.no_grad():
            imagined = model.predictor(centers[:, None].expand(-1, imagined_per_cluster, -1), actions)
        second = torch.cdist(imagined.reshape(-1, centers.shape[1]), centers).argmin(1).numpy()
        first = np.repeat(np.arange(clusters), imagined_per_cluster)
    else:
        raise ValueError(source)
    pairs, counts = np.unique(np.stack([first, second], 1), axis=0, return_counts=True)
    return LatentGraph(centers, pairs[counts >= min_count], model.encode(goal_pos[None])[0])


class NeighbourGraph:
    """The nearest-neighbour graph baseline: shortest paths over latent distances.

    Follows the idea of Latent World Models Should Plan with Distances Along the
    Latent Manifold: mutual k-nearest neighbours, Euclidean edge weights, no actions.
    """

    def __init__(self, nodes: torch.Tensor, goal: torch.Tensor, neighbours: int = 15):
        self.nodes = torch.cat([nodes, goal[None]])
        count = len(self.nodes)
        distance = torch.cdist(self.nodes, self.nodes)
        near = distance.topk(neighbours + 1, largest=False).indices[:, 1:].numpy()
        linked = np.zeros((count, count), dtype=bool)
        linked[np.arange(count)[:, None], near] = True
        linked &= linked.T
        first, second = np.nonzero(linked)
        weights = distance.numpy()[first, second].astype(np.float64)
        value = np.full(count, np.inf)
        value[-1] = 0.0
        for _ in range(count):
            updated = value.copy()
            np.minimum.at(updated, first, value[second] + weights)
            if np.allclose(updated, value, equal_nan=True):
                break
            value = updated
        finite = np.isfinite(value)
        self.value = torch.tensor(np.where(finite, value, value[finite].max() * 1.5), dtype=torch.float32)

    def cost(self, latent: torch.Tensor, neighbours: int = 4) -> torch.Tensor:
        near = torch.cdist(latent, self.nodes).topk(neighbours, largest=False)
        weights = torch.softmax(-near.values / near.values.mean().clamp(min=1e-6) * 4.0, dim=-1)
        return (weights * self.value[near.indices]).sum(-1)


def graph_planner(model, graph, goal_pos: np.ndarray, horizon: int, generator: torch.Generator,
                  tie_weight: float = 0.02):
    """CEM as before, with the graph's shots-to-go replacing straight-line latent distance."""
    goal = model.encode(goal_pos[None])[0]

    def choose(pos: np.ndarray) -> np.ndarray:
        latent = model.encode(pos[None])

        def cost(actions: torch.Tensor) -> torch.Tensor:
            imagined = model.rollout(latent, actions)
            flat = imagined.reshape(-1, imagined.shape[-1])
            steps = graph.cost(flat).view(imagined.shape[:2])
            near = (imagined - goal).pow(2).sum(-1)
            return (steps + tie_weight * near).min(dim=1).values

        return cem(cost, horizon, generator=generator)[0].numpy()

    return choose


def finishing_planner(model, graph, goal_pos: np.ndarray, generator: torch.Generator, threshold: float = 1.5):
    """Follow the graph while far away, then use exact latent distance for the final shots."""
    route = graph_planner(model, graph, goal_pos, 1, generator)
    finish = latent_planner(model, goal_pos, 1, generator)

    def choose(pos: np.ndarray) -> np.ndarray:
        return finish(pos) if graph.cost(model.encode(pos[None]))[0] <= threshold else route(pos)

    return choose
