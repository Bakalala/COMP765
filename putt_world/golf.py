"""Continuous top-down mini-golf: friction, wall bounces, sand, and a hole.

The unit of interaction is a shot. A ball at rest receives a velocity, the physics
runs until the ball stops or sinks, and the next observation is the resting image.
All functions are vectorized over balls so planners can test many shots at once.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from heapq import heappop, heappush

import numpy as np

MAX_SPEED = 1.6
FRICTION = 1.5
SAND_FRICTION = 5.0
RESTITUTION = 0.8
DT = 0.008
MAX_TICKS = 260
HOLE_RADIUS = 0.04
CAPTURE_SPEED = 0.9
BALL_RADIUS = 0.018
WALL = 0.03
IMAGE_SIZE = 64


@dataclass(frozen=True)
class Course:
    name: str
    start: tuple[float, float]
    hole: tuple[float, float]
    # Axis-aligned rectangles as (x0, y0, x1, y1), in the unit square.
    walls: tuple[tuple[float, float, float, float], ...] = ()
    sand: tuple[tuple[float, float, float, float], ...] = field(default=())
    # Drawn exactly like walls, but the ball rolls through them. Used as a control.
    ghosts: tuple[tuple[float, float, float, float], ...] = field(default=())

    @property
    def solid(self) -> np.ndarray:
        border = (
            (0.0, 0.0, 1.0, WALL), (0.0, 1.0 - WALL, 1.0, 1.0),
            (0.0, 0.0, WALL, 1.0), (1.0 - WALL, 0.0, 1.0, 1.0),
        )
        return np.array(border + self.walls, dtype=np.float64)


COURSES = {
    "open": Course("open", (0.15, 0.5), (0.85, 0.5), sand=((0.42, 0.62, 0.62, 0.86),)),
    "gate": Course(
        "gate", (0.15, 0.5), (0.85, 0.5),
        walls=((0.485, 0.25, 0.515, 1.0),),
    ),
    "slalom": Course(
        "slalom", (0.12, 0.5), (0.88, 0.5),
        walls=((0.32, 0.27, 0.35, 1.0), (0.65, 0.0, 0.68, 0.73)),
    ),
    "gate_ghost": Course(
        "gate_ghost", (0.15, 0.5), (0.85, 0.5),
        ghosts=((0.485, 0.25, 0.515, 1.0),),
    ),
    "pocket": Course(
        "pocket", (0.15, 0.5), (0.75, 0.5),
        walls=((0.56, 0.28, 0.59, 0.72), (0.56, 0.28, 0.9, 0.31), (0.56, 0.69, 0.9, 0.72)),
    ),
}


MAIN_COURSES = ("open", "gate", "slalom", "pocket")


def in_rects(pos: np.ndarray, rects: np.ndarray, margin: float = 0.0) -> np.ndarray:
    if len(rects) == 0:
        return np.zeros(len(pos), dtype=bool)
    x, y = pos[:, 0, None], pos[:, 1, None]
    inside = ((x > rects[:, 0] - margin) & (x < rects[:, 2] + margin)
              & (y > rects[:, 1] - margin) & (y < rects[:, 3] + margin))
    return inside.any(axis=1)


def is_free(course: Course, pos: np.ndarray) -> np.ndarray:
    inside = ((pos >= 0.0) & (pos <= 1.0)).all(axis=1)
    return inside & ~in_rects(pos, course.solid, BALL_RADIUS)


def sample_free(course: Course, count: int, rng: np.random.Generator) -> np.ndarray:
    points = np.empty((0, 2))
    while len(points) < count:
        batch = rng.uniform(0.0, 1.0, (2 * count, 2))
        points = np.concatenate([points, batch[is_free(course, batch)]])
    return points[:count]


def simulate(course: Course, pos: np.ndarray, action: np.ndarray, return_paths: bool = False):
    """Play one shot per ball. An action is a velocity in units of MAX_SPEED.

    Returns final positions, a sunk flag, and optionally the path of each ball.
    """
    pos = np.array(pos, dtype=np.float64)
    action = np.asarray(action, dtype=np.float64)
    norm = np.linalg.norm(action, axis=1, keepdims=True)
    vel = MAX_SPEED * action / np.maximum(norm, 1.0)
    solid = course.solid
    sand = np.array(course.sand, dtype=np.float64).reshape(-1, 4)
    hole = np.array(course.hole)
    sunk = np.zeros(len(pos), dtype=bool)
    moving = np.linalg.norm(vel, axis=1) > 1e-6
    paths = [pos.copy()] if return_paths else None

    for _ in range(MAX_TICKS):
        if not moving.any():
            break
        # Move each axis separately so a wall reflects only the blocked component.
        for axis in (0, 1):
            trial = pos.copy()
            trial[:, axis] += vel[:, axis] * DT
            blocked = moving & in_rects(trial, solid, BALL_RADIUS)
            step = moving & ~blocked
            pos[step, axis] = trial[step, axis]
            vel[blocked, axis] *= -RESTITUTION
        speed = np.linalg.norm(vel, axis=1)
        captured = moving & (np.linalg.norm(pos - hole, axis=1) < HOLE_RADIUS) & (speed < CAPTURE_SPEED)
        sunk |= captured
        pos[captured] = hole
        moving &= ~captured
        friction = np.where(in_rects(pos, sand), SAND_FRICTION, FRICTION)
        slowed = np.maximum(speed - friction * DT, 0.0)
        vel *= (slowed / np.maximum(speed, 1e-9))[:, None]
        moving &= slowed > 1e-3
        if return_paths:
            paths.append(pos.copy())

    near = np.linalg.norm(pos - hole, axis=1) < HOLE_RADIUS
    sunk |= near
    pos[near] = hole
    if return_paths:
        return pos, sunk, np.stack(paths, axis=1)
    return pos, sunk


def background(course: Course, size: int = IMAGE_SIZE) -> np.ndarray:
    """RGB course image in [0, 1], without the ball."""
    centers = (np.arange(size) + 0.5) / size
    xx, yy = np.meshgrid(centers, centers)
    points = np.stack([xx.ravel(), yy.ravel()], axis=1)
    image = np.tile(np.array([0.28, 0.58, 0.32]), (size * size, 1))
    sand = np.array(course.sand, dtype=np.float64).reshape(-1, 4)
    image[in_rects(points, sand)] = (0.86, 0.76, 0.5)
    image[in_rects(points, course.solid)] = (0.24, 0.25, 0.3)
    image[in_rects(points, np.array(course.ghosts, dtype=np.float64).reshape(-1, 4))] = (0.24, 0.25, 0.3)
    image[np.linalg.norm(points - np.array(course.hole), axis=1) < HOLE_RADIUS] = (0.04, 0.05, 0.05)
    return image.reshape(size, size, 3)


def geodesic_map(course: Course, size: int = 96) -> np.ndarray:
    """Shortest free-space path length from every grid cell to the hole."""
    centers = (np.arange(size) + 0.5) / size
    xx, yy = np.meshgrid(centers, centers)
    free = is_free(course, np.stack([xx.ravel(), yy.ravel()], axis=1)).reshape(size, size)
    hole = np.array(course.hole)
    start = (int(hole[1] * size), int(hole[0] * size))
    dist = np.full((size, size), np.inf)
    dist[start] = 0.0
    heap = [(0.0, start)]
    moves = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx]
    while heap:
        d, (y, x) = heappop(heap)
        if d > dist[y, x]:
            continue
        for dy, dx in moves:
            ny, nx = y + dy, x + dx
            if 0 <= ny < size and 0 <= nx < size and free[ny, nx]:
                nd = d + np.hypot(dy, dx) / size
                if nd < dist[ny, nx]:
                    dist[ny, nx] = nd
                    heappush(heap, (nd, (ny, nx)))
    return dist


def lookup(grid: np.ndarray, pos: np.ndarray) -> np.ndarray:
    size = grid.shape[0]
    cells = np.clip((pos * size).astype(int), 0, size - 1)
    return grid[cells[:, 1], cells[:, 0]]
