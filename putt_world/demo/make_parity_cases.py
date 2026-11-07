"""Write parity_cases.json: reference shots from golf.py for parity_test.mjs.

Run from anywhere: PYTHONDONTWRITEBYTECODE=1 ../.venv/bin/python make_parity_cases.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import golf  # noqa: E402

RANDOM_SHOTS = 1000   # per course, so 4000 in all
NEAR_HOLE_SHOTS = 150  # per course, aimed around the hole to exercise the capture rule
NEAR_SAND_SHOTS = 150  # per course with sand
PATHS = 60             # per course, compared tick by tick


def shots(rng: np.random.Generator, count: int) -> np.ndarray:
    angle = rng.uniform(0, 2 * np.pi, count)
    # Some actions lie outside the unit disc, to exercise the norm clamp.
    power = rng.uniform(0, 1.3, count)
    return np.stack([power * np.cos(angle), power * np.sin(angle)], axis=1)


def main() -> None:
    rng = np.random.default_rng(765)
    cases = {}
    for name, course in golf.COURSES.items():
        start = golf.sample_free(course, RANDOM_SHOTS, rng)
        action = shots(rng, RANDOM_SHOTS)

        # Rest positions near the hole, and shots aimed at it from anywhere.
        ring = golf.sample_free(course, NEAR_HOLE_SHOTS, rng)
        offset = np.array(course.hole) - ring
        speed = np.sqrt(2 * golf.FRICTION * np.linalg.norm(offset, axis=1)) / golf.MAX_SPEED
        aimed = offset / np.linalg.norm(offset, axis=1, keepdims=True) * speed[:, None]
        aimed = aimed * rng.uniform(0.8, 1.3, (NEAR_HOLE_SHOTS, 1))
        start = np.concatenate([start, ring])
        action = np.concatenate([action, aimed])

        if course.sand:
            box = np.array(course.sand[0])
            inside = rng.uniform(box[:2] - 0.05, box[2:] + 0.05, (NEAR_SAND_SHOTS, 2))
            keep = golf.is_free(course, inside)
            start = np.concatenate([start, inside[keep]])
            action = np.concatenate([action, shots(rng, int(keep.sum()))])

        # Edge cases: no shot at all, and a ball already resting on the hole.
        start = np.concatenate([start, [course.start, course.hole]])
        action = np.concatenate([action, [[0.0, 0.0], [0.0, 0.0]]])

        final, sunk = golf.simulate(course, start, action)
        paths = []
        for index in rng.choice(len(start) - 2, PATHS, replace=False):
            _, _, path = golf.simulate(course, start[index:index + 1], action[index:index + 1], return_paths=True)
            paths.append({"index": int(index), "path": path[0].tolist()})
        cases[name] = {
            "start": course.start, "hole": course.hole, "walls": course.walls, "sand": course.sand,
            "solid": course.solid.tolist(),
            "starts": start.tolist(), "actions": action.tolist(),
            "finals": final.tolist(), "sunk": sunk.astype(int).tolist(), "paths": paths,
        }
    constants = {key: getattr(golf, key) for key in (
        "MAX_SPEED", "FRICTION", "SAND_FRICTION", "RESTITUTION", "DT", "MAX_TICKS", "HOLE_RADIUS",
        "CAPTURE_SPEED", "BALL_RADIUS", "WALL")}
    (HERE / "parity_cases.json").write_text(json.dumps({"constants": constants, "courses": cases}))
    print("wrote", sum(len(c["starts"]) for c in cases.values()), "shots")


if __name__ == "__main__":
    main()
