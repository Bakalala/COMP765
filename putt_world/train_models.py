"""Train the main world models. Checkpoints are written to checkpoints/."""

from __future__ import annotations

import time
from pathlib import Path

from golf import MAIN_COURSES
from model import save, train

CHECKPOINTS = Path(__file__).parent / "checkpoints"
SEEDS = (0, 1, 2)


def path_for(course: str, seed: int) -> Path:
    return CHECKPOINTS / f"{course}_seed{seed}.pt"


def main() -> None:
    # Seeds run in the outer loop so every course has a model as early as possible.
    for seed in SEEDS:
        for course in MAIN_COURSES:
            path = path_for(course, seed)
            if path.exists():
                continue
            began = time.time()
            state = train(course, seed=seed, log_every=1000)
            save(state, path)
            print(f"saved {path.name} in {time.time() - began:.0f}s", flush=True)


if __name__ == "__main__":
    main()
