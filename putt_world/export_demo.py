"""Export trained world models for the browser demo.

Reads checkpoints/{course}_seed0.pt and writes demo/model_data.js, which defines
window.PUTT_MODEL, plus demo/predictor_cases.json for predictor_test.mjs. Missing
checkpoints are skipped, so this can run while training is still going.

Plain float text would be about 20 MB, so large arrays are packed: integers split
into byte planes, zlib-compressed, base85-encoded. demo/planner.js
unpacks them. The small first and last predictor layers stay float32; the two large
hidden layers are integers with one scale per output row. WEIGHT_BITS was chosen so
the unpacked predictor stays within 1e-4 of torch, which predictor_test.mjs checks.
"""

from __future__ import annotations

import argparse
import base64
import json
import zlib
from pathlib import Path

import numpy as np
import torch

from golf import COURSES, is_free
from model import load, make_dataset
from plan import build_graph

ROOT = Path(__file__).parent
GRID = 48
SEED = 0
WEIGHT_BITS = (None, 17, 16, None)  # per linear layer; None keeps float32
LATENT_SCALE = 0.001  # latents rounded to 3 decimals
TEST_CASES = 256


def pack_ints(values: np.ndarray) -> str:
    """Little-endian 24-bit integers as three byte planes, compressed."""
    raw = values.astype("<i4").reshape(-1).view(np.uint8).reshape(-1, 4)[:, :3]
    planes = np.ascontiguousarray(raw.T).tobytes()
    return base64.b85encode(zlib.compress(planes, 9)).decode()


def pack_float32(values: np.ndarray) -> str:
    return base64.b85encode(values.astype("<f4").tobytes()).decode()


def pack_weight(weight: np.ndarray, bits: int | None) -> dict:
    if bits is None:
        return {"rows": weight.shape[0], "cols": weight.shape[1], "f": pack_float32(weight)}
    limit = 2 ** (bits - 1) - 1
    scale = (np.abs(weight).max(axis=1, keepdims=True) / limit).astype(np.float32)
    quantized = np.rint(weight / np.maximum(scale, 1e-30)).astype(np.int32)
    return {"rows": weight.shape[0], "cols": weight.shape[1], "q": pack_ints(quantized),
            "scale": pack_float32(scale[:, 0])}


def predictor_layers(model, bits: tuple) -> list[dict]:
    layers = [layer for layer in model.predictor.net if isinstance(layer, torch.nn.Linear)]
    assert len(layers) == 4
    return [{"w": pack_weight(layer.weight.detach().numpy(), width),
             "b": pack_float32(layer.bias.detach().numpy())} for layer, width in zip(layers, bits)]


def grid_points() -> np.ndarray:
    centers = (np.arange(GRID) + 0.5) / GRID
    xx, yy = np.meshgrid(centers, centers)
    return np.stack([xx.ravel(), yy.ravel()], axis=1)


def export_course(name: str, path: Path, bits: tuple) -> tuple[dict, dict]:
    model = load(path)
    course = COURSES[name]
    hole = np.array(course.hole)
    points = grid_points()
    latents = model.encode(points)
    goal = model.encode(hole[None])[0]
    graph = build_graph(model, make_dataset(course, 60000, SEED), hole, source="replay", seed=SEED)
    with torch.no_grad():
        shot_cost = graph.cost(latents).numpy()

    packed = np.rint(latents.numpy() / LATENT_SCALE)
    assert np.abs(packed).max() < 2 ** 15
    entry = {
        "grid": GRID,
        "dim": int(latents.shape[1]),
        "latents": {"q": pack_ints(packed), "scale": LATENT_SCALE},
        "free": is_free(course, points).astype(int).tolist(),
        "goal": np.round(goal.numpy().astype(np.float64), 3).tolist(),
        "shot_cost": np.round(shot_cost.astype(np.float64), 3).tolist(),
        "predictor": predictor_layers(model, bits),
    }

    # Reference outputs from the full-precision torch predictor.
    rng = torch.Generator().manual_seed(765)
    latent = torch.randn(TEST_CASES, latents.shape[1], generator=rng)
    action = torch.rand(TEST_CASES, 2, generator=rng) * 2 - 1
    with torch.no_grad():
        output = model.predictor(latent, action)
    cases = {"latent": latent.tolist(), "action": action.tolist(), "output": output.tolist()}
    return entry, cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, default=ROOT / "checkpoints")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    courses, cases = {}, {}
    for name in COURSES:
        path = args.checkpoints / f"{name}_seed{SEED}.pt"
        if not path.exists():
            print(f"skip {name}: {path} not found")
            continue
        courses[name], cases[name] = export_course(name, path, WEIGHT_BITS)
        print(f"exported {name}", flush=True)

    demo = ROOT / "demo"
    text = "window.PUTT_MODEL = " + json.dumps({"courses": courses}, separators=(",", ":")) + ";\n"
    (demo / "model_data.js").write_text(text)
    (demo / "predictor_cases.json").write_text(json.dumps(cases))
    print(f"wrote model_data.js: {len(text) / 1e6:.2f} MB, {len(courses)} of {len(COURSES)} courses")


if __name__ == "__main__":
    main()
