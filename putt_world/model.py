"""A small LeWorldModel: pixel encoder, latent predictor, and the SIGReg regularizer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from golf import BALL_RADIUS, COURSES, IMAGE_SIZE, Course, background, is_free, sample_free, simulate

RENDER_RADIUS = 0.03
BALL_COLOR = (0.97, 0.22, 0.16)


class Renderer:
    """Draw a soft ball on the course image, so sub-pixel position stays visible."""

    def __init__(self, course: Course, device: torch.device):
        image = background(course).transpose(2, 0, 1)
        self.background = torch.tensor(image, dtype=torch.float32, device=device)
        centers = (torch.arange(IMAGE_SIZE, device=device) + 0.5) / IMAGE_SIZE
        self.yy, self.xx = torch.meshgrid(centers, centers, indexing="ij")
        self.color = torch.tensor(BALL_COLOR, device=device).view(1, 3, 1, 1)

    def __call__(self, pos: torch.Tensor) -> torch.Tensor:
        dist = torch.sqrt((self.xx - pos[:, 0, None, None]) ** 2 + (self.yy - pos[:, 1, None, None]) ** 2)
        alpha = ((RENDER_RADIUS - dist) * IMAGE_SIZE + 0.5).clamp(0, 1).unsqueeze(1)
        return (1 - alpha) * self.background + alpha * self.color


class Encoder(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        layers: list[nn.Module] = []
        channels = (3, 32, 64, 128, 128)
        for inputs, outputs in zip(channels, channels[1:]):
            layers += [nn.Conv2d(inputs, outputs, 3, stride=2, padding=1), nn.GELU()]
        self.conv = nn.Sequential(*layers)
        # BatchNorm matters: the ball is a few pixels, and at a constant latent the
        # SIGReg gradient vanishes, so without it the encoder collapses.
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(128 * 4 * 4, 512), nn.BatchNorm1d(512), nn.GELU(),
                                  nn.Linear(512, latent_dim), nn.BatchNorm1d(latent_dim))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.head(self.conv(image))


class Predictor(nn.Module):
    def __init__(self, latent_dim: int, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 2, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, latent_dim),
        )

    def forward(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([latent, action], dim=-1))


def sigreg(latent: torch.Tensor, projections: int = 1024, knots: int = 17, limit: float = 3.0) -> torch.Tensor:
    """Epps-Pulley test of random 1-D projections against a standard normal."""
    directions = torch.randn(latent.shape[1], projections, device=latent.device)
    directions = directions / directions.norm(dim=0, keepdim=True)
    projected = latent @ directions
    t = torch.linspace(0, limit, knots, device=latent.device)
    target = torch.exp(-0.5 * t ** 2)
    angles = projected.unsqueeze(-1) * t
    error = (angles.cos().mean(0) - target) ** 2 + angles.sin().mean(0) ** 2
    weighted = error * target
    step = t[1] - t[0]
    integral = step * (weighted.sum(-1) - 0.5 * (weighted[..., 0] + weighted[..., -1]))
    return latent.shape[0] * integral.mean()


def random_shots(count: int, rng: np.random.Generator) -> np.ndarray:
    angle = rng.uniform(0, 2 * np.pi, count)
    power = np.sqrt(rng.uniform(0, 1, count))
    return np.stack([power * np.cos(angle), power * np.sin(angle)], axis=1)


def make_dataset(course: Course, count: int, seed: int, power_scale: float = 1.0) -> dict[str, np.ndarray]:
    """Reward-free random shots from random resting positions.

    Stored actions are normalized; `power_scale` shrinks what they do in the world.
    """
    rng = np.random.default_rng(seed)
    start = sample_free(course, count, rng)
    action = random_shots(count, rng)
    final, _ = simulate(course, start, power_scale * action)
    return {"start": start.astype(np.float32), "action": action.astype(np.float32),
            "final": final.astype(np.float32)}


def pick_device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def train(course_name: str, *, seed: int = 0, latent_dim: int = 32, weight: float = 0.09, steps: int = 4000,
          transitions: int = 60000, batch: int = 256, lr: float = 1e-3, device: torch.device | None = None,
          log_every: int = 0, power_scale: float = 1.0) -> dict:
    device = device or pick_device()
    torch.manual_seed(seed)
    course = COURSES[course_name]
    data = make_dataset(course, transitions, seed, power_scale)
    start = torch.tensor(data["start"], device=device)
    action = torch.tensor(data["action"], device=device)
    final = torch.tensor(data["final"], device=device)
    render = Renderer(course, device)
    encoder = Encoder(latent_dim).to(device)
    predictor = Predictor(latent_dim).to(device)
    optimizer = torch.optim.AdamW([*encoder.parameters(), *predictor.parameters()], lr=lr, weight_decay=1e-5)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
    history = []
    for step in range(1, steps + 1):
        index = torch.randint(0, transitions, (batch,), device=device)
        now, after = encoder(render(torch.cat([start[index], final[index]]))).chunk(2)
        prediction = predictor(now, action[index])
        predict_loss = (prediction - after).pow(2).sum(-1).mean()
        regularizer = sigreg(torch.cat([now, after]))
        loss = predict_loss + weight * regularizer
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        schedule.step()
        if log_every and step % log_every == 0:
            history.append((step, predict_loss.item(), regularizer.item()))
            print(f"{course_name} step {step} predict {predict_loss.item():.4f} sigreg {regularizer.item():.3f}", flush=True)
    return {"course": course_name, "seed": seed, "latent_dim": latent_dim, "weight": weight,
            "power_scale": power_scale, "transitions": transitions,
            "encoder": {k: v.cpu() for k, v in encoder.state_dict().items()},
            "predictor": {k: v.cpu() for k, v in predictor.state_dict().items()}, "history": history}


class WorldModel:
    """A trained model for inference on the CPU."""

    def __init__(self, state: dict):
        self.course = COURSES[state["course"]]
        self.power_scale = state.get("power_scale", 1.0)
        self.device = torch.device("cpu")
        self.encoder = Encoder(state["latent_dim"])
        self.encoder.load_state_dict(state["encoder"])
        self.predictor = Predictor(state["latent_dim"])
        self.predictor.load_state_dict(state["predictor"])
        self.encoder.eval()
        self.predictor.eval()
        self.render = Renderer(self.course, self.device)

    @torch.no_grad()
    def encode(self, pos: np.ndarray, chunk: int = 2048) -> torch.Tensor:
        tensor = torch.tensor(np.asarray(pos), dtype=torch.float32)
        return torch.cat([self.encoder(self.render(part)) for part in tensor.split(chunk)])

    @torch.no_grad()
    def rollout(self, latent: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Imagine shots. `actions` is (samples, horizon, 2); returns (samples, horizon, dim)."""
        states = []
        current = latent.expand(actions.shape[0], -1)
        for step in range(actions.shape[1]):
            current = self.predictor(current, actions[:, step])
            states.append(current)
        return torch.stack(states, dim=1)


def save(state: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load(path: Path) -> WorldModel:
    return WorldModel(torch.load(path, weights_only=False))
