import numpy as np
import torch

from evaluate import wall_gap
from golf import COURSES


class PositionModel:
    """A stand-in whose latent is the ball position, so it cannot know about walls."""

    def __init__(self, name):
        self.course = COURSES[name]

    def encode(self, pos):
        return torch.tensor(np.asarray(pos), dtype=torch.float32)


def test_no_interior_wall_gives_no_gap():
    assert np.isnan(wall_gap(PositionModel("open"), np.random.default_rng(0)))


def test_position_latent_reads_one_at_a_real_wall():
    assert abs(wall_gap(PositionModel("gate"), np.random.default_rng(0)) - 1.0) < 0.01


def test_position_latent_reads_one_at_a_ghost_wall():
    assert abs(wall_gap(PositionModel("gate_ghost"), np.random.default_rng(0)) - 1.0) < 0.01
