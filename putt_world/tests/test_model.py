import numpy as np
import torch

from golf import COURSES, simulate
from model import Encoder, Predictor, Renderer, make_dataset, random_shots, sigreg
from plan import LatentGraph, aim_planner, cem, project_disc, run_episode


def test_sigreg_prefers_gaussian_over_collapse():
    torch.manual_seed(0)
    gaussian = torch.randn(2048, 16)
    assert sigreg(gaussian) < sigreg(torch.zeros(2048, 16)) / 20
    assert sigreg(gaussian) < sigreg(gaussian * 4)


def test_renderer_moves_ball():
    render = Renderer(COURSES["open"], torch.device("cpu"))
    pos = torch.tensor([[0.3, 0.3], [0.7, 0.7]])
    image = render(pos)
    assert image.shape == (2, 3, 64, 64)
    assert (image[0] - image[1]).abs().sum() > 1


def test_shapes():
    latent = Encoder(8)(torch.zeros(4, 3, 64, 64))
    assert latent.shape == (4, 8)
    assert Predictor(8)(latent, torch.zeros(4, 2)).shape == (4, 8)


def test_random_shots_stay_in_disc():
    assert np.linalg.norm(random_shots(1000, np.random.default_rng(0)), axis=1).max() <= 1.0


def test_dataset_matches_simulator():
    course = COURSES["gate"]
    data = make_dataset(course, 50, 3)
    final, _ = simulate(course, data["start"].astype(np.float64), data["action"].astype(np.float64))
    assert np.abs(final - data["final"]).max() < 1e-2


def test_cem_finds_target():
    target = torch.tensor([0.3, -0.4])
    generator = torch.Generator().manual_seed(0)
    best = cem(lambda a: (a[:, 0] - target).pow(2).sum(-1), 1, generator=generator)
    assert (best[0] - target).norm() < 0.05


def test_project_disc():
    assert project_disc(torch.tensor([[3.0, 4.0]])).norm() == 1.0


def test_value_iteration_counts_shots():
    centers = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [9.0, 9.0]])
    edges = np.array([[0, 1], [1, 2]])
    graph = LatentGraph(centers, edges, torch.tensor([2.0, 0.0]))
    assert graph.value.tolist() == [2.0, 1.0, 0.0, 4.0]


def test_aim_sinks_on_open_course_but_not_through_wall():
    assert run_episode(COURSES["open"], aim_planner(COURSES["open"]))["sunk"]
    assert not run_episode(COURSES["gate"], aim_planner(COURSES["gate"]))["sunk"]
