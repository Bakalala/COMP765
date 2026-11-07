import numpy as np
import pytest

from golf import (BALL_RADIUS, COURSES, FRICTION, MAX_SPEED, geodesic_map, in_rects, is_free,
                  lookup, sample_free, simulate)


def test_straight_shot_stops_at_friction_distance():
    course = COURSES["gate"]
    pos, sunk = simulate(course, np.array([[0.1, 0.1]]), np.array([[0.4, 0.0]]))
    expected = (0.4 * MAX_SPEED) ** 2 / (2 * FRICTION)
    assert pos[0, 0] - 0.1 == pytest.approx(expected, abs=0.01)
    assert pos[0, 1] == pytest.approx(0.1)
    assert not sunk[0]


def test_zero_shot_does_not_move():
    pos, _ = simulate(COURSES["open"], np.array([[0.3, 0.3]]), np.zeros((1, 2)))
    assert pos[0] == pytest.approx([0.3, 0.3])


def test_ball_never_enters_solid_walls():
    rng = np.random.default_rng(0)
    for course in COURSES.values():
        start = sample_free(course, 3000, rng)
        action = rng.uniform(-1, 1, (3000, 2))
        pos, sunk = simulate(course, start, action)
        assert is_free(course, pos[~sunk]).mean() > 0.999
        assert not in_rects(pos, course.solid).any()


def test_wall_blocks_direct_path():
    course = COURSES["gate"]
    pos, sunk = simulate(course, np.array([course.start]), np.array([[1.0, 0.0]]))
    assert pos[0, 0] < 0.5
    assert not sunk[0]


def test_gentle_shot_sinks():
    course = COURSES["open"]
    start = np.array([[course.hole[0] - 0.2, course.hole[1]]])
    _, sunk = simulate(course, start, np.array([[0.55, 0.0]]))
    assert sunk[0]


def test_geodesic_exceeds_straight_line_behind_wall():
    course = COURSES["gate"]
    grid = geodesic_map(course)
    start = np.array([course.start])
    straight = np.linalg.norm(start[0] - np.array(course.hole))
    assert lookup(grid, start)[0] > straight * 1.2


def test_every_course_is_solvable_from_start():
    for course in COURSES.values():
        assert np.isfinite(lookup(geodesic_map(course), np.array([course.start]))[0])


def test_simulation_is_deterministic():
    course = COURSES["slalom"]
    start, action = np.array([[0.2, 0.2]]), np.array([[0.7, 0.4]])
    assert np.array_equal(simulate(course, start, action)[0], simulate(course, start, action)[0])


def test_points_outside_the_arena_are_not_free():
    assert not is_free(COURSES["open"], np.array([[1.03, 0.5], [-0.02, 0.5], [0.5, 1.2]])).any()
