"""Small deterministic grid mini-golf environment for tabular Dyna-Q."""

from __future__ import annotations

from dataclasses import dataclass


COURSES: dict[str, tuple[str, ...]] = {
    "easy": (
        "###########",
        "#.........#",
        "#S........#",
        "#.........#",
        "#.........#",
        "#.........#",
        "#........H#",
        "###########",
    ),
    "wall_gate": (
        "###########",
        "#.........#",
        "#....#....#",
        "#....#....#",
        "#S...#....#",
        "#..B.#....#",
        "#....#....#",
        "#....#...H#",
        "###########",
    ),
    # Serpentine family for the route-length study in run_extended.py.
    "ladder_1": (
        "###########",
        "#.........#",
        "#....#....#",
        "#....#....#",
        "#S...#...H#",
        "#....#....#",
        "#....#....#",
        "###########",
    ),
    "ladder_2": (
        "#############",
        "#...#.......#",
        "#...#...#...#",
        "#...#...#...#",
        "#S..#...#..H#",
        "#...#...#...#",
        "#.......#...#",
        "#############",
    ),
    "ladder_3": (
        "#################",
        "#...#.......#...#",
        "#...#...#...#...#",
        "#...#...#...#...#",
        "#S..#...#...#..H#",
        "#...#...#...#...#",
        "#.......#.......#",
        "#################",
    ),
    "ladder_4": (
        "#####################",
        "#...#.......#.......#",
        "#...#...#...#...#...#",
        "#...#...#...#...#...#",
        "#S..#...#...#...#..H#",
        "#...#...#...#...#...#",
        "#.......#.......#...#",
        "#####################",
    ),
}

# Clockwise from east. Each action is a direction at low or high strength.
DIRECTIONS: tuple[tuple[int, int], ...] = (
    (1, 0), (1, 1), (0, 1), (-1, 1),
    (-1, 0), (-1, -1), (0, -1), (1, -1),
)
POWER_STEPS = (2, 4)
N_ACTIONS = len(DIRECTIONS) * len(POWER_STEPS)


@dataclass(frozen=True)
class Course:
    name: str
    rows: tuple[str, ...]


class MiniGolfEnv:
    """A discrete shot-to-shot game with a small rendered board.

    The ball moves one grid cell per simulation tick. Walls reflect the blocked
    component of its direction; bumpers reflect the full direction. The agent
    observes only the resting cell between shots, making each shot a tabular
    state-action transition suitable for Dyna-Q.
    """

    def __init__(self, course: str = "easy", max_strokes: int = 20):
        if course not in COURSES:
            raise ValueError(f"Unknown course {course!r}; choose from {tuple(COURSES)}")
        self.course = Course(course, COURSES[course])
        self.max_strokes = max_strokes
        self.height = len(self.course.rows)
        self.width = len(self.course.rows[0])
        if any(len(row) != self.width for row in self.course.rows):
            raise ValueError("Course rows must all have the same width")

        self.start_cell = self._find_cell("S")
        self.hole_cell = self._find_cell("H")
        valid_cells = [
            (x, y)
            for y, row in enumerate(self.course.rows)
            for x, tile in enumerate(row)
            if tile not in "#B"
        ]
        self.cell_to_state = {cell: i for i, cell in enumerate(valid_cells)}
        self.state_to_cell = tuple(valid_cells)
        self.n_states = len(valid_cells)
        self.n_actions = N_ACTIONS
        self.hole_state = self.cell_to_state[self.hole_cell]
        self.state = self.cell_to_state[self.start_cell]
        self.strokes = 0
        self.done = False
        self.truncated = False

    def _find_cell(self, marker: str) -> tuple[int, int]:
        matches = [
            (x, y)
            for y, row in enumerate(self.course.rows)
            for x, tile in enumerate(row)
            if tile == marker
        ]
        if len(matches) != 1:
            raise ValueError(f"Course must contain exactly one {marker!r}")
        return matches[0]

    def _tile(self, x: int, y: int) -> str:
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return "#"
        return self.course.rows[y][x]

    def reset(self, seed: int | None = None) -> int:
        """Reset the course. ``seed`` is accepted for a gym-like interface."""
        del seed  # This first environment version is deterministic.
        self.state = self.cell_to_state[self.start_cell]
        self.strokes = 0
        self.done = False
        self.truncated = False
        return self.state

    def decode_action(self, action_id: int) -> tuple[int, int, int]:
        if not 0 <= action_id < self.n_actions:
            raise ValueError(f"action_id must be in [0, {self.n_actions})")
        direction_id, power_id = divmod(action_id, len(POWER_STEPS))
        dx, dy = DIRECTIONS[direction_id]
        return dx, dy, POWER_STEPS[power_id]

    def encode_action(self, direction_id: int, power_id: int) -> int:
        if not 0 <= direction_id < len(DIRECTIONS):
            raise ValueError("direction_id must be in [0, 8)")
        if not 0 <= power_id < len(POWER_STEPS):
            raise ValueError("power_id must be 0 (low) or 1 (high)")
        return direction_id * len(POWER_STEPS) + power_id

    def step(self, action_id: int) -> tuple[int, float, bool, dict]:
        if self.done or self.truncated:
            raise RuntimeError("Episode is over; call reset() before another shot")
        dx, dy, ticks = self.decode_action(action_id)
        x, y = self.state_to_cell[self.state]
        path = [(x, y)]

        for _ in range(ticks):
            nx, ny = x + dx, y + dy
            tile = self._tile(nx, ny)
            if tile == "B":
                dx, dy = -dx, -dy
                nx, ny = x + dx, y + dy
                if self._tile(nx, ny) in "#B":
                    break
            elif tile == "#":
                x_blocked = self._tile(x + dx, y) == "#" if dx else False
                y_blocked = self._tile(x, y + dy) == "#" if dy else False
                if x_blocked and not y_blocked:
                    dx = -dx
                elif y_blocked and not x_blocked:
                    dy = -dy
                elif x_blocked and y_blocked:
                    dx, dy = -dx, -dy
                else:
                    dx, dy = -dx, -dy
                nx, ny = x + dx, y + dy
                if self._tile(nx, ny) in "#B":
                    break

            x, y = nx, ny
            path.append((x, y))
            if (x, y) == self.hole_cell:
                break

        self.state = self.cell_to_state[(x, y)]
        self.strokes += 1
        sank = (x, y) == self.hole_cell
        self.done = sank
        self.truncated = not sank and self.strokes >= self.max_strokes
        info = {
            "path": path,
            "sank": sank,
            "truncated": self.truncated,
            "strokes": self.strokes,
            "course": self.course.name,
        }
        return self.state, -1.0, self.done or self.truncated, info

    def tile_at(self, x: int, y: int) -> str:
        return self.course.rows[y][x]

    def describe_action(self, action_id: int) -> str:
        dx, dy, ticks = self.decode_action(action_id)
        direction_id = DIRECTIONS.index((dx, dy))
        names = ("E", "SE", "S", "SW", "W", "NW", "N", "NE")
        return f"{names[direction_id]} / {'high' if ticks == 4 else 'low'}"

