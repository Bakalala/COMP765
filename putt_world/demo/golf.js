/* Port of golf.py. Shared by the browser (window.PuttGolf) and node (require).
 * Every expression keeps the Python operation order, so results agree bit for bit
 * unless a libm difference appears; parity_test.mjs checks this against golf.py. */
(function (root) {
  "use strict";

  const MAX_SPEED = 1.6;
  const FRICTION = 1.5;
  const SAND_FRICTION = 5.0;
  const RESTITUTION = 0.8;
  const DT = 0.008;
  const MAX_TICKS = 260;
  const HOLE_RADIUS = 0.04;
  const CAPTURE_SPEED = 0.9;
  const BALL_RADIUS = 0.018;
  const WALL = 0.03;

  function course(name, start, hole, walls = [], sand = []) {
    const border = [
      [0.0, 0.0, 1.0, WALL], [0.0, 1.0 - WALL, 1.0, 1.0],
      [0.0, 0.0, WALL, 1.0], [1.0 - WALL, 0.0, 1.0, 1.0],
    ];
    return { name, start, hole, walls, sand, solid: border.concat(walls) };
  }

  const COURSES = {
    open: course("open", [0.15, 0.5], [0.85, 0.5], [], [[0.42, 0.62, 0.62, 0.86]]),
    gate: course("gate", [0.15, 0.5], [0.85, 0.5], [[0.485, 0.25, 0.515, 1.0]]),
    slalom: course("slalom", [0.12, 0.5], [0.88, 0.5],
      [[0.32, 0.27, 0.35, 1.0], [0.65, 0.0, 0.68, 0.73]]),
    pocket: course("pocket", [0.15, 0.5], [0.75, 0.5],
      [[0.56, 0.28, 0.59, 0.72], [0.56, 0.28, 0.9, 0.31], [0.56, 0.69, 0.9, 0.72]]),
  };

  function inRects(x, y, rects, margin = 0.0) {
    for (let i = 0; i < rects.length; i++) {
      const r = rects[i];
      if (x > r[0] - margin && x < r[2] + margin && y > r[1] - margin && y < r[3] + margin) return true;
    }
    return false;
  }

  function isFree(c, x, y) {
    return !inRects(x, y, c.solid, BALL_RADIUS);
  }

  /* One shot. Returns {x, y, sunk} and, with wantPath, the per-tick path plus the
   * ticks where a wall was hit. An action is a velocity in units of MAX_SPEED. */
  function simulate(c, x0, y0, ax, ay, wantPath = false) {
    let x = x0, y = y0;
    const norm = Math.sqrt(ax * ax + ay * ay);
    const scale = Math.max(norm, 1.0);
    let vx = (MAX_SPEED * ax) / scale;
    let vy = (MAX_SPEED * ay) / scale;
    const solid = c.solid, sand = c.sand;
    const hx = c.hole[0], hy = c.hole[1];
    let sunk = false;
    let moving = Math.sqrt(vx * vx + vy * vy) > 1e-6;
    const xs = wantPath ? [x] : null, ys = wantPath ? [y] : null;
    const hits = wantPath ? [] : null;

    for (let tick = 0; tick < MAX_TICKS && moving; tick++) {
      // Move each axis separately so a wall reflects only the blocked component.
      const tx = x + vx * DT;
      let hit = false;
      if (inRects(tx, y, solid, BALL_RADIUS)) { vx *= -RESTITUTION; hit = true; } else x = tx;
      const ty = y + vy * DT;
      if (inRects(x, ty, solid, BALL_RADIUS)) { vy *= -RESTITUTION; hit = true; } else y = ty;
      const speed = Math.sqrt(vx * vx + vy * vy);
      const dx = x - hx, dy = y - hy;
      if (Math.sqrt(dx * dx + dy * dy) < HOLE_RADIUS && speed < CAPTURE_SPEED) {
        sunk = true;
        x = hx;
        y = hy;
        moving = false;
      }
      const friction = inRects(x, y, sand) ? SAND_FRICTION : FRICTION;
      const slowed = Math.max(speed - friction * DT, 0.0);
      const factor = slowed / Math.max(speed, 1e-9);
      vx *= factor;
      vy *= factor;
      if (!(slowed > 1e-3)) moving = false;
      if (wantPath) {
        xs.push(x);
        ys.push(y);
        if (hit) hits.push({ tick: tick + 1, speed });
      }
    }

    const dx = x - hx, dy = y - hy;
    if (Math.sqrt(dx * dx + dy * dy) < HOLE_RADIUS) {
      sunk = true;
      x = hx;
      y = hy;
    }
    if (wantPath) {
      return { x, y, sunk, xs, ys, hits };
    }
    return { x, y, sunk };
  }

  const api = {
    MAX_SPEED, FRICTION, SAND_FRICTION, RESTITUTION, DT, MAX_TICKS, HOLE_RADIUS,
    CAPTURE_SPEED, BALL_RADIUS, WALL, COURSES, inRects, isFree, simulate,
  };
  root.PuttGolf = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof window !== "undefined" ? window : globalThis);
