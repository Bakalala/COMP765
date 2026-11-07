/* Painting: the static course layers, the Mind view heatmap, and the ball. */
(function (root) {
  "use strict";
  const G = root.PuttGolf;

  function canvasOf(width, height = width) {
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    return canvas;
  }

  /* Seeded, so the course looks the same on every resize. */
  function randomizer(seed) {
    let state = seed >>> 0;
    return () => {
      state = (state + 0x6d2b79f5) >>> 0;
      let t = Math.imul(state ^ (state >>> 15), 1 | state);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function noiseField(seed, size = 256) {
    const random = randomizer(seed);
    const field = new Float32Array(size * size);
    for (let i = 0; i < field.length; i++) field[i] = random();
    return (x, y) => {
      const x0 = Math.floor(x), y0 = Math.floor(y);
      let fx = x - x0, fy = y - y0;
      fx = fx * fx * (3 - 2 * fx);
      fy = fy * fy * (3 - 2 * fy);
      const m = size - 1;
      const a = field[(y0 & m) * size + (x0 & m)], b = field[(y0 & m) * size + ((x0 + 1) & m)];
      const c = field[((y0 + 1) & m) * size + (x0 & m)], d = field[((y0 + 1) & m) * size + ((x0 + 1) & m)];
      return a + (b - a) * fx + (c - a) * fy + (a - b - c + d) * fx * fy;
    };
  }

  /* ---------- grass ---------- */

  const grassCache = new Map();
  function grass(S) {
    if (grassCache.has(S)) return grassCache.get(S);
    const canvas = canvasOf(S);
    const ctx = canvas.getContext("2d");
    const image = ctx.createImageData(S, S);
    const data = image.data;
    const patch = noiseField(11), blades = noiseField(12), speck = randomizer(13);
    const stripes = 7.5;
    for (let y = 0; y < S; y++) {
      const v = y / S;
      for (let x = 0; x < S; x++) {
        const u = x / S;
        const along = (u + v) * 0.7071, across = (u - v) * 0.7071;
        // Diagonal mown bands with slightly soft edges.
        const band = Math.tanh(Math.sin(Math.PI * across * stripes + 0.6) * 7);
        const clump = patch(u * 5, v * 5) - 0.5;
        const fine = blades(u * 620, v * 620) - 0.5;
        const light = 1 + 0.062 * band + 0.05 * clump + 0.07 * fine + 0.05 * (speck() - 0.5);
        const edge = Math.max(Math.abs(u - 0.5), Math.abs(v - 0.5)) * 2;
        const shade = light * (1 - 0.1 * edge * edge * edge);
        const i = (y * S + x) * 4;
        data[i] = 70 * shade * (1 + 0.5 * clump);
        data[i + 1] = 143 * shade;
        data[i + 2] = 72 * shade;
        data[i + 3] = 255;
      }
    }
    ctx.putImageData(image, 0, 0);
    grassCache.clear();
    grassCache.set(S, canvas);
    return canvas;
  }

  /* ---------- sand ---------- */

  function roundedRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.roundRect(x, y, w, h, r);
  }

  function paintSand(ctx, rect, S) {
    const [x0, y0, x1, y1] = rect.map((v) => v * S);
    const w = Math.ceil(x1 - x0), h = Math.ceil(y1 - y0), r = 0.014 * S;
    const texture = canvasOf(w, h);
    const tctx = texture.getContext("2d");
    const image = tctx.createImageData(w, h);
    const soft = noiseField(21), speck = randomizer(22);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const u = x / S, v = y / S;
        // Shallow rake lines curving across the bunker.
        const rake = Math.sin((v + 0.05 * Math.sin(u * 16)) * 210) * 0.022;
        const light = 1 + rake + 0.08 * (soft(u * 14, v * 14) - 0.5) + 0.12 * (speck() - 0.5);
        const i = (y * w + x) * 4;
        image.data[i] = 233 * light;
        image.data[i + 1] = 208 * light;
        image.data[i + 2] = 152 * light;
        image.data[i + 3] = 255;
      }
    }
    tctx.putImageData(image, 0, 0);

    ctx.save();
    // A pale lip around the bunker.
    roundedRect(ctx, x0 - 0.004 * S, y0 - 0.004 * S, w + 0.008 * S, h + 0.008 * S, r * 1.2);
    ctx.fillStyle = "rgb(205 214 150 / 0.55)";
    ctx.fill();
    roundedRect(ctx, x0, y0, w, h, r);
    ctx.clip();
    ctx.drawImage(texture, x0, y0);
    // Sunken edge: shade thrown in from the top-left rim.
    ctx.shadowColor = "rgb(95 66 22 / 0.55)";
    ctx.shadowBlur = 0.018 * S;
    ctx.shadowOffsetX = 0.005 * S;
    ctx.shadowOffsetY = 0.006 * S;
    ctx.lineWidth = 0.012 * S;
    ctx.strokeStyle = "rgb(0 0 0 / 1)";
    roundedRect(ctx, x0 - 0.006 * S, y0 - 0.006 * S, w + 0.012 * S, h + 0.012 * S, r);
    ctx.stroke();
    ctx.restore();
  }

  /* ---------- hole and tee ---------- */

  function paintTee(ctx, course, S) {
    const [x, y] = course.start.map((v) => v * S);
    const half = 0.034 * S;
    ctx.save();
    roundedRect(ctx, x - half, y - half, 2 * half, 2 * half, 0.012 * S);
    ctx.fillStyle = "rgb(24 62 40 / 0.5)";
    ctx.fill();
    ctx.lineWidth = Math.max(1, 0.0022 * S);
    ctx.strokeStyle = "rgb(226 240 214 / 0.32)";
    ctx.stroke();
    ctx.restore();
  }

  function paintCollar(ctx, course, S) {
    const [x, y] = course.hole.map((v) => v * S);
    const r = G.HOLE_RADIUS * S;
    // Close-cut grass around the cup.
    const collar = ctx.createRadialGradient(x, y, r, x, y, r * 2.5);
    collar.addColorStop(0, "rgb(176 222 148 / 0.42)");
    collar.addColorStop(1, "rgb(176 222 148 / 0)");
    ctx.fillStyle = collar;
    ctx.beginPath();
    ctx.arc(x, y, r * 2.5, 0, Math.PI * 2);
    ctx.fill();
  }

  function paintHole(ctx, course, S) {
    const [x, y] = course.hole.map((v) => v * S);
    const r = G.HOLE_RADIUS * S;
    ctx.save();
    // The cup: shadow falls from the top-left rim.
    const cup = ctx.createRadialGradient(x + r * 0.3, y + r * 0.35, r * 0.1, x, y, r);
    cup.addColorStop(0, "#2a3a30");
    cup.addColorStop(0.65, "#101a15");
    cup.addColorStop(1, "#050a07");
    ctx.fillStyle = cup;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fill();
    ctx.lineWidth = r * 0.11;
    ctx.strokeStyle = "rgb(240 248 226 / 0.75)";
    ctx.beginPath();
    ctx.arc(x, y, r * 1.02, -0.1 * Math.PI, 0.72 * Math.PI);
    ctx.stroke();
    ctx.strokeStyle = "rgb(10 28 16 / 0.5)";
    ctx.beginPath();
    ctx.arc(x, y, r * 1.02, 0.75 * Math.PI, 1.88 * Math.PI);
    ctx.stroke();
    ctx.restore();
  }

  /* ---------- walls ---------- */

  function paintWood(ctx, rect, S, random) {
    const [x0, y0, x1, y1] = rect.map((v) => v * S);
    const w = x1 - x0, h = y1 - y0, horizontal = w >= h;
    const length = horizontal ? w : h, thick = horizontal ? h : w;
    ctx.save();
    ctx.beginPath();
    ctx.rect(x0, y0, w, h);
    ctx.clip();
    const tone = ctx.createLinearGradient(x0, y0, horizontal ? x0 : x1, horizontal ? y1 : y0);
    tone.addColorStop(0, "#d2a671");
    tone.addColorStop(0.5, "#c4945e");
    tone.addColorStop(1, "#b48250");
    ctx.fillStyle = tone;
    ctx.fillRect(x0, y0, w, h);
    ctx.lineCap = "round";
    const streaks = Math.round(length / S * 60 + 14);
    for (let i = 0; i < streaks; i++) {
      const across = random() * thick, start = random() * length, run = (0.05 + 0.25 * random()) * S;
      const dark = random() < 0.7;
      ctx.strokeStyle = dark ? `rgb(98 60 28 / ${0.1 + 0.2 * random()})` : `rgb(255 232 196 / ${0.08 + 0.14 * random()})`;
      ctx.lineWidth = Math.max(0.6, 0.0012 * S * (1 + random()));
      ctx.beginPath();
      if (horizontal) {
        ctx.moveTo(x0 + start, y0 + across);
        ctx.lineTo(x0 + start + run, y0 + across + (random() - 0.5) * 2);
      } else {
        ctx.moveTo(x0 + across, y0 + start);
        ctx.lineTo(x0 + across + (random() - 0.5) * 2, y0 + start + run);
      }
      ctx.stroke();
    }
    // Plank joints.
    ctx.strokeStyle = "rgb(84 50 24 / 0.5)";
    ctx.lineWidth = Math.max(1, 0.0016 * S);
    const plank = 0.24 * S;
    for (let at = plank * (0.4 + 0.6 * random()); at < length; at += plank) {
      ctx.beginPath();
      if (horizontal) { ctx.moveTo(x0 + at, y0); ctx.lineTo(x0 + at, y1); } else { ctx.moveTo(x0, y0 + at); ctx.lineTo(x1, y0 + at); }
      ctx.stroke();
    }
    ctx.restore();
  }

  function paintWalls(ctx, course, S) {
    const mask = canvasOf(S), mctx = mask.getContext("2d");
    mctx.fillStyle = "#000";
    for (const [x0, y0, x1, y1] of course.solid) mctx.fillRect(x0 * S, y0 * S, (x1 - x0) * S, (y1 - y0) * S);

    // Soft cast shadow, then a tight contact shadow, both falling down and right.
    ctx.save();
    ctx.shadowColor = "rgb(6 26 14 / 0.5)";
    ctx.shadowBlur = 0.03 * S;
    ctx.shadowOffsetX = 0.012 * S;
    ctx.shadowOffsetY = 0.016 * S;
    ctx.drawImage(mask, 0, 0);
    ctx.shadowColor = "rgb(6 26 14 / 0.55)";
    ctx.shadowBlur = 0.006 * S;
    ctx.shadowOffsetX = 0.003 * S;
    ctx.shadowOffsetY = 0.004 * S;
    ctx.drawImage(mask, 0, 0);
    ctx.restore();

    const wood = canvasOf(S), wctx = wood.getContext("2d");
    const random = randomizer(31);
    for (const rect of course.solid) paintWood(wctx, rect, S, random);

    // Bevel: the shadow of everything outside the walls falls just inside their edges.
    const outside = canvasOf(S), octx = outside.getContext("2d");
    octx.fillRect(0, 0, S, S);
    octx.globalCompositeOperation = "destination-out";
    octx.drawImage(mask, 0, 0);
    wctx.globalCompositeOperation = "source-atop";
    const bevel = 0.0048 * S;
    for (const [color, sign] of [["rgb(255 238 205 / 0.95)", 1], ["rgb(62 32 12 / 0.8)", -1]]) {
      wctx.shadowColor = color;
      wctx.shadowBlur = bevel * 0.9;
      wctx.shadowOffsetX = sign * bevel;
      wctx.shadowOffsetY = sign * bevel;
      wctx.drawImage(outside, 0, 0);
    }
    ctx.drawImage(wood, 0, 0);
  }

  /* The ground goes under the heatmap; the walls and hole go over it. */
  function buildLayers(course, S) {
    const ground = canvasOf(S), gctx = ground.getContext("2d");
    gctx.drawImage(grass(S), 0, 0);
    for (const rect of course.sand) paintSand(gctx, rect, S);
    paintTee(gctx, course, S);
    paintCollar(gctx, course, S);
    const top = canvasOf(S), tctx = top.getContext("2d");
    paintHole(tctx, course, S);
    paintWalls(tctx, course, S);
    return { ground, top };
  }

  function paintThumb(canvas, course) {
    const S = canvas.width;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#4d9a55";
    ctx.fillRect(0, 0, S, S);
    ctx.fillStyle = "#e9d098";
    for (const [x0, y0, x1, y1] of course.sand) ctx.fillRect(x0 * S, y0 * S, (x1 - x0) * S, (y1 - y0) * S);
    ctx.fillStyle = "#a9794a";
    for (const [x0, y0, x1, y1] of course.solid) ctx.fillRect(x0 * S, y0 * S, (x1 - x0) * S, (y1 - y0) * S);
    ctx.fillStyle = "#10201a";
    ctx.beginPath();
    ctx.arc(course.hole[0] * S, course.hole[1] * S, 0.055 * S, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "#fff";
    ctx.beginPath();
    ctx.arc(course.start[0] * S, course.start[1] * S, 0.04 * S, 0, Math.PI * 2);
    ctx.fill();
  }

  /* ---------- Mind view heatmap ---------- */

  const HEAT = [[243, 239, 255], [188, 174, 242], [106, 82, 207], [36, 22, 115]];
  const BANDS = 10;

  function heatColor(t) {
    // A soft staircase: bands read like contour lines without jagged edges.
    const scaled = Math.min(t, 0.9999) * BANDS, band = Math.floor(scaled), within = scaled - band;
    const rise = Math.min(Math.max((within - 0.8) / 0.2, 0), 1);
    const stepped = (band + 0.5 + rise * rise * (3 - 2 * rise)) / BANDS;
    const v = Math.min(0.6 * stepped + 0.4 * t, 1);
    const at = v * (HEAT.length - 1), i = Math.min(Math.floor(at), HEAT.length - 2), f = at - i;
    return HEAT[i].map((c, k) => c + (HEAT[i + 1][k] - c) * f);
  }

  /* Paint a cost grid (NaN where the ball cannot rest). Blocked cells next to free
   * ones borrow from them, so colour reaches each wall without crossing it. */
  function paintHeat(costs, grid) {
    const values = Array.from(costs).filter(Number.isFinite).sort((a, b) => a - b);
    const low = values[0];
    const high = Math.max(values[Math.floor(values.length * 0.98)], low + 1e-6);

    let filled = Float64Array.from(costs);
    for (let pass = 0; pass < 2; pass++) {
      const next = Float64Array.from(filled);
      for (let y = 0; y < grid; y++) {
        for (let x = 0; x < grid; x++) {
          if (Number.isFinite(filled[y * grid + x])) continue;
          let sum = 0, count = 0;
          for (let dy = -1; dy <= 1; dy++) {
            for (let dx = -1; dx <= 1; dx++) {
              const nx = x + dx, ny = y + dy;
              if (nx < 0 || ny < 0 || nx >= grid || ny >= grid) continue;
              const value = filled[ny * grid + nx];
              if (Number.isFinite(value)) { sum += value; count++; }
            }
          }
          if (count) next[y * grid + x] = sum / count;
        }
      }
      filled = next;
    }

    const size = grid * 12;
    const canvas = canvasOf(size);
    const ctx = canvas.getContext("2d");
    const image = ctx.createImageData(size, size);
    for (let py = 0; py < size; py++) {
      const gy = ((py + 0.5) / size) * grid - 0.5, y0 = Math.floor(gy), fy = gy - y0;
      for (let px = 0; px < size; px++) {
        const gx = ((px + 0.5) / size) * grid - 0.5, x0 = Math.floor(gx), fx = gx - x0;
        let sum = 0, weight = 0;
        for (let j = 0; j < 2; j++) {
          for (let i = 0; i < 2; i++) {
            const cx = Math.min(Math.max(x0 + i, 0), grid - 1), cy = Math.min(Math.max(y0 + j, 0), grid - 1);
            const value = filled[cy * grid + cx];
            if (!Number.isFinite(value)) continue;
            const w = (i ? fx : 1 - fx) * (j ? fy : 1 - fy);
            sum += w * value;
            weight += w;
          }
        }
        const o = (py * size + px) * 4;
        if (weight === 0) continue;
        const t = Math.min(Math.max((sum / weight - low) / (high - low), 0), 1);
        const color = heatColor(t);
        image.data[o] = color[0];
        image.data[o + 1] = color[1];
        image.data[o + 2] = color[2];
        image.data[o + 3] = 255;
      }
    }
    ctx.putImageData(image, 0, 0);
    return { canvas, low, high, filled };
  }

  /* ---------- ball ---------- */

  const dimples = (() => {
    const random = randomizer(41);
    return Array.from({ length: 34 }, () => {
      const z = random() * 2 - 1, angle = random() * Math.PI * 2, r = Math.sqrt(1 - z * z);
      return [r * Math.cos(angle), r * Math.sin(angle), z];
    });
  })();

  /* Spin the ball's dimples: roll by `distance` along direction (dx, dy). */
  function rollBall(spin, dx, dy, angle) {
    const c = Math.cos(angle), s = Math.sin(angle);
    for (const p of spin) {
      const along = p[0] * dx + p[1] * dy, z = p[2];
      const rolled = along * c + z * s;
      p[2] = z * c - along * s;
      p[0] += (rolled - along) * dx;
      p[1] += (rolled - along) * dy;
    }
  }

  function drawBall(ctx, x, y, r, spin, shadowLift = 1) {
    ctx.save();
    const sx = x + r * 0.55 * shadowLift, sy = y + r * 0.7 * shadowLift;
    const shadow = ctx.createRadialGradient(sx, sy, r * 0.2, sx, sy, r * 1.35);
    shadow.addColorStop(0, "rgb(4 20 10 / 0.5)");
    shadow.addColorStop(1, "rgb(4 20 10 / 0)");
    ctx.fillStyle = shadow;
    ctx.beginPath();
    ctx.arc(sx, sy, r * 1.35, 0, Math.PI * 2);
    ctx.fill();

    const body = ctx.createRadialGradient(x - r * 0.38, y - r * 0.42, r * 0.08, x, y, r * 1.02);
    body.addColorStop(0, "#ffffff");
    body.addColorStop(0.55, "#f3f0e8");
    body.addColorStop(1, "#b9b4a6");
    ctx.fillStyle = body;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = "rgb(112 106 94 / 0.34)";
    for (const [px, py, pz] of spin) {
      if (pz < 0.15) continue;
      ctx.beginPath();
      ctx.ellipse(x + px * r * 0.94, y + py * r * 0.94, r * 0.1, r * 0.1 * Math.max(pz, 0.45), Math.atan2(py, px), 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();
  }

  /* ---------- flag ---------- */

  function drawFlag(ctx, course, S, time, alpha) {
    const x = course.hole[0] * S, y = course.hole[1] * S;
    const height = 0.13 * S, top = y - height;
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.lineCap = "round";
    // The pole's shadow lies on the grass.
    const fall = ctx.createLinearGradient(x, y, x + 0.06 * S, y + 0.08 * S);
    fall.addColorStop(0, "rgb(4 20 10 / 0.34)");
    fall.addColorStop(1, "rgb(4 20 10 / 0)");
    ctx.strokeStyle = fall;
    ctx.lineWidth = 0.0055 * S;
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x + 0.06 * S, y + 0.08 * S);
    ctx.stroke();
    const pole = ctx.createLinearGradient(x - 3, 0, x + 3, 0);
    pole.addColorStop(0, "#fffdf4");
    pole.addColorStop(1, "#b5ac96");
    ctx.strokeStyle = pole;
    ctx.lineWidth = 0.0052 * S;
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x, top);
    ctx.stroke();

    const length = 0.072 * S, drop = 0.044 * S;
    ctx.beginPath();
    ctx.moveTo(x, top);
    const samples = 14;
    const wave = (k) => Math.sin(k * 5 - time * 4.5) * 0.0065 * S * k;
    for (let i = 1; i <= samples; i++) {
      const k = i / samples;
      ctx.lineTo(x + k * length, top + (k * drop) / 2 + wave(k));
    }
    for (let i = samples; i >= 0; i--) {
      const k = i / samples;
      ctx.lineTo(x + k * length, top + drop - (k * drop) / 2 + wave(k));
    }
    ctx.closePath();
    const cloth = ctx.createLinearGradient(x, top, x + length, top + drop);
    cloth.addColorStop(0, "#e9553b");
    cloth.addColorStop(1, "#b92f1f");
    ctx.fillStyle = cloth;
    ctx.fill();
    ctx.restore();
  }

  root.PuttRender = { buildLayers, paintThumb, paintHeat, drawBall, drawFlag, rollBall, dimples };
})(window);
