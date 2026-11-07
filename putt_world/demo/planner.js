/* The agent: unpack the exported model, run the predictor, and plan with CEM.
 * Mirrors plan.py and model.py. Shared by the browser (window.PuttPlanner) and node. */
(function (root) {
  "use strict";

  // plan.cem defaults.
  const SAMPLES = 300, ELITES = 30, ITERATIONS = 6, INIT_STD = 0.7, MIN_STD = 0.02;
  const HORIZON = 1;
  const TIE_WEIGHT = 0.02; // plan.graph_planner

  /* ---------- unpacking ---------- */

  const B85 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!#$%&()*+-;<=>?@^_`{|}~";
  const B85_VALUE = new Uint8Array(128);
  for (let i = 0; i < B85.length; i++) B85_VALUE[B85.charCodeAt(i)] = i;

  /* Python's base64.b85decode. */
  function fromBase85(text) {
    const padded = (5 - (text.length % 5)) % 5;
    const bytes = new Uint8Array(((text.length + padded) / 5) * 4);
    for (let i = 0, o = 0; i < text.length; i += 5, o += 4) {
      let value = 0;
      for (let k = 0; k < 5; k++) {
        value = value * 85 + (i + k < text.length ? B85_VALUE[text.charCodeAt(i + k)] : 84);
      }
      bytes[o] = value / 16777216;
      bytes[o + 1] = (value / 65536) & 255;
      bytes[o + 2] = (value / 256) & 255;
      bytes[o + 3] = value & 255;
    }
    return bytes.subarray(0, bytes.length - padded);
  }

  function float32s(text) {
    const bytes = fromBase85(text);
    return new Float32Array(bytes.buffer, 0, bytes.length / 4);
  }

  async function inflate(bytes) {
    const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("deflate"));
    return new Uint8Array(await new Response(stream).arrayBuffer());
  }

  /* Signed 24-bit integers stored as three byte planes. */
  async function ints(text) {
    const planes = await inflate(fromBase85(text));
    const count = planes.length / 3;
    const out = new Int32Array(count);
    for (let i = 0; i < count; i++) {
      const value = planes[i] | (planes[count + i] << 8) | (planes[2 * count + i] << 16);
      out[i] = (value << 8) >> 8;
    }
    return out;
  }

  /* Build a ready-to-run model from one window.PUTT_MODEL.courses entry. */
  async function unpack(entry) {
    const { grid, dim } = entry;
    const packed = await ints(entry.latents.q);
    const latents = new Float32Array(packed.length);
    for (let i = 0; i < packed.length; i++) latents[i] = packed[i] * entry.latents.scale;

    const layers = [];
    for (const { w, b } of entry.predictor) {
      let weight;
      if (w.f) {
        weight = float32s(w.f);
      } else {
        const q = await ints(w.q);
        const scale = float32s(w.scale);
        weight = new Float32Array(q.length);
        for (let r = 0; r < w.rows; r++) {
          for (let c = 0; c < w.cols; c++) weight[r * w.cols + c] = q[r * w.cols + c] * scale[r];
        }
      }
      layers.push({ rows: w.rows, cols: w.cols, weight, bias: float32s(b) });
    }

    const freeCells = [];
    entry.free.forEach((value, cell) => { if (value) freeCells.push(cell); });
    const freeLatents = new Float32Array(freeCells.length * dim);
    freeCells.forEach((cell, i) => {
      freeLatents.set(latents.subarray(cell * dim, (cell + 1) * dim), i * dim);
    });
    const width = layers.reduce((most, layer) => Math.max(most, layer.rows, layer.cols), 0);
    return {
      grid, dim, latents, layers, free: entry.free, freeCells, freeLatents,
      goal: Float64Array.from(entry.goal),
      shotCost: entry.shot_cost ? Float64Array.from(entry.shot_cost) : null,
      scratch: [new Float64Array(width), new Float64Array(width)],
    };
  }

  /* ---------- predictor ---------- */

  /* erf to double precision: a series below 2.5, a continued fraction above. */
  function erf(x) {
    const a = Math.abs(x);
    if (a > 6) return Math.sign(x);
    let value;
    if (a < 2.5) {
      let term = a, sum = a;
      for (let n = 1; n < 80; n++) {
        term *= (2 * a * a) / (2 * n + 1);
        sum += term;
        if (term < sum * 1e-17) break;
      }
      value = (2 / Math.sqrt(Math.PI)) * Math.exp(-a * a) * sum;
    } else {
      let fraction = 0;
      for (let k = 60; k >= 1; k--) fraction = (k / 2) / (a + fraction);
      value = 1 - Math.exp(-a * a) / ((a + fraction) * Math.sqrt(Math.PI));
    }
    return x < 0 ? -value : value;
  }

  function gelu(x) {
    return 0.5 * x * (1 + erf(x * Math.SQRT1_2));
  }

  /* next = predictor(cat([latent, action])); latent may equal out. */
  function predict(model, latent, ax, ay, out) {
    const { layers, dim, scratch } = model;
    let input = scratch[0], output = scratch[1];
    for (let i = 0; i < dim; i++) input[i] = latent[i];
    input[dim] = ax;
    input[dim + 1] = ay;
    for (let l = 0; l < layers.length; l++) {
      const { rows, cols, weight, bias } = layers[l];
      const last = l === layers.length - 1;
      for (let r = 0; r < rows; r++) {
        const base = r * cols;
        let s0 = 0, s1 = 0, s2 = 0, s3 = 0, c = 0;
        for (; c + 3 < cols; c += 4) {
          s0 += weight[base + c] * input[c];
          s1 += weight[base + c + 1] * input[c + 1];
          s2 += weight[base + c + 2] * input[c + 2];
          s3 += weight[base + c + 3] * input[c + 3];
        }
        for (; c < cols; c++) s0 += weight[base + c] * input[c];
        const value = s0 + s1 + s2 + s3 + bias[r];
        output[r] = last ? value : gelu(value);
      }
      const swap = input; input = output; output = swap;
    }
    for (let i = 0; i < dim; i++) out[i] = input[i];
    return out;
  }

  /* ---------- latent grid lookups ---------- */

  /* Index into freeCells of the nearest grid latent. */
  function nearestFree(model, latent) {
    const { freeLatents, dim } = model;
    const count = model.freeCells.length;
    let best = 0, bestDistance = Infinity;
    for (let i = 0; i < count; i++) {
      const base = i * dim;
      let distance = 0;
      for (let d = 0; d < dim && distance < bestDistance; d++) {
        const delta = freeLatents[base + d] - latent[d];
        distance += delta * delta;
      }
      if (distance < bestDistance) { bestDistance = distance; best = i; }
    }
    return best;
  }

  /* Decode a latent to a position: the centre of the nearest free grid cell. */
  function decode(model, latent) {
    const cell = model.freeCells[nearestFree(model, latent)];
    const { grid } = model;
    return [((cell % grid) + 0.5) / grid, (Math.floor(cell / grid) + 0.5) / grid];
  }

  /* Latent of a real ball position: bilinear over the free cells around it. The
   * encoder itself is not exported, so the grid stands in for it. */
  function latentAt(model, x, y) {
    const { grid, dim, latents, free } = model;
    const gx = x * grid - 0.5, gy = y * grid - 0.5;
    const x0 = Math.floor(gx), y0 = Math.floor(gy);
    const out = new Float64Array(dim);
    let total = 0;
    for (let j = 0; j < 2; j++) {
      for (let i = 0; i < 2; i++) {
        const cx = x0 + i, cy = y0 + j;
        if (cx < 0 || cy < 0 || cx >= grid || cy >= grid || !free[cy * grid + cx]) continue;
        const weight = (i ? gx - x0 : 1 - gx + x0) * (j ? gy - y0 : 1 - gy + y0);
        if (weight <= 0) continue;
        total += weight;
        const base = (cy * grid + cx) * dim;
        for (let d = 0; d < dim; d++) out[d] += weight * latents[base + d];
      }
    }
    if (total > 0) {
      for (let d = 0; d < dim; d++) out[d] /= total;
      return out;
    }
    // Hard against a wall all four cells can be blocked: use the closest free cell.
    let best = model.freeCells[0], bestDistance = Infinity;
    for (const cell of model.freeCells) {
      const distance = ((cell % grid) - gx) ** 2 + (Math.floor(cell / grid) - gy) ** 2;
      if (distance < bestDistance) { bestDistance = distance; best = cell; }
    }
    return Float64Array.from(latents.subarray(best * dim, (best + 1) * dim));
  }

  function squaredDistance(a, b, offset = 0) {
    let sum = 0;
    for (let d = 0; d < b.length; d++) {
      const delta = a[offset + d] - b[d];
      sum += delta * delta;
    }
    return sum;
  }

  /* Cost-to-goal of every grid cell for the Mind view. NaN where the ball cannot rest. */
  function costMap(model, mode) {
    const { grid, dim, latents, free, goal, shotCost } = model;
    const out = new Float64Array(grid * grid).fill(NaN);
    if (mode === "graph" && !shotCost) return null;
    for (let cell = 0; cell < out.length; cell++) {
      if (!free[cell]) continue;
      out[cell] = mode === "graph" ? shotCost[cell] : squaredDistance(latents, goal, cell * dim);
    }
    return out;
  }

  /* ---------- CEM ---------- */

  function gaussian() {
    let u = 0;
    while (u === 0) u = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * Math.random());
  }

  /* Costs of plan.latent_planner (terminal) and plan.graph_planner. */
  function sequenceCost(model, mode, start, actions, offset, horizon, buffer) {
    buffer.set(start);
    let best = Infinity;
    for (let step = 0; step < horizon; step++) {
      predict(model, buffer, actions[offset + 2 * step], actions[offset + 2 * step + 1], buffer);
      if (mode === "graph") {
        const steps = model.shotCost[model.freeCells[nearestFree(model, buffer)]];
        best = Math.min(best, steps + TIE_WEIGHT * squaredDistance(buffer, model.goal));
      }
    }
    return mode === "graph" ? best : squaredDistance(buffer, model.goal);
  }

  function projectDisc(values, offset) {
    const norm = Math.max(Math.hypot(values[offset], values[offset + 1]), 1);
    values[offset] /= norm;
    values[offset + 1] /= norm;
  }

  /* Imagine one action sequence and decode each imagined rest position. */
  function imagine(model, start, actions, offset, horizon) {
    const latent = Float64Array.from(start);
    const points = [];
    for (let step = 0; step < horizon; step++) {
      predict(model, latent, actions[offset + 2 * step], actions[offset + 2 * step + 1], latent);
      points.push(decode(model, latent));
    }
    return points;
  }

  /* plan.cem as a generator, so the page can keep drawing while it searches.
   * Yields {samples: n} as it works, {iteration, elites} after each round, and
   * returns the best mean sequence. */
  function* cem(model, mode, start, { horizon = HORIZON, shown = 8 } = {}) {
    const width = 2 * horizon;
    const mean = new Float64Array(width);
    const std = new Float64Array(width).fill(INIT_STD);
    const actions = new Float64Array(SAMPLES * width);
    const costs = new Float64Array(SAMPLES);
    const buffer = new Float64Array(model.dim);
    const order = Array.from({ length: SAMPLES }, (_, i) => i);

    for (let iteration = 1; iteration <= ITERATIONS; iteration++) {
      for (let s = 0; s < SAMPLES; s++) {
        const offset = s * width;
        for (let k = 0; k < width; k++) actions[offset + k] = mean[k] + std[k] * gaussian();
        for (let step = 0; step < horizon; step++) projectDisc(actions, offset + 2 * step);
        costs[s] = sequenceCost(model, mode, start, actions, offset, horizon, buffer);
        yield { samples: s + 1, iteration };
      }
      order.sort((a, b) => costs[a] - costs[b]);
      for (let k = 0; k < width; k++) {
        let sum = 0;
        for (let e = 0; e < ELITES; e++) sum += actions[order[e] * width + k];
        const average = sum / ELITES;
        let spread = 0;
        for (let e = 0; e < ELITES; e++) spread += (actions[order[e] * width + k] - average) ** 2;
        mean[k] = average;
        std[k] = Math.sqrt(spread / (ELITES - 1)) + MIN_STD; // torch.std is unbiased
      }
      const elites = [];
      for (let e = 0; e < shown; e++) {
        const pick = order[Math.floor((e * ELITES) / shown)];
        elites.push(imagine(model, start, actions, pick * width, horizon));
      }
      yield { iteration, elites, bestCost: costs[order[0]] };
    }
    const plan = Float64Array.from(mean);
    for (let step = 0; step < horizon; step++) projectDisc(plan, 2 * step);
    return { actions: plan, imagined: imagine(model, start, plan, 0, horizon) };
  }

  const api = {
    SAMPLES, ELITES, ITERATIONS, HORIZON, unpack, erf, gelu, predict, decode, latentAt, costMap, cem,
  };
  root.PuttPlanner = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof window !== "undefined" ? window : globalThis);
