/* The game: state, input, the agent loop, and the per-frame drawing. */
(function () {
  "use strict";
  const G = window.PuttGolf, P = window.PuttPlanner, R = window.PuttRender;
  const $ = (id) => document.getElementById(id);

  const PULL_MAX = 0.27;     // drag length, in course units, for full power
  const PULL_MIN = 0.02;     // shorter drags cancel
  const GRAB_RADIUS = 0.055; // how close to the ball a drag must begin
  const MAX_SHOTS = 10;      // plan.MAX_SHOTS: where the evaluation gives up
  const SLICE_MS = 12;       // planning time per frame, so drawing stays smooth
  const HEAT_ALPHA = 0.86;

  const MODE_INFO = {
    latent: { title: "Squared latent distance to the hole", unit: "", name: "straight-line latent distance" },
    graph: { title: "Estimated shots to the hole", unit: " shots", name: "the shot graph" },
  };

  const board = $("board"), canvas = $("course"), ctx = canvas.getContext("2d");
  const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;

  const game = {
    course: G.COURSES.open,
    phase: "idle",           // idle, aiming, rolling, planning, previewing, sinking, sunk
    x: 0, y: 0,
    spin: R.dimples.map((p) => p.slice()),
    shots: { you: 0, model: 0 },
    drag: null,              // {px, py, power, ax, ay}
    roll: null,              // {result, began, tick, by}
    ghosts: [],              // {points, iteration, born}
    plan: null,              // {points}: the imagined line
    landing: null,           // imagined resting point of the shot under way
    trail: [],
    particles: [],
    ripples: [],
    sinkBegan: 0,
    token: 0,                // bumps to cancel a plan in flight
    taught: false,
  };
  const view = { S: 0, layers: null, mode: "latent", mind: false, heat: null, fading: null, fadeBegan: 0 };
  const models = {};         // course name -> {state, model?, heat: {mode: painted}}

  /* ---------- small helpers ---------- */

  let statusTimer = 0;
  function say(text, hold = 0) {
    const status = $("status");
    status.textContent = text;
    status.classList.toggle("show", Boolean(text));
    clearTimeout(statusTimer);
    if (hold) statusTimer = setTimeout(() => status.classList.remove("show"), hold);
  }

  function setPhase(phase) {
    game.phase = phase;
    board.dataset.phase = phase;
    board.dataset.ball = `${game.x},${game.y}`;
    refreshControls();
    refreshCaption();
  }

  function totalShots() {
    return game.shots.you + game.shots.model;
  }

  function refreshScore() {
    $("shots").textContent = totalShots();
    $("shots-label").textContent = totalShots() === 1 ? "shot" : "shots";
    $("shots-you").textContent = game.shots.you;
    $("shots-model").textContent = game.shots.model;
  }

  function currentModel() {
    return models[game.course.name];
  }

  function refreshControls() {
    const plan = $("plan"), entry = currentModel();
    const ready = entry && entry.state === "ready";
    const planning = game.phase === "planning" || game.phase === "previewing";
    plan.textContent = planning ? "Stop planning" : "Let the world model plan";
    plan.classList.toggle("busy", planning);
    plan.classList.toggle("waiting", !planning && (!ready || !["idle"].includes(game.phase)));
    const note = $("model-note");
    if (!entry || entry.state === "missing") {
      note.textContent = `No trained model for ${game.course.name} yet. You can still play it.`;
    } else if (entry.state === "loading") {
      note.textContent = "Unpacking the model...";
    } else if (entry.state === "broken") {
      note.textContent = "This browser could not unpack the model. Try a current Chrome, Edge, Safari or Firefox.";
    } else if (entry.model.shotCost === null && view.mode === "graph") {
      note.textContent = "This model has no shot graph. Switch to straight-line latent distance.";
    } else {
      note.textContent = `${P.SAMPLES} imagined shot sequences per round, ${P.ITERATIONS} rounds.`;
    }
  }

  function refreshCaption() {
    let text;
    if (game.phase === "planning" || game.phase === "previewing") {
      text = "The faint trails are shots the model is imagining, not real ones. Each dot is where it expects the ball "
        + "to stop, looked up from its memory of the course. The best few guide the next round, so the trails should narrow.";
    } else if (view.mind && view.mode === "latent") {
      text = "Colour shows how far each spot is from the hole, measured in a straight line through the model's "
        + "learned space. Pale means close. Look along a wall: does the colour change there, or flow straight across it?";
    } else if (view.mind) {
      text = "Colour shows roughly how many shots each spot needs, counted on a graph of shots the model saw land. "
        + "Pale means few. Switch between the two measures and watch what happens at the walls.";
    } else if (game.landing) {
      text = "The ring marks where the model imagined this shot would stop. The gap between ring and ball is its "
        + "prediction error. Turn on Mind view to see how it judges the rest of the course.";
    } else {
      text = "This model learned by watching random shots as tiny 64 by 64 pictures, with no score and no rules. "
        + "Play a shot yourself, then let it plan. Our question: does distance in its mind respect walls?";
    }
    $("caption").textContent = text;
  }

  /* ---------- models ---------- */

  async function ensureModel(name) {
    if (models[name]) return models[name];
    const entry = { state: "missing", heat: {} };
    models[name] = entry;
    const data = window.PUTT_MODEL && window.PUTT_MODEL.courses && window.PUTT_MODEL.courses[name];
    if (!data) return entry;
    entry.state = "loading";
    refreshControls();
    try {
      entry.model = await P.unpack(data);
      entry.state = "ready";
    } catch (error) {
      console.error(error);
      entry.state = "broken";
    }
    refreshControls();
    refreshHeat();
    return entry;
  }

  function heatFor(entry, mode) {
    if (!entry || entry.state !== "ready") return null;
    if (!(mode in entry.heat)) {
      const costs = P.costMap(entry.model, mode);
      entry.heat[mode] = costs ? R.paintHeat(costs, entry.model.grid) : null;
    }
    return entry.heat[mode];
  }

  function refreshHeat() {
    const next = view.mind ? heatFor(currentModel(), view.mode) : null;
    if (next !== view.heat) {
      view.fading = view.heat;
      view.fadeBegan = performance.now();
      view.heat = next;
    }
    const legend = $("legend");
    legend.hidden = !next;
    if (next) {
      const info = MODE_INFO[view.mode];
      const digits = view.mode === "graph" ? 1 : 0;
      $("legend-title").textContent = info.title;
      $("legend-low").textContent = `${next.low.toFixed(digits)}${info.unit}, near`;
      $("legend-high").textContent = `far, ${next.high.toFixed(digits)}${info.unit} or more`;
      moveLegendMarker();
    }
  }

  function moveLegendMarker() {
    const heat = view.heat;
    if (!heat) return;
    const grid = Math.round(Math.sqrt(heat.filled.length));
    const cell = Math.min(Math.floor(game.y * grid), grid - 1) * grid + Math.min(Math.floor(game.x * grid), grid - 1);
    const value = heat.filled[cell];
    const t = Number.isFinite(value) ? (value - heat.low) / (heat.high - heat.low) : 0;
    $("legend-marker").style.left = `${Math.min(Math.max(t, 0), 1) * 100}%`;
  }

  /* ---------- course and reset ---------- */

  function resetBall() {
    game.token++;
    game.x = game.course.start[0];
    game.y = game.course.start[1];
    game.shots = { you: 0, model: 0 };
    game.drag = game.roll = game.plan = game.landing = null;
    game.ghosts = [];
    game.trail = [];
    game.particles = [];
    game.ripples = [];
    refreshScore();
    setPhase("idle");
    moveLegendMarker();
  }

  function selectCourse(name) {
    game.course = G.COURSES[name];
    for (const button of $("courses").children) {
      button.setAttribute("aria-pressed", String(button.dataset.course === name));
    }
    rebuildLayers();
    resetBall();
    ensureModel(name).then(() => { refreshControls(); refreshHeat(); });
    refreshHeat();
    say(`${name[0].toUpperCase()}${name.slice(1)}. Pull back from the ball to aim.`, 2600);
  }

  function rebuildLayers() {
    const S = Math.max(Math.round(canvas.clientWidth * Math.min(devicePixelRatio || 1, 2)), 64);
    canvas.width = S;
    canvas.height = S;
    view.S = S;
    view.layers = R.buildLayers(game.course, S);
  }

  /* ---------- shooting ---------- */

  function shoot(ax, ay, by) {
    const result = G.simulate(game.course, game.x, game.y, ax, ay, true);
    game.shots[by]++;
    refreshScore();
    game.roll = { result, began: performance.now(), by, hit: 0 };
    game.taught = true;
    game.trail = [];
    setPhase("rolling");
    if (by === "you") say("");
  }

  function emitDust(x, y, color, count, spread) {
    if (reducedMotion) return;
    for (let i = 0; i < count; i++) {
      const angle = Math.random() * Math.PI * 2, speed = spread * (0.4 + Math.random());
      game.particles.push({
        x, y, vx: Math.cos(angle) * speed, vy: Math.sin(angle) * speed, life: 0, span: 0.3 + 0.3 * Math.random(),
        size: 0.0025 + 0.003 * Math.random(), color,
      });
    }
  }

  function confetti() {
    if (reducedMotion) return;
    const [hx, hy] = game.course.hole;
    const colors = ["#fff7e0", "#e9553b", "#f2c14e", "#7fd1a0"];
    for (let i = 0; i < 46; i++) {
      const angle = Math.random() * Math.PI * 2, speed = 0.12 + Math.random() * 0.34;
      game.particles.push({
        x: hx, y: hy, vx: Math.cos(angle) * speed, vy: Math.sin(angle) * speed - 0.12, life: 0,
        span: 0.8 + 0.7 * Math.random(), size: 0.004 + 0.004 * Math.random(), color: colors[i % 4], gravity: 0.5,
      });
    }
  }

  function advanceRoll(now) {
    const roll = game.roll;
    const { xs, ys, hits, x, y, sunk } = roll.result;
    const last = xs.length - 1;
    const position = Math.min((now - roll.began) / (G.DT * 1000), last);
    const tick = Math.floor(position), f = position - tick;
    const nx = xs[tick] + (xs[Math.min(tick + 1, last)] - xs[tick]) * f;
    const ny = ys[tick] + (ys[Math.min(tick + 1, last)] - ys[tick]) * f;

    const dx = nx - game.x, dy = ny - game.y, moved = Math.hypot(dx, dy);
    if (moved > 1e-9) R.rollBall(game.spin, dx / moved, dy / moved, moved / G.BALL_RADIUS);
    game.x = nx;
    game.y = ny;
    if (moved > 1e-4) game.trail.push({ x: nx, y: ny, at: now });

    while (roll.hit < hits.length && hits[roll.hit].tick <= position) {
      const impact = hits[roll.hit++];
      emitDust(xs[impact.tick], ys[impact.tick], "#f5e6c8", 3 + Math.round(impact.speed * 4), 0.16);
    }
    if (G.inRects(nx, ny, game.course.sand) && moved > 1e-3 && Math.random() < 0.5) {
      emitDust(nx, ny, "#e4cb93", 1, 0.08);
    }
    moveLegendMarker();

    if (position >= last) {
      game.roll = null;
      game.x = x;
      game.y = y;
      if (sunk) {
        game.sinkBegan = now;
        setPhase("sinking");
      } else {
        settle(roll.by);
      }
    }
  }

  function settle(by) {
    setPhase("idle");
    if (by === "model" && game.shots.model >= MAX_SHOTS) {
      say(`${game.shots.model} model shots. The evaluation stops counting at ${MAX_SHOTS}, but you can carry on.`, 4500);
    } else if (by === "model") {
      say("Ball at rest. Your turn, or let the model plan again.", 3000);
    } else {
      say("Ball at rest. Take another shot.", 2200);
    }
  }

  function finishSink() {
    setPhase("sunk");
    game.ripples.push({ began: performance.now() });
    confetti();
    const total = totalShots();
    say(`In the hole! ${total} ${total === 1 ? "shot" : "shots"}. You ${game.shots.you}, model ${game.shots.model}.`);
    $("reset").classList.add("nudge");
    setTimeout(() => $("reset").classList.remove("nudge"), 1100);
  }

  /* ---------- agent ---------- */

  function explainBusy() {
    if (game.phase === "rolling" || game.phase === "sinking") say("Wait for the ball to stop first.", 1800);
    else if (game.phase === "sunk") {
      say("The ball is already in. Press Reset or pick a course.", 2600);
      $("reset").classList.add("nudge");
      setTimeout(() => $("reset").classList.remove("nudge"), 1100);
    }
  }

  function modelProblem(entry) {
    if (!entry || entry.state === "missing") return `No trained model for ${game.course.name} yet, so there is nothing to plan with. You can still play.`;
    if (entry.state === "loading") return "The model is still unpacking. Try again in a moment.";
    if (entry.state === "broken") return "This browser could not unpack the model.";
    if (view.mode === "graph" && !entry.model.shotCost) return "This model has no shot graph. Use straight-line latent distance.";
    return "";
  }

  function stopPlanning(message) {
    game.token++;
    game.ghosts = [];
    game.plan = null;
    setPhase("idle");
    say(message, 2600);
  }

  async function planShot() {
    if (game.phase === "planning" || game.phase === "previewing") return stopPlanning("Planning stopped.");
    if (game.phase !== "idle") return explainBusy();
    const entry = currentModel();
    const problem = modelProblem(entry);
    if (problem) return say(problem, 3600);

    const token = ++game.token;
    const mode = view.mode, model = entry.model;
    game.ghosts = [];
    game.plan = null;
    game.landing = null;
    setPhase("planning");
    say(`Imagining shots: round 1 of ${P.ITERATIONS}`);

    const start = P.latentAt(model, game.x, game.y);
    const search = P.cem(model, mode, start);
    const nextFrame = () => new Promise(requestAnimationFrame);
    const origin = [game.x, game.y];
    let step = search.next();
    while (!step.done) {
      const sliceEnd = performance.now() + SLICE_MS;
      while (!step.done && performance.now() < sliceEnd) {
        if (step.value.elites) {
          const { iteration, elites } = step.value;
          const born = performance.now();
          game.ghosts = game.ghosts.filter((ghost) => ghost.iteration >= iteration - 1);
          for (const points of elites) game.ghosts.push({ points: [origin, ...points], iteration, born });
          if (iteration < P.ITERATIONS) say(`Imagining shots: round ${iteration + 1} of ${P.ITERATIONS}`);
        }
        step = search.next();
      }
      await nextFrame();
      if (token !== game.token) return;
    }

    const { actions, imagined } = step.value;
    game.plan = { points: [origin, ...imagined] };
    setPhase("previewing");
    say("Best plan found. Taking its first shot.");
    await new Promise((resolve) => setTimeout(resolve, reducedMotion ? 300 : 1100));
    if (token !== game.token) return;
    game.ghosts = [];
    game.landing = imagined[0];
    game.plan = null;
    shoot(actions[0], actions[1], "model");
    say(`Model shot ${game.shots.model}. The ring marks where it expects the ball to stop.`);
  }

  /* ---------- pointer input ---------- */

  function toCourse(event) {
    const box = canvas.getBoundingClientRect();
    return [(event.clientX - box.left) / box.width, (event.clientY - box.top) / box.height];
  }

  function updateDrag(px, py) {
    const pullX = game.x - px, pullY = game.y - py;
    const length = Math.hypot(pullX, pullY);
    const power = Math.min(length / PULL_MAX, 1);
    game.drag = {
      px, py, power, length,
      ax: length > 0 ? (pullX / length) * power : 0,
      ay: length > 0 ? (pullY / length) * power : 0,
    };
    if (length < PULL_MIN) say("Pull further back to shoot, or let go to cancel.");
    else say(`Power ${Math.round(power * 100)}%. Release to putt.`);
  }

  canvas.addEventListener("pointerdown", (event) => {
    const [px, py] = toCourse(event);
    if (game.phase === "planning" || game.phase === "previewing") return say("The model is planning. Press Stop planning to take over.", 2200);
    if (game.phase !== "idle") return explainBusy();
    if (Math.hypot(px - game.x, py - game.y) > GRAB_RADIUS) return say("Grab the ball, then pull back to aim.", 2000);
    canvas.setPointerCapture(event.pointerId);
    game.landing = null;
    setPhase("aiming");
    updateDrag(px, py);
  });

  canvas.addEventListener("pointermove", (event) => {
    const [px, py] = toCourse(event);
    if (game.phase === "aiming") {
      updateDrag(px, py);
    } else {
      const near = game.phase === "idle" && Math.hypot(px - game.x, py - game.y) <= GRAB_RADIUS;
      canvas.style.cursor = near ? "grab" : "";
    }
  });

  function endDrag(release) {
    if (game.phase !== "aiming") return;
    const drag = game.drag;
    game.drag = null;
    canvas.style.cursor = "";
    if (release && drag.length >= PULL_MIN) {
      shoot(drag.ax, drag.ay, "you");
    } else {
      setPhase("idle");
      say("Shot cancelled.", 1400);
    }
  }
  canvas.addEventListener("pointerup", () => endDrag(true));
  canvas.addEventListener("pointercancel", () => endDrag(false));
  addEventListener("keydown", (event) => { if (event.key === "Escape") endDrag(false); });

  /* ---------- controls ---------- */

  for (const name of Object.keys(G.COURSES)) {
    const button = document.createElement("button");
    button.type = "button";
    button.dataset.course = name;
    const thumb = document.createElement("canvas");
    thumb.width = thumb.height = 108;
    R.paintThumb(thumb, G.COURSES[name]);
    button.append(thumb, name[0].toUpperCase() + name.slice(1));
    button.addEventListener("click", () => selectCourse(name));
    $("courses").append(button);
  }

  $("reset").addEventListener("click", () => {
    resetBall();
    say("Back on the tee.", 1500);
  });
  $("plan").addEventListener("click", planShot);

  for (const radio of document.querySelectorAll("input[name=mode]")) {
    radio.addEventListener("change", () => {
      view.mode = radio.value;
      if (game.phase === "planning" || game.phase === "previewing") stopPlanning("Scoring changed, so planning stopped. Press plan to start again.");
      else say(`Scoring with ${MODE_INFO[view.mode].name}.`, 1800);
      refreshHeat();
      refreshControls();
      refreshCaption();
    });
  }

  $("mind").addEventListener("change", (event) => {
    view.mind = event.target.checked;
    const entry = currentModel();
    if (view.mind && (!entry || entry.state !== "ready")) {
      say(modelProblem(entry) || "No model to look inside here.", 3200);
      event.target.checked = false;
      view.mind = false;
    } else if (view.mind && !heatFor(entry, view.mode)) {
      say(modelProblem(entry), 3200);
    }
    refreshHeat();
    refreshCaption();
  });

  /* ---------- drawing ---------- */

  function firstObstacle(x, y, dx, dy, reach) {
    for (let d = 0; d < reach; d += 0.006) {
      if (!G.isFree(game.course, x + dx * d, y + dy * d)) return d;
    }
    return reach;
  }

  function drawAim(S) {
    const drag = game.drag;
    if (!drag || drag.length < PULL_MIN) return;
    const r = G.BALL_RADIUS * S, x = game.x * S, y = game.y * S;
    const dirX = drag.ax / drag.power, dirY = drag.ay / drag.power;
    const speed = drag.power * G.MAX_SPEED;
    // How far the shot would roll on grass, cut short at the first wall.
    const reach = firstObstacle(game.x, game.y, dirX, dirY, (speed * speed) / (2 * G.FRICTION));

    ctx.save();
    ctx.lineCap = "round";
    // The pull-back: a thin band from the ball to the pointer.
    ctx.strokeStyle = "rgb(255 255 255 / 0.3)";
    ctx.lineWidth = 0.004 * S;
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(drag.px * S, drag.py * S);
    ctx.stroke();
    ctx.fillStyle = "rgb(255 255 255 / 0.55)";
    ctx.beginPath();
    ctx.arc(drag.px * S, drag.py * S, 0.008 * S, 0, Math.PI * 2);
    ctx.fill();

    // Dotted aim line.
    const hue = 48 - 40 * drag.power;
    const gap = 0.024;
    ctx.shadowColor = "rgb(4 20 10 / 0.5)";
    ctx.shadowBlur = 0.003 * S;
    ctx.shadowOffsetY = 0.0015 * S;
    for (let d = 0.036; d < reach; d += gap) {
      const fade = 1 - (d / Math.max(reach, 0.05)) * 0.55;
      ctx.fillStyle = `rgb(255 251 238 / ${fade})`;
      ctx.beginPath();
      ctx.arc((game.x + dirX * d) * S, (game.y + dirY * d) * S, 0.0056 * S * (0.6 + 0.4 * fade), 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.shadowColor = "transparent";

    // Power arc around the ball.
    ctx.strokeStyle = `hsl(${hue} 95% 66%)`;
    ctx.lineWidth = 0.0058 * S;
    ctx.beginPath();
    const facing = Math.atan2(dirY, dirX);
    ctx.arc(x, y, r * 1.95, facing - Math.PI * drag.power, facing + Math.PI * drag.power);
    ctx.stroke();
    ctx.restore();
  }

  function tracePoints(points, S) {
    ctx.beginPath();
    ctx.moveTo(points[0][0] * S, points[0][1] * S);
    for (let i = 1; i < points.length; i++) ctx.lineTo(points[i][0] * S, points[i][1] * S);
  }

  function drawGhosts(S, now) {
    const newest = game.ghosts.reduce((most, ghost) => Math.max(most, ghost.iteration), 0);
    const onHeat = view.heat !== null;
    ctx.save();
    ctx.lineJoin = ctx.lineCap = "round";
    for (const ghost of game.ghosts) {
      const appear = Math.min((now - ghost.born) / 260, 1);
      const alpha = (ghost.iteration === newest ? 0.4 : 0.13) * appear;
      ctx.lineWidth = 0.0028 * S;
      for (let i = 1; i < ghost.points.length; i++) {
        // Later shots in a sequence are less certain, so they fade.
        const shade = alpha * (1 - 0.22 * (i - 1));
        const color = onHeat ? `rgb(255 196 92 / ${shade})` : `rgb(255 255 255 / ${shade})`;
        ctx.strokeStyle = color;
        ctx.fillStyle = color;
        tracePoints(ghost.points.slice(i - 1, i + 1), S);
        ctx.stroke();
        ctx.beginPath();
        ctx.arc(ghost.points[i][0] * S, ghost.points[i][1] * S, 0.0056 * S, 0, Math.PI * 2);
        ctx.fill();
      }
    }
    if (game.plan) {
      const { points } = game.plan;
      ctx.strokeStyle = "#ffd76a";
      ctx.lineWidth = 0.0052 * S;
      ctx.setLineDash([0.016 * S, 0.012 * S]);
      ctx.lineDashOffset = -now * 0.04;
      tracePoints(points, S);
      ctx.stroke();
      ctx.setLineDash([]);
      points.slice(1).forEach(([px, py], i) => {
        ctx.fillStyle = "#ffd76a";
        ctx.beginPath();
        ctx.arc(px * S, py * S, 0.0105 * S, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = "#3a2c05";
        ctx.font = `600 ${0.0138 * S}px ${getComputedStyle(document.body).fontFamily}`;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(String(i + 1), px * S, py * S + 0.0006 * S);
      });
    }
    ctx.restore();
  }

  function drawLanding(S) {
    if (!game.landing) return;
    const [x, y] = game.landing;
    ctx.save();
    ctx.strokeStyle = "#ffd76a";
    ctx.lineWidth = 0.004 * S;
    ctx.setLineDash([0.008 * S, 0.006 * S]);
    ctx.beginPath();
    ctx.arc(x * S, y * S, 0.021 * S, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();
  }

  function drawTrail(S, now) {
    const points = game.trail;
    while (points.length && now - points[0].at > 420) points.shift();
    ctx.save();
    ctx.lineCap = "round";
    for (let i = 1; i < points.length; i++) {
      const fade = 1 - (now - points[i].at) / 420;
      ctx.strokeStyle = `rgb(255 255 255 / ${0.3 * fade})`;
      ctx.lineWidth = G.BALL_RADIUS * S * 1.1 * fade;
      ctx.beginPath();
      ctx.moveTo(points[i - 1].x * S, points[i - 1].y * S);
      ctx.lineTo(points[i].x * S, points[i].y * S);
      ctx.stroke();
    }
    ctx.restore();
  }

  function drawParticles(S, delta) {
    game.particles = game.particles.filter((p) => (p.life += delta) < p.span);
    ctx.save();
    for (const p of game.particles) {
      p.x += p.vx * delta;
      p.y += p.vy * delta;
      if (p.gravity) p.vy += p.gravity * delta;
      p.vx *= 1 - 1.5 * delta;
      ctx.globalAlpha = 1 - p.life / p.span;
      ctx.fillStyle = p.color;
      ctx.beginPath();
      ctx.arc(p.x * S, p.y * S, p.size * S, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();
  }

  function drawRipples(S, now) {
    const [x, y] = game.course.hole;
    game.ripples = game.ripples.filter((ripple) => now - ripple.began < 900);
    ctx.save();
    for (const ripple of game.ripples) {
      const t = (now - ripple.began) / 900;
      ctx.strokeStyle = `rgb(255 247 224 / ${0.7 * (1 - t)})`;
      ctx.lineWidth = 0.005 * S * (1 - t);
      ctx.beginPath();
      ctx.arc(x * S, y * S, (G.HOLE_RADIUS + 0.13 * t) * S, 0, Math.PI * 2);
      ctx.stroke();
    }
    ctx.restore();
  }

  function drawBallState(S, now) {
    const r = G.BALL_RADIUS * S;
    if (game.phase === "sunk") return;
    if (game.phase === "sinking") {
      const t = Math.min((now - game.sinkBegan) / 240, 1);
      ctx.save();
      ctx.globalAlpha = 1 - t * 0.85;
      R.drawBall(ctx, game.x * S, game.y * S, r * (1 - 0.4 * t), game.spin, 1 - t);
      ctx.restore();
      if (t >= 1) finishSink();
      return;
    }
    if (game.phase === "idle" && !game.taught) {
      // Until the first drag, a slow pulse shows where to start.
      const t = (now % 1700) / 1700;
      ctx.save();
      ctx.strokeStyle = `rgb(255 255 255 / ${0.55 * (1 - t)})`;
      ctx.lineWidth = 0.004 * S;
      ctx.beginPath();
      ctx.arc(game.x * S, game.y * S, r * (1.3 + 1.7 * t), 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();
    }
    R.drawBall(ctx, game.x * S, game.y * S, r, game.spin);
  }

  function drawHeat(now) {
    const S = view.S;
    const mix = Math.min((now - view.fadeBegan) / 320, 1);
    ctx.save();
    ctx.imageSmoothingQuality = "high";
    if (view.fading && mix < 1) {
      ctx.globalAlpha = HEAT_ALPHA * (1 - mix);
      ctx.drawImage(view.fading.canvas, 0, 0, S, S);
    }
    if (view.heat) {
      ctx.globalAlpha = HEAT_ALPHA * mix;
      ctx.drawImage(view.heat.canvas, 0, 0, S, S);
    }
    ctx.restore();
  }

  let lastFrame = performance.now();
  function frame(now) {
    const delta = Math.min((now - lastFrame) / 1000, 0.05);
    lastFrame = now;
    if (game.roll) advanceRoll(now);
    const S = view.S;
    ctx.drawImage(view.layers.ground, 0, 0);
    drawHeat(now);
    ctx.drawImage(view.layers.top, 0, 0);
    drawRipples(S, now);
    drawTrail(S, now);
    drawGhosts(S, now);
    drawLanding(S);
    drawAim(S);
    drawBallState(S, now);
    const toHole = Math.hypot(game.x - game.course.hole[0], game.y - game.course.hole[1] + 0.07);
    R.drawFlag(ctx, game.course, S, now / 1000, toHole < 0.09 && game.phase !== "sunk" ? 0.45 : 1);
    drawParticles(S, delta);
    requestAnimationFrame(frame);
  }

  /* ---------- start ---------- */

  let resizeTimer = 0;
  new ResizeObserver(() => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(rebuildLayers, 120);
  }).observe(board);

  selectCourse("open");
  say("Pull back from the ball and release to putt.");
  requestAnimationFrame(frame);
})();
