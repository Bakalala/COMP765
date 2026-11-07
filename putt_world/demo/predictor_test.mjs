// Proves the unpacked predictor in model_data.js matches torch, and exercises CEM.
// Run: node predictor_test.mjs (after export_demo.py).
import { createRequire } from "node:module";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import vm from "node:vm";

const require = createRequire(import.meta.url);
const golf = require("./golf.js");
const planner = require("./planner.js");
const path = (name) => fileURLToPath(new URL(name, import.meta.url));
const window = {};
vm.runInNewContext(readFileSync(path("model_data.js"), "utf8"), { window });
const cases = JSON.parse(readFileSync(path("predictor_cases.json"), "utf8"));
const TOLERANCE = 1e-4;
let failed = false;

// erf and GELU against values from Python's math.erf.
const erfs = [[0.1, 0.1124629160182849], [0.5, 0.5204998778130465], [1, 0.8427007929497149],
  [2, 0.9953222650189527], [2.6, 0.9997639655834707], [3.5, 0.9999992569016276]];
for (const [x, value] of erfs) {
  const error = Math.abs(planner.erf(x) - value);
  if (error > 1e-13) { console.error("FAIL erf", x, error); failed = true; }
}

for (const [name, entry] of Object.entries(window.PUTT_MODEL.courses)) {
  const model = await planner.unpack(entry);
  const reference = cases[name];
  let worst = 0, sum = 0, count = 0;
  const out = new Float64Array(model.dim);
  reference.latent.forEach((latent, i) => {
    planner.predict(model, Float64Array.from(latent), reference.action[i][0], reference.action[i][1], out);
    reference.output[i].forEach((value, d) => {
      const error = Math.abs(out[d] - value);
      worst = Math.max(worst, error);
      sum += error;
      count++;
    });
  });

  // A full CEM run from the tee in each cost mode, with timing.
  const course = golf.COURSES[name];
  const start = planner.latentAt(model, course.start[0], course.start[1]);
  const notes = [];
  for (const mode of ["latent", "graph"]) {
    const began = performance.now();
    const search = planner.cem(model, mode, start);
    let step = search.next();
    while (!step.done) step = search.next();
    const { actions, imagined } = step.value;
    const norm = Math.hypot(actions[0], actions[1]);
    if (!(norm <= 1 + 1e-9) || imagined.length !== planner.HORIZON) { failed = true; console.error("FAIL cem", name); }
    notes.push(`${mode} ${(performance.now() - began).toFixed(0)} ms`);
  }
  const ok = worst <= TOLERANCE;
  failed ||= !ok;
  console.log(`${name.padEnd(7)} worst ${worst.toExponential(2)}, mean ${(sum / count).toExponential(2)} over `
    + `${reference.latent.length} inputs: ${ok ? "ok" : "FAIL"}; CEM ${notes.join(", ")}`);
}
console.log(failed ? "FAILED" : "PASS");
process.exit(failed ? 1 : 0);
