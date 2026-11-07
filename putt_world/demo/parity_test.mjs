// Proves golf.js matches golf.py. Run: node parity_test.mjs
// Regenerate the reference first with make_parity_cases.py.
import { createRequire } from "node:module";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

const require = createRequire(import.meta.url);
const golf = require("./golf.js");
const file = fileURLToPath(new URL("parity_cases.json", import.meta.url));
const reference = JSON.parse(readFileSync(file, "utf8"));
const TOLERANCE = 1e-6;
let failures = 0;
const fail = (message) => { if (failures++ < 10) console.error("FAIL", message); };

for (const [key, value] of Object.entries(reference.constants)) {
  if (golf[key] !== value) fail(`constant ${key}: ${golf[key]} vs ${value}`);
}

let total = 0, totalSunk = 0, totalTicks = 0, worst = 0, worstPath = 0;
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
for (const [name, ref] of Object.entries(reference.courses)) {
  const course = golf.COURSES[name];
  for (const key of ["start", "hole", "walls", "sand", "solid"]) {
    if (!same(course[key], ref[key])) fail(`${name}.${key} differs`);
  }
  let sunk = 0, courseWorst = 0;
  ref.starts.forEach(([x, y], i) => {
    const [ax, ay] = ref.actions[i];
    const out = golf.simulate(course, x, y, ax, ay);
    const error = Math.max(Math.abs(out.x - ref.finals[i][0]), Math.abs(out.y - ref.finals[i][1]));
    courseWorst = Math.max(courseWorst, error);
    if (!(error <= TOLERANCE)) fail(`${name} #${i}: error ${error}`);
    if (+out.sunk !== ref.sunk[i]) fail(`${name} #${i}: sunk ${out.sunk} vs ${ref.sunk[i]}`);
    sunk += ref.sunk[i];
  });
  for (const { index, path } of ref.paths) {
    const [x, y] = ref.starts[index];
    const [ax, ay] = ref.actions[index];
    const out = golf.simulate(course, x, y, ax, ay, true);
    if (out.xs.length !== path.length) fail(`${name} path #${index}: ${out.xs.length} vs ${path.length} ticks`);
    path.forEach(([px, py], t) => {
      const error = Math.max(Math.abs(out.xs[t] - px), Math.abs(out.ys[t] - py));
      worstPath = Math.max(worstPath, error);
      if (!(error <= TOLERANCE)) fail(`${name} path #${index} tick ${t}: error ${error}`);
    });
    totalTicks += path.length;
  }
  console.log(`${name.padEnd(7)} ${ref.starts.length} shots, ${sunk} sunk, worst error ${courseWorst.toExponential(2)}`);
  total += ref.starts.length;
  totalSunk += sunk;
  worst = Math.max(worst, courseWorst);
}
console.log(`total   ${total} shots, ${totalSunk} sunk, worst final error ${worst.toExponential(2)}, `
  + `worst path error ${worstPath.toExponential(2)} over ${totalTicks} ticks`);
console.log(failures ? `FAILED: ${failures} mismatches` : "PASS");
process.exit(failures ? 1 : 0);
