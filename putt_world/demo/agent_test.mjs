// Plays full episodes with the exact planner the page uses, to confirm the agent sinks.
// Run: node agent_test.mjs (after export_demo.py).
import { createRequire } from "node:module";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import vm from "node:vm";

const require = createRequire(import.meta.url);
const golf = require("./golf.js");
const planner = require("./planner.js");
const window = {};
vm.runInNewContext(readFileSync(fileURLToPath(new URL("model_data.js", import.meta.url)), "utf8"), { window });

const EPISODES = 10, MAX_SHOTS = 10;
for (const [name, entry] of Object.entries(window.PUTT_MODEL.courses)) {
  const model = await planner.unpack(entry);
  const course = golf.COURSES[name];
  for (const mode of ["latent", "graph"]) {
    let sunk = 0, shots = 0;
    for (let episode = 0; episode < EPISODES; episode++) {
      let [x, y] = course.start;
      for (let shot = 1; shot <= MAX_SHOTS; shot++) {
        const search = planner.cem(model, mode, planner.latentAt(model, x, y));
        let step = search.next();
        while (!step.done) step = search.next();
        const [ax, ay] = step.value.actions;
        const result = golf.simulate(course, x, y, ax, ay);
        x = result.x; y = result.y;
        if (result.sunk) { sunk++; shots += shot; break; }
      }
    }
    console.log(name.padEnd(8), mode.padEnd(7), `sunk ${sunk}/${EPISODES}`, sunk ? `mean shots ${(shots / sunk).toFixed(1)}` : "");
  }
}
