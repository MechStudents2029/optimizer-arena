// Race every optimizer on every surface and print a Markdown results table.
//   npm run race                 → table on stdout
//   npm run race -- --json       → raw numbers as JSON
import { optimizerIds, race, surfaces, ESCAPE_LEVEL } from "../src/index.ts";

const STEPS = 10_000;
const TOL = 1e-6;
const names: Record<string, string> = {
  sgd: "GD",
  momentum: "Momentum",
  nesterov: "Nesterov",
  rmsprop: "RMSProp",
  adam: "Adam",
};

// The saddle has no minimum to tune for: with a large enough η every method
// just jumps off it in one step. A shared, small η shows how fast each one
// amplifies a 1e-3 nudge off the stable axis.
const SADDLE_LR = 0.01;
const results = surfaces.map((s) => ({
  surface: s,
  entries: race(s, { steps: STEPS, tol: TOL, lrGrid: s.id === "saddle" ? [SADDLE_LR] : undefined }),
}));

if (process.argv.includes("--json")) {
  const out = results.map(({ surface, entries }) => ({
    surface: surface.id,
    start: surface.start,
    entries: entries.map(({ trajectory, ...e }) => ({ ...e, final: trajectory?.final ?? null })),
  }));
  console.log(JSON.stringify(out, null, 2));
} else {
  const fmtLoss = (x: number) => (Number.isFinite(x) ? x.toExponential(1) : String(x));
  console.log(`| surface (start) | ${optimizerIds.map((id) => names[id]).join(" | ")} |`);
  console.log(`| --- | ${optimizerIds.map(() => "---").join(" | ")} |`);
  for (const { surface, entries } of results) {
    const cells = entries.map((e) => {
      if (e.lr === null) return "diverged at every η";
      const lr = `η=${e.lr}`;
      return e.steps !== null ? `**${e.steps}** (${lr})` : `— f=${fmtLoss(e.finalLoss)} (${lr})`;
    });
    console.log(`| ${surface.name} (${surface.start.join(", ")}) | ${cells.join(" | ")} |`);
  }
  console.log(
    `\nSteps to f − f* < ${TOL} (saddle: steps until f < ${ESCAPE_LEVEL} at a fixed η=${SADDLE_LR}), best η per cell, ` +
      `budget ${STEPS} steps; "—" means the budget ran out, with the final loss shown.`,
  );
}
