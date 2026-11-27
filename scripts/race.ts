// Race every optimizer on every surface and print a Markdown results table.
//   npm run race                 → table on stdout
//   npm run race -- --json       → raw numbers as JSON
//   npm run race -- --schedule cosine   → every run uses that learning-rate schedule
//   npm run race -- --start-off-lattice → nudge starts by (√2, e)·1e-3 so Lion cannot land exactly
import { optimizerIds, race, scheduleIds, surfaces, ESCAPE_LEVEL, type ScheduleId } from "../src/index.ts";

const STEPS = 10_000;
const TOL = 1e-6;
const names: Record<string, string> = {
  sgd: "GD",
  momentum: "Momentum",
  nesterov: "Nesterov",
  rmsprop: "RMSProp",
  adam: "Adam",
  adagrad: "AdaGrad",
  adamw: "AdamW",
  lion: "Lion",
};

const arg = (name: string): string | undefined => {
  const i = process.argv.indexOf(name);
  return i < 0 ? undefined : process.argv[i + 1];
};
const schedule = (arg("--schedule") ?? "constant") as ScheduleId;
if (!scheduleIds.includes(schedule)) {
  console.error(`unknown schedule "${schedule}"; pick one of ${scheduleIds.join(", ")}`);
  process.exit(1);
}
const offLattice = process.argv.includes("--start-off-lattice");

// The saddle has no minimum to tune for: with a large enough η every method
// just jumps off it in one step. A shared, small η shows how fast each one
// amplifies a 1e-3 nudge off the stable axis.
const SADDLE_LR = 0.01;
const results = surfaces.map((s) => ({
  surface: s,
  entries: race(s, {
    steps: STEPS,
    tol: TOL,
    lrGrid: s.id === "saddle" ? [SADDLE_LR] : undefined,
    schedule,
    start: offLattice && Number.isFinite(s.minValue) ? [s.start[0] + Math.SQRT2 * 1e-3, s.start[1] + Math.E * 1e-3] : undefined,
  }),
}));

if (process.argv.includes("--json")) {
  const out = results.map(({ surface, entries }) => ({
    surface: surface.id,
    start: surface.start,
    schedule,
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
    `\nSchedule: ${schedule}${offLattice ? "; starts nudged by (√2, e)·1e-3, saddle excepted" : ""}. ` +
      `Steps to f − f* < ${TOL} (saddle: steps until f < ${ESCAPE_LEVEL} at a fixed η=${SADDLE_LR}), best η per cell, ` +
      `budget ${STEPS} steps; "—" means the budget ran out, with the final loss shown.`,
  );
}
