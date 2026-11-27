// Measure what each demo slider does: vary one hyperparameter, keep the rest at
// their defaults, and record steps to f − f* < 1e-6 at a fixed η.
//   npm run sweep
//   npm run sweep -- --schedule cosine   → same sweeps under a learning-rate schedule
import {
  makeOptimizer,
  makeSchedule,
  run,
  scheduleIds,
  surfaceById,
  type Hyperparams,
  type OptimizerId,
  type ScheduleId,
} from "../src/index.ts";

const STEPS = 10_000;
const i = process.argv.indexOf("--schedule");
const scheduleId = (i < 0 ? "constant" : process.argv[i + 1]) as ScheduleId;
if (!scheduleIds.includes(scheduleId)) {
  console.error(`unknown schedule "${scheduleId}"; pick one of ${scheduleIds.join(", ")}`);
  process.exit(1);
}
const schedule = makeSchedule(scheduleId, STEPS);

interface Sweep {
  title: string;
  surface: string;
  optimizer: OptimizerId;
  lr: number;
  key: keyof Hyperparams;
  values: number[];
  /** Override the start point (Lion needs one off its η-lattice, see docs/lr-schedules.md). */
  start?: [number, number];
  /** Override the schedule for this sweep. */
  schedule?: ScheduleId;
}

// Beale's start (1, 1.5) and minimum (3, 0.5) differ by whole multiples of 0.01, so
// Lion at η = 0.01 can land on the minimum exactly; nudge it off that lattice.
const BEALE_OFF: [number, number] = [1 + Math.SQRT2 * 1e-3, 1.5 + Math.E * 1e-3];

const sweeps: Sweep[] = [
  { title: "Momentum β", surface: "quadratic", optimizer: "momentum", lr: 0.01, key: "beta", values: [0, 0.5, 0.8, 0.9, 0.95, 0.99] },
  { title: "Momentum β", surface: "rosenbrock", optimizer: "momentum", lr: 0.001, key: "beta", values: [0, 0.5, 0.8, 0.9, 0.95, 0.99] },
  { title: "RMSProp ρ", surface: "beale", optimizer: "rmsprop", lr: 0.001, key: "rho", values: [0, 0.5, 0.9, 0.99, 0.999] },
  { title: "Adam β₁", surface: "rosenbrock", optimizer: "adam", lr: 0.1, key: "beta1", values: [0, 0.5, 0.9, 0.95, 0.99] },
  { title: "Adam β₂", surface: "rosenbrock", optimizer: "adam", lr: 0.1, key: "beta2", values: [0.9, 0.99, 0.999, 0.9999] },
  { title: "Adam ε", surface: "quadratic", optimizer: "adam", lr: 0.1, key: "eps", values: [1e-8, 1e-4, 1e-2, 0.1, 1] },
  { title: "AdamW λ", surface: "booth", optimizer: "adamw", lr: 0.3, key: "weightDecay", values: [0, 1e-5, 1e-4, 1e-3, 1e-2, 0.1] },
  { title: "AdamW λ", surface: "quadratic", optimizer: "adamw", lr: 0.3, key: "weightDecay", values: [0, 1e-3, 1e-2, 0.1, 1] },
  { title: "AdaGrad ε", surface: "quadratic", optimizer: "adagrad", lr: 1, key: "eps", values: [1e-8, 1e-4, 1e-2, 0.1, 1] },
  { title: "Lion β₁", surface: "beale", optimizer: "lion", lr: 0.01, key: "lionBeta1", values: [0, 0.5, 0.9, 0.95, 0.99], start: BEALE_OFF, schedule: "cosine" },
  { title: "Lion β₂", surface: "beale", optimizer: "lion", lr: 0.01, key: "lionBeta2", values: [0, 0.9, 0.99, 0.999], start: BEALE_OFF, schedule: "cosine" },
];

if (scheduleId !== "constant") console.log(`Schedule: ${scheduleId} over ${STEPS} steps.`);

for (const s of sweeps) {
  const surface = surfaceById(s.surface);
  const sched = s.schedule ? makeSchedule(s.schedule, STEPS) : schedule;
  const extra = [s.schedule && `${s.schedule} schedule`, s.start && `start (${s.start.map((x) => x.toFixed(4)).join(", ")})`]
    .filter(Boolean)
    .join(", ");
  console.log(`\n### ${s.title} — ${s.optimizer} on ${surface.name}, η = ${s.lr}${extra ? `, ${extra}` : ""}\n`);
  console.log(`| ${s.key} | steps to f − f* < 1e-6 | final f |`);
  console.log("| --- | --- | --- |");
  for (const v of s.values) {
    const t = run(surface, makeOptimizer(s.optimizer, s.lr, { [s.key]: v }, sched), { steps: STEPS, start: s.start });
    const steps = t.diverged ? "diverged" : t.stepsToTol === null ? `— (${STEPS} budget)` : String(t.stepsToTol);
    console.log(`| ${v} | ${steps} | ${t.diverged ? "—" : t.finalLoss.toExponential(1)} |`);
  }
}
