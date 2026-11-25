// Measure what each demo slider does: vary one hyperparameter, keep the rest at
// their defaults, and record steps to f − f* < 1e-6 at a fixed η.
//   npm run sweep
import { makeOptimizer, run, surfaceById, type Hyperparams, type OptimizerId } from "../src/index.ts";

const STEPS = 10_000;

interface Sweep {
  title: string;
  surface: string;
  optimizer: OptimizerId;
  lr: number;
  key: keyof Hyperparams;
  values: number[];
}

const sweeps: Sweep[] = [
  { title: "Momentum β", surface: "quadratic", optimizer: "momentum", lr: 0.01, key: "beta", values: [0, 0.5, 0.8, 0.9, 0.95, 0.99] },
  { title: "Momentum β", surface: "rosenbrock", optimizer: "momentum", lr: 0.001, key: "beta", values: [0, 0.5, 0.8, 0.9, 0.95, 0.99] },
  { title: "RMSProp ρ", surface: "beale", optimizer: "rmsprop", lr: 0.001, key: "rho", values: [0, 0.5, 0.9, 0.99, 0.999] },
  { title: "Adam β₁", surface: "rosenbrock", optimizer: "adam", lr: 0.1, key: "beta1", values: [0, 0.5, 0.9, 0.95, 0.99] },
  { title: "Adam β₂", surface: "rosenbrock", optimizer: "adam", lr: 0.1, key: "beta2", values: [0.9, 0.99, 0.999, 0.9999] },
  { title: "Adam ε", surface: "quadratic", optimizer: "adam", lr: 0.1, key: "eps", values: [1e-8, 1e-4, 1e-2, 0.1, 1] },
];

for (const s of sweeps) {
  const surface = surfaceById(s.surface);
  console.log(`\n### ${s.title} — ${s.optimizer} on ${surface.name}, η = ${s.lr}\n`);
  console.log(`| ${s.key} | steps to f − f* < 1e-6 | final f |`);
  console.log("| --- | --- | --- |");
  for (const v of s.values) {
    const t = run(surface, makeOptimizer(s.optimizer, s.lr, { [s.key]: v }), { steps: STEPS });
    const steps = t.diverged ? "diverged" : t.stepsToTol === null ? `— (${STEPS} budget)` : String(t.stepsToTol);
    console.log(`| ${v} | ${steps} | ${t.diverged ? "—" : t.finalLoss.toExponential(1)} |`);
  }
}
