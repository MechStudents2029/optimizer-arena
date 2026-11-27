import { describe, expect, it } from "vitest";
import {
  adagrad,
  adamw,
  cosine,
  initState,
  lion,
  quadratic,
  race,
  rosenbrock,
  run,
  stepDecay,
  warmupCosine,
  type GradFn,
  type Vec2,
} from "../src/index.ts";

const constGrad =
  (g: Vec2): GradFn =>
  () =>
    g;
const OFF_LATTICE: Vec2 = [-2 * Math.SQRT2, Math.E / 2];

describe("schedule edge cases", () => {
  it("cosine clamps t below 0 to the start value", () => {
    expect(cosine(100)(-5)).toBe(1);
  });

  it("step decay with γ = 1 is the constant schedule", () => {
    const s = stepDecay(3, 1);
    for (const t of [0, 3, 30, 3000]) expect(s(t)).toBe(1);
  });

  it("a one-step warmup takes the first step at the full rate", () => {
    const s = warmupCosine(100, 1);
    expect(s(0)).toBe(1);
    expect(s(1)).toBe(1);
  });

  it("race with schedule 'constant' reproduces the default race exactly", () => {
    const a = race(rosenbrock, { steps: 500 });
    const b = race(rosenbrock, { steps: 500, schedule: "constant" });
    expect(b.map((e) => [e.optimizer, e.lr, e.steps, e.finalLoss])).toEqual(
      a.map((e) => [e.optimizer, e.lr, e.steps, e.finalLoss]),
    );
  });
});

describe("optimizer edge cases", () => {
  it("AdamW's decay is scaled by the schedule too: s = ½ halves the shrink rate", () => {
    const opt = adamw({ lr: 0.1, weightDecay: 0.2, schedule: () => 0.5 });
    const s = opt.step(initState([3, -2]), constGrad([0, 0]));
    expect(s.params[0]).toBeCloseTo(3 * (1 - 0.5 * 0.1 * 0.2), 14);
    expect(s.params[1]).toBeCloseTo(-2 * (1 - 0.5 * 0.1 * 0.2), 14);
  });

  it("AdaGrad freezes instead of producing NaN when g² overflows", () => {
    const s = adagrad({ lr: 0.1 }).step(initState([1, 1]), constGrad([1e200, 1]));
    expect(s.v[0]).toBe(Infinity);
    expect(s.params[0]).toBe(1); // g / √∞ = 0
    expect(s.params[1]).toBeCloseTo(0.9, 6);
  });

  it("Lion turns a NaN gradient into a NaN step, which run() reports as divergence", () => {
    const s = lion({ lr: 0.1 }).step(initState([1, 1]), constGrad([NaN, 0]));
    expect(Number.isNaN(s.params[0])).toBe(true);
    const nanSurface = { ...quadratic, grad: (): Vec2 => [NaN, NaN] };
    expect(run(nanSurface, lion({ lr: 0.1 }), { steps: 5 }).diverged).toBe(true);
  });

  it("Lion's constant-rate floor on the bowl grows with η (≈ η² scale)", () => {
    const floor = (lr: number) =>
      run(quadratic, lion({ lr }), { steps: 3000, start: OFF_LATTICE, tol: 0 }).finalLoss;
    const small = floor(0.01);
    const big = floor(0.1);
    expect(big).toBeGreaterThan(10 * small);
    // it ends within one step of the minimum per coordinate: ½(x² + 10y²) ≤ 5.5η²
    expect(big).toBeLessThanOrEqual(5.5 * 0.1 ** 2);
    expect(small).toBeLessThanOrEqual(5.5 * 0.01 ** 2);
  });
});
