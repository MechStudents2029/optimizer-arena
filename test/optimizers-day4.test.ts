import { describe, expect, it } from "vitest";
import {
  adagrad,
  adam,
  adamw,
  booth,
  initState,
  lion,
  makeOptimizer,
  makeSchedule,
  optimizerIds,
  quadratic,
  rosenbrock,
  run,
  type GradFn,
  type Vec2,
} from "../src/index.ts";

const constGrad =
  (g: Vec2): GradFn =>
  () =>
    g;
const zeroGrad = constGrad([0, 0]);
/** A start off every η-lattice, so Lion cannot land on the minimum by accident. */
const OFF_LATTICE: Vec2 = [-2 * Math.SQRT2, Math.E / 2];

describe("AdaGrad", () => {
  it("first step is η·sign(g) (with ε = 0), second divides by √(g₁² + g₂²)", () => {
    const opt = adagrad({ lr: 0.1, eps: 0 });
    const s1 = opt.step(initState([0, 0]), constGrad([2, -8]));
    expect(s1.params[0]).toBeCloseTo(-0.1, 14);
    expect(s1.params[1]).toBeCloseTo(0.1, 14);
    expect(s1.v).toEqual([4, 64]);
    const s2 = opt.step(s1, constGrad([1, 0]));
    // G = 4 + 1 = 5 → step 0.1·1/√5 ; y: G = 64, g = 0 → no move
    expect(s2.params[0]).toBeCloseTo(-0.1 - 0.1 / Math.sqrt(5), 14);
    expect(s2.params[1]).toBeCloseTo(0.1, 14);
    expect(s2.v).toEqual([5, 64]);
  });

  it("under a constant gradient the k-th step is exactly η/√k: the step only shrinks", () => {
    const opt = adagrad({ lr: 0.5, eps: 0 });
    let s = initState([0, 0]);
    let prev = 0;
    let expected = 0;
    for (let k = 1; k <= 100; k++) {
      const next = opt.step(s, constGrad([3, 3]));
      const size = s.params[0] - next.params[0];
      expect(size).toBeCloseTo(0.5 / Math.sqrt(k), 12);
      if (k > 1) expect(size).toBeLessThan(prev);
      prev = size;
      expected += 0.5 / Math.sqrt(k);
      s = next;
    }
    expect(-s.params[0]).toBeCloseTo(expected, 10);
  });

  it("a zero gradient with the default ε leaves θ alone (no 0/0)", () => {
    const s = adagrad({ lr: 1 }).step(initState([1, -1]), zeroGrad);
    expect(s.params).toEqual([1, -1]);
    expect(Number.isNaN(s.params[0])).toBe(false);
  });

  it("converges on the convex bowl at η = 1 but stalls at η = 0.05", () => {
    expect(run(quadratic, adagrad({ lr: 1 }), { steps: 3000 }).stepsToTol).not.toBeNull();
    const slow = run(quadratic, adagrad({ lr: 0.05 }), { steps: 3000 });
    expect(slow.stepsToTol).toBeNull();
    expect(slow.diverged).toBe(false);
  });
});

describe("AdamW", () => {
  it("first step is θ − η(sign(g) + λθ) with ε = 0", () => {
    const s = adamw({ lr: 0.1, eps: 0, weightDecay: 0.5 }).step(initState([2, -1]), constGrad([1, 1]));
    // x: 2 − 0.1·(1 + 0.5·2) = 1.8 ; y: −1 − 0.1·(1 + 0.5·(−1)) = −1.05
    expect(s.params[0]).toBeCloseTo(1.8, 14);
    expect(s.params[1]).toBeCloseTo(-1.05, 14);
  });

  it("with λ = 0 it is Adam (to rounding: η·(a/b) vs (η·a)/b)", () => {
    const a = run(rosenbrock, adam({ lr: 0.05 }), { steps: 300, tol: 0 });
    const w = run(rosenbrock, adamw({ lr: 0.05, weightDecay: 0 }), { steps: 300, tol: 0 });
    expect(w.points).toHaveLength(a.points.length);
    w.points.forEach((p, k) => {
      expect(p[0]).toBeCloseTo(a.points[k]![0], 10);
      expect(p[1]).toBeCloseTo(a.points[k]![1], 10);
    });
  });

  it("decoupled decay on a flat surface shrinks θ by exactly (1 − ηλ) per step", () => {
    const opt = adamw({ lr: 0.1, weightDecay: 0.2 });
    let s = initState([3, -2]);
    for (let k = 0; k < 50; k++) s = opt.step(s, zeroGrad);
    expect(s.params[0]).toBeCloseTo(3 * 0.98 ** 50, 12);
    expect(s.params[1]).toBeCloseTo(-2 * 0.98 ** 50, 12);
  });

  it("L2-in-Adam on the same flat surface ignores λ's size: the first step is η·sign(θ) for any λ", () => {
    // L2 regularisation adds λθ to the gradient, and Adam then normalises it away.
    for (const lambda of [1e-4, 0.2, 10]) {
      const l2: GradFn = (p) => [lambda * p[0], lambda * p[1]];
      const s = adam({ lr: 0.1, eps: 0 }).step(initState([3, -2]), l2);
      expect(s.params[0]).toBeCloseTo(2.9, 12);
      expect(s.params[1]).toBeCloseTo(-1.9, 12);
    }
  });

  it("decay moves the fixed point toward the origin: on Booth it stops short of (1, 3)", () => {
    const r = run(booth, adamw({ lr: 0.3, weightDecay: 0.01 }), { steps: 10_000, tol: 0 });
    expect(r.diverged).toBe(false);
    const [x, y] = r.final;
    expect(Math.hypot(x, y)).toBeLessThan(Math.hypot(1, 3));
    expect(Math.hypot(x - 1, y - 3)).toBeGreaterThan(1e-4);
    expect(Math.hypot(x - 1, y - 3)).toBeLessThan(1e-2);
  });
});

describe("Lion", () => {
  it("first step is −η·sign(g), and m only takes (1 − β₂)g", () => {
    const s = lion({ lr: 0.1 }).step(initState([1, 1]), constGrad([5, -1e-6]));
    expect(s.params[0]).toBeCloseTo(0.9, 14);
    expect(s.params[1]).toBeCloseTo(1.1, 14);
    expect(s.m[0]).toBeCloseTo(0.05, 14);
    expect(s.m[1]).toBeCloseTo(-1e-8, 20);
  });

  it("second step uses sign(β₁m + (1 − β₁)g), not sign(g)", () => {
    const opt = lion({ lr: 0.1, beta1: 0.9, beta2: 0.99 });
    const s1 = opt.step(initState([0, 0]), constGrad([1, 1]));
    const s2 = opt.step(s1, constGrad([-0.05, -0.2]));
    // c = 0.9·0.01 + 0.1·g = [0.004, −0.011]: x keeps going despite g_x < 0, y turns
    expect(s2.params[0]).toBeCloseTo(-0.2, 14);
    expect(s2.params[1]).toBeCloseTo(0, 14);
    // m = 0.99·0.01 + 0.01·g
    expect(s2.m[0]).toBeCloseTo(0.0094, 14);
    expect(s2.m[1]).toBeCloseTo(0.0079, 14);
  });

  it("every coordinate moves by exactly η whatever the gradient's scale", () => {
    for (const g of [1e-9, 1, 1e9]) {
      const s = lion({ lr: 0.03 }).step(initState([0, 0]), constGrad([g, -g]));
      expect(Math.abs(s.params[0])).toBeCloseTo(0.03, 14);
      expect(Math.abs(s.params[1])).toBeCloseTo(0.03, 14);
    }
  });

  it("keeps one buffer: the second-moment slot is never written", () => {
    const opt = lion({ lr: 0.01 });
    let s = initState([1, 1]);
    for (let k = 0; k < 20; k++) s = opt.step(s, (p) => [p[0], p[1]]);
    expect(s.v).toEqual([0, 0]);
  });

  it("does not move on a zero gradient (sign 0 = 0) unless weight decay is on", () => {
    expect(lion({ lr: 0.1 }).step(initState([2, 2]), zeroGrad).params).toEqual([2, 2]);
    const d = lion({ lr: 0.1, weightDecay: 0.5 }).step(initState([2, 2]), zeroGrad);
    expect(d.params[0]).toBeCloseTo(1.9, 14);
  });

  it("from an integer start with η = 1 it lands exactly on the bowl's minimum in 4 steps", () => {
    // (−4, 2) is on the lattice θ₀ + ηℤ², and so is the minimum (0, 0).
    const r = run(quadratic, lion({ lr: 1 }), { steps: 100 });
    expect(r.stepsToTol).toBe(4);
    expect(r.final).toEqual([0, 0]);
  });

  it("off the lattice, a constant η leaves it rattling; a decaying schedule lets it settle", () => {
    const flat = run(quadratic, lion({ lr: 0.01 }), { steps: 3000, start: OFF_LATTICE });
    expect(flat.stepsToTol).toBeNull();
    expect(flat.finalLoss).toBeGreaterThan(1e-5);
    const cos = run(quadratic, lion({ lr: 0.01, schedule: makeSchedule("cosine", 3000) }), {
      steps: 3000,
      start: OFF_LATTICE,
    });
    expect(cos.stepsToTol).not.toBeNull();
  });
});

describe("all eight optimizers", () => {
  it("are built by makeOptimizer with matching ids", () => {
    expect(optimizerIds).toHaveLength(8);
    for (const id of optimizerIds) expect(makeOptimizer(id, 0.1).id).toBe(id);
  });

  it("a schedule of 0 freezes every one of them, decay included", () => {
    for (const id of optimizerIds) {
      const r = run(quadratic, makeOptimizer(id, 0.1, {}, () => 0), { steps: 20, tol: 0 });
      for (const p of r.points) expect(p).toEqual(quadratic.start);
    }
  });

  it("a schedule of 1 is the same as no schedule", () => {
    for (const id of optimizerIds) {
      const a = run(rosenbrock, makeOptimizer(id, 0.001), { steps: 200, tol: 0 });
      const b = run(rosenbrock, makeOptimizer(id, 0.001, {}, () => 1), { steps: 200, tol: 0 });
      expect(b.points).toEqual(a.points);
    }
  });

  it("weightDecay, lionBeta1 and lionBeta2 reach the optimizers through makeOptimizer", () => {
    const a = run(rosenbrock, makeOptimizer("adamw", 0.01, { weightDecay: 0.3 }), { steps: 50, tol: 0 });
    const b = run(rosenbrock, adamw({ lr: 0.01, weightDecay: 0.3 }), { steps: 50, tol: 0 });
    expect(a.points).toEqual(b.points);
    const c = run(rosenbrock, makeOptimizer("lion", 0.01, { lionBeta1: 0.5, lionBeta2: 0.9 }), { steps: 50, tol: 0 });
    const d = run(rosenbrock, lion({ lr: 0.01, beta1: 0.5, beta2: 0.9 }), { steps: 50, tol: 0 });
    expect(c.points).toEqual(d.points);
  });
});
