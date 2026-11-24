import { describe, expect, it } from "vitest";
import {
  adam,
  initState,
  makeOptimizer,
  momentum,
  nesterov,
  optimizerIds,
  quadratic,
  rmsprop,
  run,
  saddle,
  sgd,
  type GradFn,
} from "../src/index.ts";

const constGrad =
  (g: [number, number]): GradFn =>
  () =>
    g;

describe("single steps against hand-computed values", () => {
  it("SGD: θ − ηg", () => {
    const s = sgd({ lr: 0.1 }).step(initState([1, 2]), constGrad([3, -4]));
    expect(s.params[0]).toBeCloseTo(0.7, 12);
    expect(s.params[1]).toBeCloseTo(2.4, 12);
    expect(s.t).toBe(1);
  });

  it("momentum accumulates velocity across steps", () => {
    const opt = momentum({ lr: 0.1, beta: 0.9 });
    const g = constGrad([1, 0]);
    const s1 = opt.step(initState([0, 0]), g);
    expect(s1.m[0]).toBeCloseTo(-0.1, 12);
    expect(s1.params[0]).toBeCloseTo(-0.1, 12);
    const s2 = opt.step(s1, g);
    // v = 0.9·(−0.1) − 0.1 = −0.19, θ = −0.1 − 0.19
    expect(s2.m[0]).toBeCloseTo(-0.19, 12);
    expect(s2.params[0]).toBeCloseTo(-0.29, 12);
  });

  it("Nesterov evaluates the gradient at the look-ahead point θ + βv", () => {
    const seen: [number, number][] = [];
    const grad: GradFn = (p) => {
      seen.push([p[0], p[1]]);
      return [p[0], p[1]];
    };
    const opt = nesterov({ lr: 0.1, beta: 0.5 });
    const s0 = { ...initState([1, 1]), m: [0.2, -0.2] as [number, number] };
    const s1 = opt.step(s0, grad);
    expect(seen[0]![0]).toBeCloseTo(1.1, 12);
    expect(seen[0]![1]).toBeCloseTo(0.9, 12);
    // v = 0.5·0.2 − 0.1·1.1 = −0.01 ; θ = 1 − 0.01
    expect(s1.m[0]).toBeCloseTo(-0.01, 12);
    expect(s1.params[0]).toBeCloseTo(0.99, 12);
  });

  it("RMSProp's first step is η·g/(√((1−ρ)g²)+ε)", () => {
    const s = rmsprop({ lr: 0.01, rho: 0.9, eps: 0 }).step(initState([0, 0]), constGrad([2, -0.5]));
    // √(0.1·g²) = |g|·√0.1, so the step is η·sign(g)/√0.1 on each coordinate
    expect(s.params[0]).toBeCloseTo(-0.01 / Math.sqrt(0.1), 12);
    expect(s.params[1]).toBeCloseTo(0.01 / Math.sqrt(0.1), 12);
  });

  it("Adam's bias-corrected first step is η·sign(g) regardless of |g|", () => {
    const opt = adam({ lr: 0.1, eps: 0 });
    const s = opt.step(initState([0, 0]), constGrad([1000, -0.001]));
    expect(s.params[0]).toBeCloseTo(-0.1, 12);
    expect(s.params[1]).toBeCloseTo(0.1, 12);
    expect(s.m[0]).toBeCloseTo(100, 9); // (1 − β₁)g
    expect(s.v[0]).toBeCloseTo(1000, 6); // (1 − β₂)g²
  });

  it("Adam's second step matches the formula worked by hand", () => {
    const opt = adam({ lr: 0.1, beta1: 0.9, beta2: 0.999, eps: 1e-8 });
    const s1 = opt.step(initState([1, 1]), constGrad([2, 0]));
    const s2 = opt.step(s1, constGrad([-1, 0]));
    // m₁ = 0.2, v₁ = 0.004 ; m₂ = 0.9·0.2 + 0.1·(−1) = 0.08 ; v₂ = 0.999·0.004 + 0.001·1 = 0.004996
    // m̂₂ = 0.08 / 0.19 ; v̂₂ = 0.004996 / 0.001999
    const mHat = 0.08 / (1 - 0.9 ** 2);
    const vHat = 0.004996 / (1 - 0.999 ** 2);
    expect(s2.m[0]).toBeCloseTo(0.08, 12);
    expect(s2.v[0]).toBeCloseTo(0.004996, 12);
    // step 1: m̂₁ = 2, √v̂₁ = 2, so θ₁ = 1 − 0.1·2/(2 + ε)
    const theta1 = 1 - (0.1 * 2) / (2 + 1e-8);
    expect(s1.params[0]).toBeCloseTo(theta1, 14);
    expect(s2.params[0]).toBeCloseTo(theta1 - (0.1 * mHat) / (Math.sqrt(vHat) + 1e-8), 12);
  });

  it("without bias correction Adam's first step would be (1−β₁)/√(1−β₂) ≈ 3.16·η; with it, exactly η", () => {
    const g = 0.5;
    const raw = (0.1 * ((1 - 0.9) * g)) / Math.sqrt((1 - 0.999) * g * g);
    expect(raw / 0.1).toBeCloseTo(0.1 / Math.sqrt(0.001), 12);
    expect(raw / 0.1).toBeGreaterThan(3.16);
    const s = adam({ lr: 0.1, eps: 0 }).step(initState([0, 0]), constGrad([g, g]));
    expect(-s.params[0]).toBeCloseTo(0.1, 12);
  });

  it("steps are pure: the input state is not mutated", () => {
    for (const id of optimizerIds) {
      const s0 = initState([1, 2]);
      const copy = structuredClone(s0);
      makeOptimizer(id, 0.1).step(s0, constGrad([1, 1]));
      expect(s0).toEqual(copy);
    }
  });
});

describe("convergence on the convex quadratic", () => {
  it.each(optimizerIds)("%s reaches f < 1e-6 from (−4, 2)", (id) => {
    const r = run(quadratic, makeOptimizer(id, 0.05), { steps: 3000, tol: 1e-6 });
    expect(r.diverged).toBe(false);
    expect(r.stepsToTol).not.toBeNull();
    expect(r.finalLoss).toBeLessThan(1e-6);
  });

  it("momentum beats plain GD on the ill-conditioned bowl at the same η", () => {
    const gd = run(quadratic, sgd({ lr: 0.02 }), { steps: 5000 });
    const hb = run(quadratic, momentum({ lr: 0.02 }), { steps: 5000 });
    expect(hb.stepsToTol!).toBeLessThan(gd.stepsToTol!);
  });

  it("GD diverges once η exceeds 2/λ_max = 0.2", () => {
    expect(run(quadratic, sgd({ lr: 0.21 }), { steps: 2000 }).diverged).toBe(true);
    expect(run(quadratic, sgd({ lr: 0.19 }), { steps: 2000 }).diverged).toBe(false);
  });
});

describe("the saddle point x² − y²", () => {
  it("GD started exactly on the x-axis stalls at the saddle", () => {
    const r = run(saddle, sgd({ lr: 0.01 }), { steps: 2000, start: [1, 0] });
    const [x, y] = r.final;
    expect(Math.abs(x)).toBeLessThan(1e-12);
    expect(y).toBe(0);
    const [gx, gy] = saddle.grad(x, y);
    expect(Math.hypot(gx, gy)).toBeLessThan(1e-12); // looks converged, but it is not a minimum
  });

  it("from a 1e-3 nudge off the axis, momentum escapes in 100 steps while GD barely moves", () => {
    const start: [number, number] = [1, 1e-3];
    const gd = run(saddle, sgd({ lr: 0.01 }), { steps: 100, start });
    const hb = run(saddle, momentum({ lr: 0.01, beta: 0.9 }), { steps: 100, start });
    // GD's y grows by (1 + 2η) per step: 1e-3 · 1.02¹⁰⁰ ≈ 7.2e-3
    expect(Math.abs(gd.final[1])).toBeCloseTo(1e-3 * 1.02 ** 100, 12);
    expect(Math.abs(gd.final[1])).toBeLessThan(0.01);
    expect(Math.abs(hb.final[1])).toBeGreaterThan(1);
    expect(hb.finalLoss).toBeLessThan(-1);
  });

  it("Adam escapes too, because it normalises the tiny y-gradient", () => {
    const r = run(saddle, adam({ lr: 0.01 }), { steps: 100, start: [1, 1e-3] });
    expect(Math.abs(r.final[1])).toBeGreaterThan(1);
  });
});
