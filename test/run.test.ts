import { describe, expect, it } from "vitest";
import { himmelblau, nearestMinimum, quadratic, rosenbrock, run, saddle, sgd, adam } from "../src/index.ts";

describe("run", () => {
  it("records the start point and one point per step", () => {
    const r = run(quadratic, sgd({ lr: 0.01 }), { steps: 10, tol: 0 });
    expect(r.points).toHaveLength(11);
    expect(r.losses).toHaveLength(11);
    expect(r.points[0]).toEqual(quadratic.start);
    expect(r.losses[0]).toBeCloseTo(quadratic.f(...quadratic.start), 12);
    expect(r.stepsToTol).toBeNull();
  });

  it("stops at tolerance unless asked to run past it", () => {
    const a = run(quadratic, sgd({ lr: 0.1 }), { steps: 1000, tol: 1e-3 });
    expect(a.points).toHaveLength(a.stepsToTol! + 1);
    const b = run(quadratic, sgd({ lr: 0.1 }), { steps: 1000, tol: 1e-3, runPastTol: true });
    expect(b.points).toHaveLength(1001);
    expect(b.stepsToTol).toBe(a.stepsToTol);
  });

  it("flags divergence and keeps only finite points", () => {
    const r = run(rosenbrock, sgd({ lr: 0.1 }), { steps: 100 });
    expect(r.diverged).toBe(true);
    expect(r.losses.every(Number.isFinite)).toBe(true);
  });

  it("never reports a tolerance hit on a surface with no minimum", () => {
    const r = run(saddle, adam({ lr: 0.05 }), { steps: 200 });
    expect(r.stepsToTol).toBeNull();
  });

  it("GD on Himmelblau settles in one of the four minima", () => {
    const r = run(himmelblau, sgd({ lr: 0.01 }), { steps: 5000 });
    expect(r.stepsToTol).not.toBeNull();
    const i = nearestMinimum(himmelblau, r.final);
    const [mx, my] = himmelblau.minima[i]!;
    expect(Math.hypot(r.final[0] - mx, r.final[1] - my)).toBeLessThan(1e-3);
  });
});
