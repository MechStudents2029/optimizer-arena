import { describe, expect, it } from "vitest";
import { ackley, booth, LR_GRID, makeOptimizer, optimizerIds, quadratic, race, rastrigin, rosenbrock, run, saddle } from "../src/index.ts";

describe("race", () => {
  it("returns one entry per optimizer, each with a learning rate from the grid", () => {
    const entries = race(quadratic, { steps: 2000 });
    expect(entries.map((e) => e.optimizer)).toEqual(optimizerIds);
    for (const e of entries) {
      expect(LR_GRID).toContain(e.lr);
      expect(e.steps).not.toBeNull();
      expect(e.finalLoss).toBeLessThan(1e-6);
    }
  });

  it("is deterministic", () => {
    const strip = (es: ReturnType<typeof race>) => es.map(({ trajectory, ...e }) => e);
    expect(strip(race(rosenbrock, { steps: 2000 }))).toEqual(strip(race(rosenbrock, { steps: 2000 })));
  });

  it("skips learning rates that diverge", () => {
    const [gd] = race(rosenbrock, { steps: 2000, lrGrid: [1, 1e-3] });
    expect(gd!.lr).toBe(1e-3);
    const [none] = race(rosenbrock, { steps: 200, lrGrid: [1] });
    expect(none!.lr).toBeNull();
  });

  it("on the saddle at a shared η, every momentum or adaptive method escapes before GD", () => {
    const entries = race(saddle, { steps: 2000, lrGrid: [0.01] });
    const gd = entries.find((e) => e.optimizer === "sgd")!;
    expect(gd.steps).not.toBeNull();
    for (const e of entries.filter((e) => e.optimizer !== "sgd")) {
      expect(e.steps!).toBeLessThan(gd.steps!);
    }
  });

  it("honours a custom start point", () => {
    const entries = race(quadratic, { steps: 500, start: [1, 1] });
    for (const e of entries) expect(e.trajectory!.points[0]).toEqual([1, 1]);
  });
});

describe("race with hyperparameters", () => {
  it("forwards hyperparameters to every run", () => {
    const strip = (es: ReturnType<typeof race>) => es.map(({ trajectory, ...e }) => e);
    const base = race(quadratic, { steps: 2000 });
    const sameAsDefault = race(quadratic, { steps: 2000, hyperparams: { beta: 0.9, rho: 0.9 } });
    expect(strip(sameAsDefault)).toEqual(strip(base));
    const lowBeta = race(quadratic, { steps: 2000, hyperparams: { beta: 0 } });
    const hb = lowBeta.find((e) => e.optimizer === "momentum")!;
    const gd = lowBeta.find((e) => e.optimizer === "sgd")!;
    // with β = 0 heavy ball is GD, so it picks the same η and takes the same number of steps
    expect(hb.lr).toBe(gd.lr);
    expect(hb.steps).toBe(gd.steps);
  });
});

describe("the Day 2 surfaces in a race", () => {
  it("Booth: every optimizer reaches f < 1e-6 and ends next to (1, 3)", () => {
    for (const e of race(booth, { steps: 2000 })) {
      expect(e.steps).not.toBeNull();
      const [x, y] = e.trajectory!.final;
      expect(Math.hypot(x - 1, y - 3)).toBeLessThan(1e-3);
    }
  });

  it("Rastrigin: from (2.2, −1.7) every optimizer is caught by the local minimum near (2, −2)", () => {
    for (const e of race(rastrigin, { steps: 10_000 })) {
      expect(e.steps).toBeNull();
      const [x, y] = e.trajectory!.final;
      // the dimple's centre is (±1.98991…), pulled toward the origin by the x² + y² bowl
      expect(Math.hypot(x - 1.98991, y + 1.98991)).toBeLessThan(1e-3);
      // f ≈ 7.9597 there, against a global minimum of 0
      expect(e.finalLoss).toBeCloseTo(7.9597, 4);
    }
  });

  it("Rastrigin: started inside the central basin, plain GD does find the global minimum", () => {
    const [gd] = race(rastrigin, { steps: 2000, start: [0.3, -0.4] });
    expect(gd!.steps).not.toBeNull();
  });

  it("Ackley: a small-η method settles in a ripple next to the start, far from the funnel", () => {
    const t = run(ackley, makeOptimizer("sgd", 1e-3), { steps: 5000 });
    const [x, y] = t.final;
    expect(Math.hypot(x, y)).toBeGreaterThan(3);
    expect(Math.hypot(...ackley.grad(x, y))).toBeLessThan(1e-3);
    expect(t.finalLoss).toBeGreaterThan(8);
  });
});
