import { describe, expect, it } from "vitest";
import { LR_GRID, optimizerIds, quadratic, race, rosenbrock, saddle } from "../src/index.ts";

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
