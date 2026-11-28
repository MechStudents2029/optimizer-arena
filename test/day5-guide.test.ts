import { describe, expect, it } from "vitest";
import { booth, makeSchedule, race, run, sgd, surfaces } from "../src/index.ts";

// Claims in docs/which-optimizer.md that are cheap enough to pin as tests.

describe("Booth's stability limit is 2/λ_max = 1/9 (λ_max = 18)", () => {
  it("constant-rate GD converges just below 1/9 and diverges just above it", () => {
    const below = run(booth, sgd({ lr: 0.105 }), { steps: 5000 });
    const above = run(booth, sgd({ lr: 0.115 }), { steps: 5000 });
    expect(below.stepsToTol).not.toBeNull();
    expect(below.diverged).toBe(false);
    expect(above.diverged).toBe(true);
  });

  it("warmup + cosine at a peak η = 1 converges in 20 steps, all taken below 1/9", () => {
    const [e] = race(booth, { steps: 10_000, schedule: "warmup-cosine", optimizers: ["sgd"], lrGrid: [1] });
    expect(e!.steps).toBe(20);
    const s = makeSchedule("warmup-cosine", 10_000);
    // step k uses s(k − 1); the fastest of the 20 steps is the last one
    expect(s(19)).toBeLessThan(1 / 9);
    expect(s(19)).toBeCloseTo(0.04, 12);
  });
});

describe("schedules against a constant rate", () => {
  it("with step decay RMSProp reaches the goal on all eight surfaces; at a constant rate on four", () => {
    const reached = (schedule: "step" | "constant") =>
      surfaces.filter((s) => race(s, { steps: 10_000, schedule, optimizers: ["rmsprop"] })[0]!.steps !== null).length;
    expect(reached("step")).toBe(8);
    expect(reached("constant")).toBe(4);
  });

  it("AdamW on Booth gets slower as λ grows: 93, 705, 4616, 8910 steps for λ = 0, 1e-4, 1e-3, 1e-2", () => {
    const steps = [0, 1e-4, 1e-3, 1e-2].map(
      (weightDecay) =>
        race(booth, { steps: 10_000, optimizers: ["adamw"], lrGrid: [0.3], hyperparams: { weightDecay } })[0]!.steps,
    );
    expect(steps).toEqual([93, 705, 4616, 8910]);
  });
});
