import { describe, expect, it } from "vitest";
import {
  constant,
  cosine,
  makeSchedule,
  quadratic,
  race,
  run,
  scheduleIds,
  sgd,
  stepDecay,
  warmupCosine,
} from "../src/index.ts";

describe("schedule shapes", () => {
  it("constant is 1 everywhere", () => {
    for (const t of [0, 1, 1e6]) expect(constant(t)).toBe(1);
  });

  it("step decay halves at every boundary and not before", () => {
    const s = stepDecay(10, 0.5);
    expect(s(0)).toBe(1);
    expect(s(9)).toBe(1);
    expect(s(10)).toBe(0.5);
    expect(s(19)).toBe(0.5);
    expect(s(20)).toBe(0.25);
    expect(s(35)).toBe(0.125);
  });

  it("cosine runs 1 → ½ → minRatio and stays there past T", () => {
    const s = cosine(100);
    expect(s(0)).toBe(1);
    expect(s(50)).toBeCloseTo(0.5, 14);
    expect(s(25)).toBeCloseTo(0.5 * (1 + Math.SQRT1_2), 14);
    expect(s(100)).toBeCloseTo(0, 14);
    expect(s(500)).toBeCloseTo(0, 14);
    const floor = cosine(100, 0.1);
    expect(floor(100)).toBeCloseTo(0.1, 14);
    expect(floor(50)).toBeCloseTo(0.55, 14);
  });

  it("cosine never increases", () => {
    const s = cosine(1000);
    for (let t = 1; t <= 1000; t++) expect(s(t)).toBeLessThanOrEqual(s(t - 1));
  });

  it("warmup ramps (t + 1)/W up to 1, then follows a cosine over T − W", () => {
    const s = warmupCosine(110, 10);
    expect(s(0)).toBeCloseTo(0.1, 14);
    expect(s(4)).toBeCloseTo(0.5, 14);
    expect(s(9)).toBe(1);
    expect(s(10)).toBe(1);
    expect(s(60)).toBeCloseTo(0.5, 14);
    expect(s(110)).toBeCloseTo(0, 14);
  });

  it("rejects lengths that make no sense", () => {
    expect(() => stepDecay(0)).toThrow(RangeError);
    expect(() => cosine(0)).toThrow(RangeError);
    expect(() => warmupCosine(10, 0)).toThrow(RangeError);
    expect(() => warmupCosine(10, 10)).toThrow(RangeError);
  });
});

describe("makeSchedule", () => {
  it("builds every id, each starting at or ramping to 1", () => {
    for (const id of scheduleIds) {
      const s = makeSchedule(id, 1000);
      const peak = Math.max(...Array.from({ length: 1000 }, (_, t) => s(t)));
      expect(peak).toBe(1);
    }
  });

  it("defaults: step halves every tenth of the budget, warmup is 5% of it", () => {
    const st = makeSchedule("step", 1000);
    expect(st(99)).toBe(1);
    expect(st(100)).toBe(0.5);
    const wc = makeSchedule("warmup-cosine", 1000);
    expect(wc(0)).toBeCloseTo(1 / 50, 14);
    expect(wc(49)).toBe(1);
  });

  it("passes options through", () => {
    expect(makeSchedule("step", 1000, { every: 5, gamma: 0.1 })(5)).toBeCloseTo(0.1, 14);
    expect(makeSchedule("cosine", 1000, { minRatio: 0.2 })(1000)).toBeCloseTo(0.2, 14);
    expect(makeSchedule("warmup-cosine", 1000, { warmup: 4 })(0)).toBe(0.25);
  });
});

describe("schedules inside an optimizer", () => {
  it("the first step uses s(0) and the k-th uses s(k − 1)", () => {
    const opt = sgd({ lr: 1, schedule: stepDecay(2, 0.5) });
    const r = run(quadratic, opt, { steps: 4, tol: 0, start: [1, 0] });
    // f = ½x² in x, so θ ← θ − η_t·θ : η = 1, 1, 0.5, 0.5
    expect(r.points.map((p) => p[0])).toEqual([1, 0, 0, 0, 0]);
    const r2 = run(quadratic, sgd({ lr: 0.5, schedule: stepDecay(2, 0.5) }), { steps: 4, tol: 0, start: [1, 0] });
    expect(r2.points.map((p) => p[0])).toEqual([1, 0.5, 0.25, 0.25 * 0.75, 0.25 * 0.75 * 0.75]);
  });

  it("race() stretches the schedule over its step budget", () => {
    const flat = race(quadratic, { steps: 500, lrGrid: [0.05], optimizers: ["sgd"] })[0]!;
    const cos = race(quadratic, { steps: 500, lrGrid: [0.05], optimizers: ["sgd"], schedule: "cosine" })[0]!;
    // decaying η slows GD down on a bowl where the constant rate was already stable
    expect(cos.steps === null || cos.steps > flat.steps!).toBe(true);
  });

  it("race() can be limited to a subset of optimizers, in the order given", () => {
    const entries = race(quadratic, { steps: 200, optimizers: ["lion", "adagrad"] });
    expect(entries.map((e) => e.optimizer)).toEqual(["lion", "adagrad"]);
  });
});
