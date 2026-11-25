import { describe, expect, it } from "vitest";
import { surfaces, surfaceById, type Surface } from "../src/index.ts";

/** Central finite difference of f at (x, y). */
function numericGrad(s: Surface, x: number, y: number, h = 1e-5): [number, number] {
  return [
    (s.f(x + h, y) - s.f(x - h, y)) / (2 * h),
    (s.f(x, y + h) - s.f(x, y - h)) / (2 * h),
  ];
}

/** Deterministic sample points inside each surface's plot window. */
function samplePoints(s: Surface, n = 25): [number, number][] {
  const [x0, x1, y0, y1] = s.bounds;
  const pts: [number, number][] = [s.start];
  let seed = 12345;
  const rand = () => ((seed = (seed * 1103515245 + 12345) % 2 ** 31) / 2 ** 31);
  for (let i = 0; i < n; i++) pts.push([x0 + (x1 - x0) * rand(), y0 + (y1 - y0) * rand()]);
  return pts;
}

describe.each(surfaces.map((s) => [s.id, s] as const))("%s", (_id, s) => {
  it("analytic gradient matches central finite differences", () => {
    for (const [x, y] of samplePoints(s)) {
      const [gx, gy] = s.grad(x, y);
      const [nx, ny] = numericGrad(s, x, y);
      const scale = 1 + Math.abs(nx) + Math.abs(ny);
      expect(Math.abs(gx - nx) / scale).toBeLessThan(1e-6);
      expect(Math.abs(gy - ny) / scale).toBeLessThan(1e-6);
    }
  });

  it("known minima are stationary points with f = f*", () => {
    for (const [x, y] of s.minima) {
      expect(s.f(x, y)).toBeCloseTo(s.minValue, 10);
      const [gx, gy] = s.grad(x, y);
      expect(Math.hypot(gx, gy)).toBeLessThan(1e-9);
    }
  });

  it("no sampled point beats the known minimum", () => {
    for (const [x, y] of samplePoints(s, 200)) {
      if (s.minima.length) expect(s.f(x, y)).toBeGreaterThanOrEqual(s.minValue);
    }
  });

  it("start point lies inside the plot window", () => {
    const [x0, x1, y0, y1] = s.bounds;
    expect(s.start[0]).toBeGreaterThanOrEqual(x0);
    expect(s.start[0]).toBeLessThanOrEqual(x1);
    expect(s.start[1]).toBeGreaterThanOrEqual(y0);
    expect(s.start[1]).toBeLessThanOrEqual(y1);
  });
});

describe("saddle", () => {
  const s = surfaceById("saddle");
  it("origin is stationary but not a minimum", () => {
    expect(s.grad(0, 0)).toEqual([0, -0]);
    expect(s.f(0, 0.1)).toBeLessThan(s.f(0, 0));
    expect(s.f(0.1, 0)).toBeGreaterThan(s.f(0, 0));
    expect(s.minValue).toBe(-Infinity);
  });
});

describe("surfaceById", () => {
  it("throws on an unknown id", () => {
    expect(() => surfaceById("nope")).toThrow(/unknown surface/);
  });
});

describe("rastrigin", () => {
  const s = surfaceById("rastrigin");
  it("has a local minimum in every integer cell, slightly pulled toward the origin", () => {
    // ∂f/∂x = 2x + 20π sin 2πx vanishes just inside each integer, where f is a local min.
    for (const k of [-3, -2, -1, 1, 2, 3]) {
      // Newton on the 1D derivative, starting at the integer
      let x = k;
      for (let i = 0; i < 20; i++) {
        const d = 2 * x + 20 * Math.PI * Math.sin(2 * Math.PI * x);
        const dd = 2 + 40 * Math.PI ** 2 * Math.cos(2 * Math.PI * x);
        x -= d / dd;
      }
      expect(Math.abs(x)).toBeLessThan(Math.abs(k));
      expect(Math.abs(x - k)).toBeLessThan(0.02);
      const [gx] = s.grad(x, 0);
      expect(Math.abs(gx)).toBeLessThan(1e-9);
      expect(s.f(x, 0)).toBeGreaterThan(s.f(0, 0));
      expect(s.f(x + 0.1, 0)).toBeGreaterThan(s.f(x, 0));
      expect(s.f(x - 0.1, 0)).toBeGreaterThan(s.f(x, 0));
    }
  });
});

describe("ackley", () => {
  const s = surfaceById("ackley");
  it("is exactly zero with a zero (sub)gradient at the origin", () => {
    expect(Math.abs(s.f(0, 0))).toBeLessThan(1e-14);
    expect(s.grad(0, 0)).toEqual([0, 0]);
  });
  it("the funnel's slope fades with distance: at (4, 4) it is half what it is at (0.5, 0.5)", () => {
    // At these points sin 2πx = sin 2πy = 0, so only the envelope 2e^{−0.2r}/r·(x, y) is left:
    // |∇f| ≈ 2.56 at (0.5, 0.5) and ≈ 1.27 at (4, 4), with r = √(½(x² + y²)).
    const near = Math.hypot(...s.grad(0.5, 0.5));
    const [gx, gy] = s.grad(4, 4);
    expect(near).toBeCloseTo(2 * Math.SQRT2 * Math.exp(-0.1), 9);
    expect(Math.hypot(gx, gy)).toBeLessThan(1.5);
    expect(near).toBeGreaterThan(Math.hypot(gx, gy));
  });
});

describe("booth", () => {
  const s = surfaceById("booth");
  it("is the quadratic with Hessian [[10, 8], [8, 10]] (eigenvalues 18 and 2)", () => {
    const h = 1e-4;
    const [gx0, gy0] = s.grad(0, 0);
    const [gx1, gy1] = s.grad(h, 0);
    const [gx2, gy2] = s.grad(0, h);
    expect((gx1 - gx0) / h).toBeCloseTo(10, 6);
    expect((gy1 - gy0) / h).toBeCloseTo(8, 6);
    expect((gx2 - gx0) / h).toBeCloseTo(8, 6);
    expect((gy2 - gy0) / h).toBeCloseTo(10, 6);
    // along (1, 1) and (1, −1)
    expect(s.f(1 + 1, 3 + 1) - s.f(1, 3)).toBeCloseTo(0.5 * 18 * 2, 9);
    expect(s.f(1 + 1, 3 - 1) - s.f(1, 3)).toBeCloseTo(0.5 * 2 * 2, 9);
  });
});
