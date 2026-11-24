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
