import { describe, expect, it } from "vitest";
import {
  HEIGHT,
  apply,
  buildMesh,
  chooseHeightMode,
  clampCamera,
  heightMap,
  lambert,
  matMul,
  painterOrder,
  project,
  quadNormal,
  rotX,
  rotZ,
  toWorldXY,
  viewMatrix,
  type Camera,
  type Mat3,
  type Vec3,
} from "../src/view3d.ts";
import { surfaceById } from "../src/index.ts";

const close = (a: Vec3, b: Vec3) => a.forEach((v, i) => expect(v).toBeCloseTo(b[i]!, 12));
const cam = (over: Partial<Camera> = {}): Camera => ({ yaw: 0, pitch: 0, distance: 5, focal: 100, cx: 0, cy: 0, ...over });

describe("rotation matrices", () => {
  it("rotZ(90°) sends x to y and y to −x; rotX(90°) sends y to z", () => {
    close(apply(rotZ(Math.PI / 2), [1, 0, 0]), [0, 1, 0]);
    close(apply(rotZ(Math.PI / 2), [0, 1, 0]), [-1, 0, 0]);
    close(apply(rotX(Math.PI / 2), [0, 1, 0]), [0, 0, 1]);
  });

  it("are orthogonal: RᵀR = I and lengths are preserved", () => {
    const r = matMul(rotX(0.7), rotZ(-1.3));
    const rt = [0, 1, 2].map((i) => [0, 1, 2].map((j) => r[j]![i]!)) as Mat3;
    const id = matMul(rt, r);
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) expect(id[i]![j]).toBeCloseTo(i === j ? 1 : 0, 12);
    const p: Vec3 = [0.3, -2, 1.5];
    const q = apply(r, p);
    expect(Math.hypot(...q)).toBeCloseTo(Math.hypot(...p), 12);
  });

  it("matMul composes: (AB)p = A(Bp)", () => {
    const a = rotX(0.4);
    const b = rotZ(1.1);
    const p: Vec3 = [1, 2, 3];
    close(apply(matMul(a, b), p), apply(a, apply(b, p)));
  });

  it("viewMatrix spins about z first, then tilts about x", () => {
    close(apply(viewMatrix({ yaw: Math.PI / 2, pitch: -Math.PI / 2 }), [1, 0, 0]), [0, 0, -1]);
  });
});

describe("project", () => {
  it("looking straight down, the scene centre lands on the screen centre", () => {
    const p = project([0, 0, HEIGHT / 2], cam({ cx: 320, cy: 240 }));
    expect(p.x).toBeCloseTo(320, 12);
    expect(p.y).toBeCloseTo(240, 12);
    expect(p.depth).toBeCloseTo(5, 12);
  });

  it("applies the perspective divide: x · focal / depth, with screen y flipped", () => {
    // From above, (1, 1, HEIGHT/2) sits at view (1, 1, 0), depth 5 → (100/5, −100/5).
    const p = project([1, 1, HEIGHT / 2], cam());
    expect(p.x).toBeCloseTo(20, 12);
    expect(p.y).toBeCloseTo(-20, 12);
  });

  it("nearer points are bigger: raising a point towards the camera spreads it out", () => {
    const low = project([1, 0, 0], cam());
    const high = project([1, 0, HEIGHT], cam());
    expect(high.depth).toBeLessThan(low.depth);
    expect(high.x).toBeGreaterThan(low.x);
    expect(high.x / low.x).toBeCloseTo((5 + HEIGHT / 2) / (5 - HEIGHT / 2), 12);
  });

  it("tilting the camera pushes the far edge (y = +1) away and makes height point up the screen", () => {
    const c = cam({ pitch: -1 });
    expect(project([0, 1, 0], c).depth).toBeGreaterThan(project([0, -1, 0], c).depth);
    expect(project([0, 0, HEIGHT], c).y).toBeLessThan(project([0, 0, 0], c).y);
  });
});

describe("height normalisation", () => {
  it("linear maps [lo, hi] onto [0, 1] and clamps outside", () => {
    const h = heightMap(2, 12, "linear");
    expect(h(2)).toBe(0);
    expect(h(7)).toBe(0.5);
    expect(h(12)).toBe(1);
    expect(h(-5)).toBe(0);
    expect(h(1e9)).toBe(1);
    expect(h(Infinity)).toBe(1);
  });

  it("log uses log(1 + f − lo), so equal ratios get equal steps", () => {
    const h = heightMap(0, 1e4 - 1, "log");
    expect(h(0)).toBe(0);
    expect(h(1e4 - 1)).toBeCloseTo(1, 12);
    expect(h(99)).toBeCloseTo(0.5, 12); // log(100) / log(10 000)
  });

  it("chooses log for Rosenbrock and Beale and linear for the bowl, Rastrigin and Ackley", () => {
    const mode = (id: string) => buildMesh(surfaceById(id), 48).mode;
    expect(mode("rosenbrock")).toBe("log");
    expect(mode("beale")).toBe("log");
    expect(mode("quadratic")).toBe("linear");
    expect(mode("rastrigin")).toBe("linear");
    expect(mode("ackley")).toBe("linear");
    expect(chooseHeightMode([])).toBe("linear");
  });
});

describe("buildMesh", () => {
  const mesh = buildMesh(surfaceById("rosenbrock"), 10);

  it("has (n + 1)² vertices and n² quads", () => {
    expect(mesh.vertices).toHaveLength(121);
    expect(mesh.values).toHaveLength(121);
    expect(mesh.quads).toHaveLength(100);
  });

  it("spans the floor [−1, 1]² with heights in [0, HEIGHT], hitting both ends", () => {
    const xs = mesh.vertices.map((v) => v[0]);
    const zs = mesh.vertices.map((v) => v[2]);
    expect(Math.min(...xs)).toBe(-1);
    expect(Math.max(...xs)).toBe(1);
    expect(Math.min(...zs)).toBe(0);
    expect(Math.max(...zs)).toBeCloseTo(HEIGHT, 12);
  });

  it("puts each vertex at its surface value", () => {
    const s = surfaceById("rosenbrock");
    const [x0, x1, y0, y1] = s.bounds;
    const k = 3 * 11 + 7; // row j = 3, column i = 7
    expect(mesh.values[k]).toBeCloseTo(s.f(x0 + ((x1 - x0) * 7) / 10, y0 + ((y1 - y0) * 3) / 10), 12);
    expect(mesh.vertices[k]![0]).toBeCloseTo(toWorldXY(s.bounds, x0 + ((x1 - x0) * 7) / 10, 0)[0], 12);
    expect(toWorldXY(s.bounds, x0, y1)).toEqual([-1, 1]);
  });
});

describe("shading and painter's order", () => {
  it("a flat quad faces straight up and is fully lit from above", () => {
    const n = quadNormal([0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]);
    close(n, [0, 0, 1]);
    expect(lambert(n, [0, 0, 1])).toBe(1);
    expect(lambert(n, [1, 0, 0])).toBe(0.35);
  });

  it("sorts far to near, keeping ties in order", () => {
    expect(painterOrder([3, 9, 1, 9, 5])).toEqual([1, 3, 4, 0, 2]);
  });

  it("from a tilted camera, mesh quads on the far edge are drawn before near ones", () => {
    const m = buildMesh(surfaceById("quadratic"), 4);
    const c = cam({ pitch: -1 });
    const depths = m.quads.map((q) => q.reduce((s, i) => s + project(m.vertices[i]!, c).depth, 0) / 4);
    const order = painterOrder(depths);
    // Quads are row-major in y, so the far row (j = 3) is 12…15 and the near row (j = 0) is 0…3.
    expect(order[0]).toBeGreaterThanOrEqual(12);
    expect(order[order.length - 1]).toBeLessThan(4);
  });

  it("clampCamera keeps pitch between side-on and top-down and bounds the zoom", () => {
    const c = clampCamera(cam({ pitch: -3, distance: 100 }));
    expect(c.pitch).toBeGreaterThan(-Math.PI / 2);
    expect(c.distance).toBe(12);
    expect(clampCamera(cam({ pitch: 1, distance: 0 }))).toMatchObject({ pitch: 0, distance: 2.5 });
  });
});
