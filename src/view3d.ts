import type { Surface } from "./surfaces.ts";

/**
 * Pure math for the 3D surface view: rotation matrices, a perspective
 * projection, a height map, the surface mesh, and painter's-algorithm ordering.
 * Nothing here touches the DOM, so all of it is unit-tested.
 *
 * World coordinates: x and y are the surface's plot window rescaled to
 * [−1, 1] × [−1, 1], and z is the normalised height in [0, HEIGHT].
 */

export type Vec3 = [number, number, number];
/** A 3 × 3 matrix stored row-major: [[a, b, c], [d, e, f], [g, h, i]]. */
export type Mat3 = [Vec3, Vec3, Vec3];

/** Height of the tallest point of the mesh in world units (the floor is 2 × 2). */
export const HEIGHT = 0.9;

/** Rotation by `a` radians about the x-axis (y turns towards z). */
export function rotX(a: number): Mat3 {
  const c = Math.cos(a);
  const s = Math.sin(a);
  return [
    [1, 0, 0],
    [0, c, -s],
    [0, s, c],
  ];
}

/** Rotation by `a` radians about the z-axis (x turns towards y). */
export function rotZ(a: number): Mat3 {
  const c = Math.cos(a);
  const s = Math.sin(a);
  return [
    [c, -s, 0],
    [s, c, 0],
    [0, 0, 1],
  ];
}

export function matMul(a: Mat3, b: Mat3): Mat3 {
  // Row i of AB is row i of A applied to B's columns: (AB)ᵢⱼ = Σₖ aᵢₖ bₖⱼ.
  const row = (r: Vec3): Vec3 => [
    r[0] * b[0][0] + r[1] * b[1][0] + r[2] * b[2][0],
    r[0] * b[0][1] + r[1] * b[1][1] + r[2] * b[2][1],
    r[0] * b[0][2] + r[1] * b[1][2] + r[2] * b[2][2],
  ];
  return [row(a[0]), row(a[1]), row(a[2])];
}

export function apply(m: Mat3, p: Vec3): Vec3 {
  return [
    m[0][0] * p[0] + m[0][1] * p[1] + m[0][2] * p[2],
    m[1][0] * p[0] + m[1][1] * p[1] + m[1][2] * p[2],
    m[2][0] * p[0] + m[2][1] * p[1] + m[2][2] * p[2],
  ];
}

export interface Camera {
  /** Spin about the vertical (z) axis, radians. */
  yaw: number;
  /** Tilt about the screen's horizontal axis, radians: 0 looks straight down, −π/2 looks from the side. */
  pitch: number;
  /** Distance from the camera to the centre of the scene, in world units. */
  distance: number;
  /** Focal length in pixels: the screen size of one world unit at depth `distance`. */
  focal: number;
  /** Screen centre in pixels. */
  cx: number;
  cy: number;
}

export const PITCH_MIN = -Math.PI / 2 + 0.05;
export const PITCH_MAX = 0;
export const DISTANCE_MIN = 2.5;
export const DISTANCE_MAX = 12;

/** Keep a camera in the range where the view stays sensible (never under the floor, never inside the mesh). */
export function clampCamera(cam: Camera): Camera {
  return {
    ...cam,
    pitch: Math.min(PITCH_MAX, Math.max(PITCH_MIN, cam.pitch)),
    distance: Math.min(DISTANCE_MAX, Math.max(DISTANCE_MIN, cam.distance)),
  };
}

/** World → view rotation: spin about z first, then tilt about x. */
export function viewMatrix(cam: Pick<Camera, "yaw" | "pitch">): Mat3 {
  return matMul(rotX(cam.pitch), rotZ(cam.yaw));
}

export interface Projected {
  /** Screen position in pixels (y grows downwards, like canvas). */
  x: number;
  y: number;
  /** Distance from the camera along the view axis; larger is farther. */
  depth: number;
}

/**
 * Perspective projection. The scene is centred on (0, 0, HEIGHT/2) and rotated
 * into view space, where the camera sits on the +z axis at `distance` looking
 * down. A point at view-space height v_z is `distance − v_z` away, and the
 * perspective divide scales its x and y by focal / depth: twice as far, half as big.
 */
export function project(p: Vec3, cam: Camera, m: Mat3 = viewMatrix(cam)): Projected {
  const [vx, vy, vz] = apply(m, [p[0], p[1], p[2] - HEIGHT / 2]);
  const depth = cam.distance - vz;
  const s = cam.focal / Math.max(depth, 1e-6);
  return { x: cam.cx + vx * s, y: cam.cy - vy * s, depth };
}

/** Linear heights, or log heights for surfaces that span orders of magnitude. */
export type HeightMode = "linear" | "log";

/**
 * Pick the height scale from sampled values. If the range is more than
 * `ratio` times the median's rise above the minimum, a linear plot would be a
 * flat sheet with a few spikes at the corners, so use log(1 + f − lo).
 */
export function chooseHeightMode(values: readonly number[], ratio = 12): HeightMode {
  const finite = values.filter(Number.isFinite);
  if (finite.length === 0) return "linear";
  const sorted = [...finite].sort((a, b) => a - b);
  const lo = sorted[0]!;
  const hi = sorted[sorted.length - 1]!;
  const med = sorted[sorted.length >> 1]!;
  if (hi - lo <= 0) return "linear";
  if (med - lo <= 0) return "log";
  return (hi - lo) / (med - lo) > ratio ? "log" : "linear";
}

/**
 * Map f to [0, 1]: lo → 0 and hi → 1, through log(1 + f − lo) in log mode.
 * Values outside [lo, hi] are clamped, so a trajectory that leaves the window
 * stays on the drawing.
 */
export function heightMap(lo: number, hi: number, mode: HeightMode): (f: number) => number {
  const span = hi - lo;
  if (!(span > 0)) return () => 0;
  const clamp = (t: number) => (Number.isFinite(t) ? Math.min(1, Math.max(0, t)) : t > 0 ? 1 : 0);
  if (mode === "log") {
    const top = Math.log1p(span);
    return (f) => clamp(Math.log1p(Math.max(0, f - lo)) / top);
  }
  return (f) => clamp((f - lo) / span);
}

/** Surface coordinates (x, y) → world floor coordinates in [−1, 1]². */
export function toWorldXY(bounds: Surface["bounds"], x: number, y: number): [number, number] {
  const [x0, x1, y0, y1] = bounds;
  return [((x - x0) / (x1 - x0)) * 2 - 1, ((y - y0) / (y1 - y0)) * 2 - 1];
}

export interface Mesh {
  /** Cells per side; the mesh has (n + 1)² vertices and n² quads. */
  n: number;
  vertices: Vec3[];
  /** f at every vertex. */
  values: number[];
  /** Normalised height in [0, 1] at every vertex. */
  heights: number[];
  /** Vertex indices of each quad, counter-clockwise seen from above. */
  quads: [number, number, number, number][];
  mode: HeightMode;
  lo: number;
  hi: number;
  /** The map used for the mesh; apply it to trajectory losses too. */
  height: (f: number) => number;
}

/** Sample the surface on an (n + 1) × (n + 1) grid over its plot window. */
export function buildMesh(surface: Surface, n: number, mode?: HeightMode): Mesh {
  const [x0, x1, y0, y1] = surface.bounds;
  const values: number[] = [];
  for (let j = 0; j <= n; j++)
    for (let i = 0; i <= n; i++) values.push(surface.f(x0 + ((x1 - x0) * i) / n, y0 + ((y1 - y0) * j) / n));
  let lo = Infinity;
  let hi = -Infinity;
  for (const v of values) {
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  const m = mode ?? chooseHeightMode(values);
  const height = heightMap(lo, hi, m);
  const heights = values.map(height);
  const vertices: Vec3[] = [];
  for (let j = 0; j <= n; j++)
    for (let i = 0; i <= n; i++) vertices.push([(2 * i) / n - 1, (2 * j) / n - 1, heights[j * (n + 1) + i]! * HEIGHT]);
  const quads: Mesh["quads"] = [];
  const row = n + 1;
  for (let j = 0; j < n; j++)
    for (let i = 0; i < n; i++) {
      const a = j * row + i;
      quads.push([a, a + 1, a + row + 1, a + row]);
    }
  return { n, vertices, values, heights, quads, mode: m, lo, hi, height };
}

/** Unit normal of a quad from the cross product of its diagonals (robust for non-planar quads). */
export function quadNormal(a: Vec3, b: Vec3, c: Vec3, d: Vec3): Vec3 {
  const u: Vec3 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
  const v: Vec3 = [d[0] - b[0], d[1] - b[1], d[2] - b[2]];
  const n: Vec3 = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
  const len = Math.hypot(n[0], n[1], n[2]);
  return len > 0 ? [n[0] / len, n[1] / len, n[2] / len] : [0, 0, 1];
}

/** Lambertian brightness in [ambient, 1] for a unit normal and a unit light direction. */
export function lambert(normal: Vec3, light: Vec3, ambient = 0.35): number {
  const d = normal[0] * light[0] + normal[1] * light[1] + normal[2] * light[2];
  return ambient + (1 - ambient) * Math.max(0, d);
}

/**
 * Painter's algorithm: indices of `depths` from farthest to nearest, so drawing
 * in this order lets nearer shapes paint over farther ones. Ties keep their
 * original order (Array.prototype.sort is stable).
 */
export function painterOrder(depths: readonly number[]): number[] {
  return depths.map((_, i) => i).sort((a, b) => depths[b]! - depths[a]!);
}
