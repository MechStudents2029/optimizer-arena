/** A point or vector in the plane. */
export type Vec2 = [number, number];

/** A 2D loss surface with an analytic gradient. */
export interface Surface {
  id: string;
  name: string;
  /** The formula in LaTeX, for the README and the demo. */
  formula: string;
  f(x: number, y: number): number;
  grad(x: number, y: number): Vec2;
  /** Known global minimisers. Empty when the surface is unbounded below. */
  minima: Vec2[];
  /** The value at the global minimum, or -Infinity when there is none. */
  minValue: number;
  /** Default starting point for a race. */
  start: Vec2;
  /** Plot window: [xMin, xMax, yMin, yMax]. */
  bounds: [number, number, number, number];
}

/**
 * An ill-conditioned convex bowl, f = ½(x² + κy²) with κ = 10. The Hessian is
 * diag(1, κ), so plain gradient descent zig-zags across the narrow axis.
 */
export const KAPPA = 10;
export const quadratic: Surface = {
  id: "quadratic",
  name: "Quadratic bowl",
  formula: String.raw`f(x, y) = \tfrac{1}{2}\left(x^2 + 10\,y^2\right)`,
  f: (x, y) => 0.5 * (x * x + KAPPA * y * y),
  grad: (x, y) => [x, KAPPA * y],
  minima: [[0, 0]],
  minValue: 0,
  start: [-4, 2],
  bounds: [-5, 5, -3, 3],
};

/** Rosenbrock's banana: a long, curved, nearly flat valley floor. */
export const rosenbrock: Surface = {
  id: "rosenbrock",
  name: "Rosenbrock",
  formula: String.raw`f(x, y) = (1 - x)^2 + 100\,(y - x^2)^2`,
  f: (x, y) => (1 - x) ** 2 + 100 * (y - x * x) ** 2,
  grad: (x, y) => [-2 * (1 - x) - 400 * x * (y - x * x), 200 * (y - x * x)],
  minima: [[1, 1]],
  minValue: 0,
  start: [-1.2, 1],
  bounds: [-2, 2, -1, 3],
};

/** Beale: flat plateaus and steep ridges that meet near the minimum at (3, ½). */
export const beale: Surface = {
  id: "beale",
  name: "Beale",
  formula: String.raw`f(x, y) = (1.5 - x + xy)^2 + (2.25 - x + xy^2)^2 + (2.625 - x + xy^3)^2`,
  f: (x, y) => {
    const a = 1.5 - x + x * y;
    const b = 2.25 - x + x * y * y;
    const c = 2.625 - x + x * y ** 3;
    return a * a + b * b + c * c;
  },
  grad: (x, y) => {
    const a = 1.5 - x + x * y;
    const b = 2.25 - x + x * y * y;
    const c = 2.625 - x + x * y ** 3;
    return [
      2 * a * (y - 1) + 2 * b * (y * y - 1) + 2 * c * (y ** 3 - 1),
      2 * a * x + 2 * b * 2 * x * y + 2 * c * 3 * x * y * y,
    ];
  },
  minima: [[3, 0.5]],
  minValue: 0,
  start: [1, 1.5],
  bounds: [-4.5, 4.5, -4.5, 4.5],
};

/**
 * The saddle f = x² − y². The origin is a stationary point but not a minimum:
 * it curves up along x and down along y, and the surface is unbounded below.
 */
export const saddle: Surface = {
  id: "saddle",
  name: "Saddle point",
  formula: String.raw`f(x, y) = x^2 - y^2`,
  f: (x, y) => x * x - y * y,
  grad: (x, y) => [2 * x, -2 * y],
  minima: [],
  minValue: -Infinity,
  start: [1, 1e-3],
  bounds: [-1.5, 1.5, -1.5, 1.5],
};

/** Himmelblau: four global minima, all with f = 0. */
export const himmelblau: Surface = {
  id: "himmelblau",
  name: "Himmelblau",
  formula: String.raw`f(x, y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2`,
  f: (x, y) => (x * x + y - 11) ** 2 + (x + y * y - 7) ** 2,
  grad: (x, y) => {
    const a = x * x + y - 11;
    const b = x + y * y - 7;
    return [4 * x * a + 2 * b, 2 * a + 4 * y * b];
  },
  minima: [
    [3, 2],
    [-2.805118086952745, 3.131312518250573],
    [-3.779310253377747, -3.283185991286170],
    [3.584428340330492, -1.848126526964404],
  ],
  minValue: 0,
  start: [-0.5, -0.5],
  bounds: [-5, 5, -5, 5],
};

export const surfaces: Surface[] = [quadratic, rosenbrock, beale, saddle, himmelblau];

export function surfaceById(id: string): Surface {
  const s = surfaces.find((s) => s.id === id);
  if (!s) throw new Error(`unknown surface: ${id}`);
  return s;
}
