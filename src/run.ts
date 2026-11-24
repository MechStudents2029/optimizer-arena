import type { Optimizer } from "./optimizers.ts";
import { initState } from "./optimizers.ts";
import type { Surface, Vec2 } from "./surfaces.ts";

export interface RunOptions {
  /** Maximum number of steps. */
  steps?: number;
  /** Starting point; defaults to the surface's own start. */
  start?: Vec2;
  /** Stop once f − f* < tol (ignored on surfaces with no minimum). */
  tol?: number;
  /** Treat |θ| beyond this, or a non-finite value, as divergence. */
  divergeAt?: number;
  /** Keep running after reaching tol (useful for drawing full trajectories). */
  runPastTol?: boolean;
}

export interface Trajectory {
  /** θ₀, θ₁, …: every point visited, starting point included. */
  points: Vec2[];
  /** f at each point. */
  losses: number[];
  /** First step index at which f − f* < tol, or null if never reached. */
  stepsToTol: number | null;
  diverged: boolean;
  final: Vec2;
  finalLoss: number;
}

/** Run an optimizer on a surface and record the whole path. */
export function run(surface: Surface, opt: Optimizer, options: RunOptions = {}): Trajectory {
  const { steps = 1000, tol = 1e-6, divergeAt = 1e6, runPastTol = false } = options;
  const grad = (p: Vec2) => surface.grad(p[0], p[1]);
  let state = initState(options.start ?? surface.start);
  const points: Vec2[] = [state.params];
  const losses: number[] = [surface.f(...state.params)];
  const hasMin = Number.isFinite(surface.minValue);
  let stepsToTol: number | null = hasMin && losses[0]! - surface.minValue < tol ? 0 : null;
  let diverged = false;

  for (let k = 1; k <= steps; k++) {
    if (stepsToTol !== null && !runPastTol) break;
    state = opt.step(state, grad);
    const [x, y] = state.params;
    const loss = surface.f(x, y);
    if (!Number.isFinite(loss) || Math.abs(x) > divergeAt || Math.abs(y) > divergeAt) {
      diverged = true;
      break;
    }
    points.push(state.params);
    losses.push(loss);
    if (stepsToTol === null && hasMin && loss - surface.minValue < tol) stepsToTol = k;
  }

  const final = points[points.length - 1]!;
  return { points, losses, stepsToTol, diverged, final, finalLoss: losses[losses.length - 1]! };
}

/** Index of the known minimum closest to p (useful on Himmelblau). */
export function nearestMinimum(surface: Surface, p: Vec2): number {
  let best = -1;
  let bestD = Infinity;
  surface.minima.forEach(([mx, my], i) => {
    const d = Math.hypot(p[0] - mx, p[1] - my);
    if (d < bestD) {
      bestD = d;
      best = i;
    }
  });
  return best;
}
