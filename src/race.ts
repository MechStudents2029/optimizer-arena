import { makeOptimizer, optimizerIds, type OptimizerId } from "./optimizers.ts";
import { run, type Trajectory } from "./run.ts";
import type { Surface } from "./surfaces.ts";

/** Learning rates every optimizer is tried with; the best one per surface wins. */
export const LR_GRID = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1];

/** On a surface with no minimum, "escaping" means f drops below this. */
export const ESCAPE_LEVEL = -1;

export interface RaceOptions {
  steps?: number;
  tol?: number;
  lrGrid?: number[];
}

export interface RaceEntry {
  optimizer: OptimizerId;
  /** Best learning rate from the grid, or null if every rate diverged. */
  lr: number | null;
  /**
   * Steps to reach f − f* < tol (or f < ESCAPE_LEVEL on a surface with no
   * minimum); null if the budget ran out first.
   */
  steps: number | null;
  finalLoss: number;
  trajectory: Trajectory | null;
}

/** First step at which the trajectory meets the surface's goal. */
export function stepsToGoal(surface: Surface, t: Trajectory): number | null {
  if (Number.isFinite(surface.minValue)) return t.stepsToTol;
  const k = t.losses.findIndex((l) => l < ESCAPE_LEVEL);
  return k < 0 ? null : k;
}

/** Ranking: reached the goal in fewer steps, then lower final loss. */
function better(a: RaceEntry, b: RaceEntry): boolean {
  if (a.steps !== null && b.steps !== null) return a.steps < b.steps;
  if (a.steps !== null) return true;
  if (b.steps !== null) return false;
  return a.finalLoss < b.finalLoss;
}

/**
 * Race every optimizer on one surface. Each optimizer gets the learning rate
 * from the grid that serves it best, so the comparison is between tuned
 * optimizers rather than one shared η that favours some of them.
 */
export function race(surface: Surface, options: RaceOptions = {}): RaceEntry[] {
  const { steps = 10_000, tol = 1e-6, lrGrid = LR_GRID } = options;
  const goalIsEscape = !Number.isFinite(surface.minValue);
  return optimizerIds.map((id) => {
    let best: RaceEntry = { optimizer: id, lr: null, steps: null, finalLoss: Infinity, trajectory: null };
    for (const lr of lrGrid) {
      // On the saddle, stop as soon as the optimizer is clearly past it.
      const t = run(surface, makeOptimizer(id, lr), {
        steps,
        tol,
        divergeAt: goalIsEscape ? 1e3 : 1e6,
      });
      if (t.diverged && !goalIsEscape) continue;
      const entry: RaceEntry = {
        optimizer: id,
        lr,
        steps: stepsToGoal(surface, t),
        finalLoss: t.finalLoss,
        trajectory: t,
      };
      if (better(entry, best)) best = entry;
    }
    return best;
  });
}
