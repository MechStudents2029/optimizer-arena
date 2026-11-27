/**
 * Learning-rate schedules. A schedule is a multiplier on the base rate:
 * the step taken from θ_t uses η_t = η · s(t), with t = 0 for the first step.
 * Every schedule here starts at (or ramps up to) 1, so η stays the peak rate
 * and the learning-rate grid in race() means the same thing with or without one.
 */
export type Schedule = (t: number) => number;

export type ScheduleId = "constant" | "step" | "cosine" | "warmup-cosine";

export const scheduleIds: ScheduleId[] = ["constant", "step", "cosine", "warmup-cosine"];

export const SCHEDULE_NAMES: Record<ScheduleId, string> = {
  constant: "Constant",
  step: "Step decay",
  cosine: "Cosine annealing",
  "warmup-cosine": "Linear warmup + cosine",
};

export interface ScheduleOptions {
  /** Step decay: multiply by gamma every `every` steps. Default 0.5. */
  gamma?: number;
  /** Step decay interval. Default: a tenth of the budget. */
  every?: number;
  /** Cosine schedules end at minRatio · η. Default 0. */
  minRatio?: number;
  /** Warmup length in steps. Default: 5% of the budget (at least 1). */
  warmup?: number;
}

/** η_t / η = 1. */
export const constant: Schedule = () => 1;

/** η_t / η = γ^⌊t / k⌋. */
export function stepDecay(every: number, gamma = 0.5): Schedule {
  if (!(every >= 1)) throw new RangeError(`step decay interval must be ≥ 1, got ${every}`);
  return (t) => gamma ** Math.floor(t / every);
}

/**
 * Cosine annealing (Loshchilov & Hutter, 2017, without restarts):
 *   η_t / η = r + (1 − r) · ½(1 + cos(π t / T)),  t clamped to [0, T].
 */
export function cosine(total: number, minRatio = 0): Schedule {
  if (!(total >= 1)) throw new RangeError(`cosine length must be ≥ 1, got ${total}`);
  return (t) => {
    const u = Math.min(Math.max(t, 0), total) / total;
    return minRatio + (1 - minRatio) * 0.5 * (1 + Math.cos(Math.PI * u));
  };
}

/**
 * Linear warmup for W steps, then cosine annealing over the remaining T − W:
 *   η_t / η = (t + 1) / W            for t < W
 *   η_t / η = cosine(T − W)(t − W)   for t ≥ W
 * The first step already moves (1/W of the peak), and step W − 1 reaches 1.
 */
export function warmupCosine(total: number, warmup: number, minRatio = 0): Schedule {
  if (!(warmup >= 1)) throw new RangeError(`warmup must be ≥ 1, got ${warmup}`);
  if (!(total > warmup)) throw new RangeError(`total (${total}) must exceed warmup (${warmup})`);
  const decay = cosine(total - warmup, minRatio);
  return (t) => (t < warmup ? (t + 1) / warmup : decay(t - warmup));
}

/** Build a schedule by id for a run of `total` steps. */
export function makeSchedule(id: ScheduleId, total: number, o: ScheduleOptions = {}): Schedule {
  switch (id) {
    case "constant":
      return constant;
    case "step":
      return stepDecay(o.every ?? Math.max(1, Math.round(total / 10)), o.gamma ?? 0.5);
    case "cosine":
      return cosine(total, o.minRatio ?? 0);
    case "warmup-cosine":
      return warmupCosine(total, o.warmup ?? Math.max(1, Math.round(total * 0.05)), o.minRatio ?? 0);
  }
}

/** Short formula for each schedule, used by the demo. */
export const SCHEDULE_RULES: Record<ScheduleId, string> = {
  constant: "η_t = η",
  step: "η_t = η·γ^⌊t/k⌋",
  cosine: "η_t = η·½(1 + cos πt/T)",
  "warmup-cosine": "η_t = η·(t+1)/W, then cosine over T − W",
};
