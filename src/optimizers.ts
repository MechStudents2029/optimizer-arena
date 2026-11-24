import type { Vec2 } from "./surfaces.ts";

/** Gradient oracle: the optimizer may ask for ∇f anywhere (Nesterov looks ahead). */
export type GradFn = (p: Vec2) => Vec2;

/** Everything an optimizer carries between steps. Never mutated in place. */
export interface OptState {
  /** Current parameters θ. */
  params: Vec2;
  /** Number of steps taken so far. */
  t: number;
  /** First buffer: velocity (momentum) or first-moment estimate m (Adam). */
  m: Vec2;
  /** Second buffer: running mean of squared gradients (RMSProp, Adam). */
  v: Vec2;
}

export interface Optimizer {
  id: string;
  name: string;
  lr: number;
  /** One step: a pure function from state to state. */
  step(state: OptState, grad: GradFn): OptState;
}

export function initState(params: Vec2): OptState {
  return { params: [params[0], params[1]], t: 0, m: [0, 0], v: [0, 0] };
}

const map2 = (f: (i: 0 | 1) => number): Vec2 => [f(0), f(1)];

/** Plain gradient descent: θ ← θ − η∇f(θ). */
export function sgd({ lr = 0.01 }: { lr?: number } = {}): Optimizer {
  return {
    id: "sgd",
    name: "Gradient descent",
    lr,
    step(s, grad) {
      const g = grad(s.params);
      return { ...s, t: s.t + 1, params: map2((i) => s.params[i] - lr * g[i]) };
    },
  };
}

/**
 * Heavy-ball (Polyak) momentum:
 *   v ← βv − η∇f(θ),  θ ← θ + v.
 */
export function momentum({ lr = 0.01, beta = 0.9 }: { lr?: number; beta?: number } = {}): Optimizer {
  return {
    id: "momentum",
    name: "Momentum",
    lr,
    step(s, grad) {
      const g = grad(s.params);
      const m = map2((i) => beta * s.m[i] - lr * g[i]);
      return { ...s, t: s.t + 1, m, params: map2((i) => s.params[i] + m[i]) };
    },
  };
}

/**
 * Nesterov accelerated gradient, in the look-ahead form of Sutskever et al. (2013):
 *   v ← βv − η∇f(θ + βv),  θ ← θ + v.
 * The gradient is taken where momentum is about to carry us, not where we are.
 */
export function nesterov({ lr = 0.01, beta = 0.9 }: { lr?: number; beta?: number } = {}): Optimizer {
  return {
    id: "nesterov",
    name: "Nesterov",
    lr,
    step(s, grad) {
      const g = grad(map2((i) => s.params[i] + beta * s.m[i]));
      const m = map2((i) => beta * s.m[i] - lr * g[i]);
      return { ...s, t: s.t + 1, m, params: map2((i) => s.params[i] + m[i]) };
    },
  };
}

/**
 * RMSProp: divide each coordinate's step by a running RMS of its gradients.
 *   s ← ρs + (1 − ρ)g²,  θ ← θ − η g / (√s + ε).
 */
export function rmsprop({
  lr = 0.01,
  rho = 0.9,
  eps = 1e-8,
}: { lr?: number; rho?: number; eps?: number } = {}): Optimizer {
  return {
    id: "rmsprop",
    name: "RMSProp",
    lr,
    step(s, grad) {
      const g = grad(s.params);
      const v = map2((i) => rho * s.v[i] + (1 - rho) * g[i] * g[i]);
      return {
        ...s,
        t: s.t + 1,
        v,
        params: map2((i) => s.params[i] - (lr * g[i]) / (Math.sqrt(v[i]) + eps)),
      };
    },
  };
}

/**
 * Adam (Kingma & Ba, 2015): momentum on the gradient, RMSProp scaling, and
 * bias correction for the zero-initialised moment estimates.
 *   m ← β₁m + (1 − β₁)g,  v ← β₂v + (1 − β₂)g²,
 *   m̂ = m / (1 − β₁ᵗ),  v̂ = v / (1 − β₂ᵗ),  θ ← θ − η m̂ / (√v̂ + ε).
 */
export function adam({
  lr = 0.01,
  beta1 = 0.9,
  beta2 = 0.999,
  eps = 1e-8,
}: { lr?: number; beta1?: number; beta2?: number; eps?: number } = {}): Optimizer {
  return {
    id: "adam",
    name: "Adam",
    lr,
    step(s, grad) {
      const g = grad(s.params);
      const t = s.t + 1;
      const m = map2((i) => beta1 * s.m[i] + (1 - beta1) * g[i]);
      const v = map2((i) => beta2 * s.v[i] + (1 - beta2) * g[i] * g[i]);
      const c1 = 1 - beta1 ** t;
      const c2 = 1 - beta2 ** t;
      return {
        params: map2((i) => s.params[i] - (lr * (m[i] / c1)) / (Math.sqrt(v[i] / c2) + eps)),
        t,
        m,
        v,
      };
    },
  };
}

export type OptimizerId = "sgd" | "momentum" | "nesterov" | "rmsprop" | "adam";

export const optimizerIds: OptimizerId[] = ["sgd", "momentum", "nesterov", "rmsprop", "adam"];

/** Build any optimizer by id with a given learning rate and default hyperparameters. */
export function makeOptimizer(id: OptimizerId, lr: number): Optimizer {
  switch (id) {
    case "sgd":
      return sgd({ lr });
    case "momentum":
      return momentum({ lr });
    case "nesterov":
      return nesterov({ lr });
    case "rmsprop":
      return rmsprop({ lr });
    case "adam":
      return adam({ lr });
  }
}
