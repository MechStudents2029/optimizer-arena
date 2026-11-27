import type { Schedule } from "./schedules.ts";
import type { Vec2 } from "./surfaces.ts";

/** Gradient oracle: the optimizer may ask for ∇f anywhere (Nesterov looks ahead). */
export type GradFn = (p: Vec2) => Vec2;

/** Everything an optimizer carries between steps. Never mutated in place. */
export interface OptState {
  /** Current parameters θ. */
  params: Vec2;
  /** Number of steps taken so far. */
  t: number;
  /** First buffer: velocity (momentum) or first-moment estimate m (Adam, AdamW, Lion). */
  m: Vec2;
  /** Second buffer: running mean (RMSProp, Adam, AdamW) or running sum (AdaGrad) of squared gradients. */
  v: Vec2;
}

export interface Optimizer {
  id: string;
  name: string;
  /** Base (peak) learning rate η; a schedule scales it per step. */
  lr: number;
  /** One step: a pure function from state to state. */
  step(state: OptState, grad: GradFn): OptState;
}

export function initState(params: Vec2): OptState {
  return { params: [params[0], params[1]], t: 0, m: [0, 0], v: [0, 0] };
}

const map2 = (f: (i: 0 | 1) => number): Vec2 => [f(0), f(1)];

/** η_t = η · s(t) for the step taken from state t (t = 0 on the first step). */
const rate = (lr: number, schedule: Schedule | undefined, t: number): number => (schedule ? lr * schedule(t) : lr);

/** Options every optimizer accepts: the base rate and an optional schedule. */
export interface BaseOptions {
  lr?: number;
  schedule?: Schedule;
}

/** Plain gradient descent: θ ← θ − η∇f(θ). */
export function sgd({ lr = 0.01, schedule }: BaseOptions = {}): Optimizer {
  return {
    id: "sgd",
    name: "Gradient descent",
    lr,
    step(s, grad) {
      const g = grad(s.params);
      const eta = rate(lr, schedule, s.t);
      return { ...s, t: s.t + 1, params: map2((i) => s.params[i] - eta * g[i]) };
    },
  };
}

/**
 * Heavy-ball (Polyak) momentum:
 *   v ← βv − η∇f(θ),  θ ← θ + v.
 */
export function momentum({ lr = 0.01, beta = 0.9, schedule }: BaseOptions & { beta?: number } = {}): Optimizer {
  return {
    id: "momentum",
    name: "Momentum",
    lr,
    step(s, grad) {
      const g = grad(s.params);
      const eta = rate(lr, schedule, s.t);
      const m = map2((i) => beta * s.m[i] - eta * g[i]);
      return { ...s, t: s.t + 1, m, params: map2((i) => s.params[i] + m[i]) };
    },
  };
}

/**
 * Nesterov accelerated gradient, in the look-ahead form of Sutskever et al. (2013):
 *   v ← βv − η∇f(θ + βv),  θ ← θ + v.
 * The gradient is taken where momentum is about to carry us, not where we are.
 */
export function nesterov({ lr = 0.01, beta = 0.9, schedule }: BaseOptions & { beta?: number } = {}): Optimizer {
  return {
    id: "nesterov",
    name: "Nesterov",
    lr,
    step(s, grad) {
      const g = grad(map2((i) => s.params[i] + beta * s.m[i]));
      const eta = rate(lr, schedule, s.t);
      const m = map2((i) => beta * s.m[i] - eta * g[i]);
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
  schedule,
}: BaseOptions & { rho?: number; eps?: number } = {}): Optimizer {
  return {
    id: "rmsprop",
    name: "RMSProp",
    lr,
    step(s, grad) {
      const g = grad(s.params);
      const v = map2((i) => rho * s.v[i] + (1 - rho) * g[i] * g[i]);
      const eta = rate(lr, schedule, s.t);
      return {
        ...s,
        t: s.t + 1,
        v,
        params: map2((i) => s.params[i] - (eta * g[i]) / (Math.sqrt(v[i]) + eps)),
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
  schedule,
}: BaseOptions & { beta1?: number; beta2?: number; eps?: number } = {}): Optimizer {
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
      const eta = rate(lr, schedule, s.t);
      return {
        params: map2((i) => s.params[i] - (eta * (m[i] / c1)) / (Math.sqrt(v[i] / c2) + eps)),
        t,
        m,
        v,
      };
    },
  };
}

/**
 * AdaGrad (Duchi, Hazan & Singer, 2011): divide by the root of the running
 * *sum* of squared gradients, so each coordinate's step can only shrink.
 *   G ← G + g²,  θ ← θ − η g / (√G + ε).
 */
export function adagrad({ lr = 0.01, eps = 1e-8, schedule }: BaseOptions & { eps?: number } = {}): Optimizer {
  return {
    id: "adagrad",
    name: "AdaGrad",
    lr,
    step(s, grad) {
      const g = grad(s.params);
      const v = map2((i) => s.v[i] + g[i] * g[i]);
      const eta = rate(lr, schedule, s.t);
      return {
        ...s,
        t: s.t + 1,
        v,
        params: map2((i) => s.params[i] - (eta * g[i]) / (Math.sqrt(v[i]) + eps)),
      };
    },
  };
}

/**
 * AdamW (Loshchilov & Hutter, 2019): Adam with *decoupled* weight decay. The
 * decay is applied to θ directly, outside the adaptive m̂/√v̂ ratio:
 *   θ ← θ − η_t (m̂ / (√v̂ + ε) + λθ).
 * L2 in Adam would instead add λθ to g, where it gets divided by √v̂ too.
 */
export function adamw({
  lr = 0.01,
  beta1 = 0.9,
  beta2 = 0.999,
  eps = 1e-8,
  weightDecay = 0.01,
  schedule,
}: BaseOptions & { beta1?: number; beta2?: number; eps?: number; weightDecay?: number } = {}): Optimizer {
  return {
    id: "adamw",
    name: "AdamW",
    lr,
    step(s, grad) {
      const g = grad(s.params);
      const t = s.t + 1;
      const m = map2((i) => beta1 * s.m[i] + (1 - beta1) * g[i]);
      const v = map2((i) => beta2 * s.v[i] + (1 - beta2) * g[i] * g[i]);
      const c1 = 1 - beta1 ** t;
      const c2 = 1 - beta2 ** t;
      const eta = rate(lr, schedule, s.t);
      return {
        params: map2(
          (i) => s.params[i] - eta * ((m[i] / c1) / (Math.sqrt(v[i] / c2) + eps) + weightDecay * s.params[i]),
        ),
        t,
        m,
        v,
      };
    },
  };
}

/**
 * Lion (Chen et al., 2023, "Symbolic Discovery of Optimization Algorithms"):
 * the step is the *sign* of an interpolation between momentum and the fresh
 * gradient, so every coordinate moves by exactly η (plus decay). Only one
 * buffer, m, is kept.
 *   c = β₁m + (1 − β₁)g,  θ ← θ − η(sign(c) + λθ),  m ← β₂m + (1 − β₂)g.
 */
export function lion({
  lr = 1e-3,
  beta1 = 0.9,
  beta2 = 0.99,
  weightDecay = 0,
  schedule,
}: BaseOptions & { beta1?: number; beta2?: number; weightDecay?: number } = {}): Optimizer {
  return {
    id: "lion",
    name: "Lion",
    lr,
    step(s, grad) {
      const g = grad(s.params);
      const c = map2((i) => beta1 * s.m[i] + (1 - beta1) * g[i]);
      const eta = rate(lr, schedule, s.t);
      return {
        ...s,
        t: s.t + 1,
        m: map2((i) => beta2 * s.m[i] + (1 - beta2) * g[i]),
        params: map2((i) => s.params[i] - eta * (Math.sign(c[i]) + weightDecay * s.params[i])),
      };
    },
  };
}

export type OptimizerId = "sgd" | "momentum" | "nesterov" | "rmsprop" | "adam" | "adagrad" | "adamw" | "lion";

export const optimizerIds: OptimizerId[] = ["sgd", "momentum", "nesterov", "rmsprop", "adam", "adagrad", "adamw", "lion"];

/** Hyperparameters other than η. Anything left out keeps the optimizer's default. */
export interface Hyperparams {
  /** Momentum / Nesterov velocity decay β. */
  beta?: number;
  /** RMSProp squared-gradient decay ρ. */
  rho?: number;
  /** Adam / AdamW first-moment decay β₁. */
  beta1?: number;
  /** Adam / AdamW second-moment decay β₂. */
  beta2?: number;
  /** Denominator guard ε for RMSProp, Adam, AdamW and AdaGrad. */
  eps?: number;
  /** AdamW's decoupled weight decay λ. */
  weightDecay?: number;
  /** Lion's interpolation β₁ (used inside the sign). */
  lionBeta1?: number;
  /** Lion's momentum-tracking β₂. */
  lionBeta2?: number;
}

export const DEFAULT_HYPERPARAMS: Required<Hyperparams> = {
  beta: 0.9,
  rho: 0.9,
  beta1: 0.9,
  beta2: 0.999,
  eps: 1e-8,
  weightDecay: 0.01,
  lionBeta1: 0.9,
  lionBeta2: 0.99,
};

/** Build any optimizer by id with a learning rate and optional hyperparameters. */
export function makeOptimizer(id: OptimizerId, lr: number, hp: Hyperparams = {}, schedule?: Schedule): Optimizer {
  const h = { ...DEFAULT_HYPERPARAMS, ...stripUndefined(hp) };
  switch (id) {
    case "sgd":
      return sgd({ lr, schedule });
    case "momentum":
      return momentum({ lr, beta: h.beta, schedule });
    case "nesterov":
      return nesterov({ lr, beta: h.beta, schedule });
    case "rmsprop":
      return rmsprop({ lr, rho: h.rho, eps: h.eps, schedule });
    case "adam":
      return adam({ lr, beta1: h.beta1, beta2: h.beta2, eps: h.eps, schedule });
    case "adagrad":
      return adagrad({ lr, eps: h.eps, schedule });
    case "adamw":
      return adamw({ lr, beta1: h.beta1, beta2: h.beta2, eps: h.eps, weightDecay: h.weightDecay, schedule });
    case "lion":
      return lion({ lr, beta1: h.lionBeta1, beta2: h.lionBeta2, schedule });
  }
}

function stripUndefined(hp: Hyperparams): Hyperparams {
  return Object.fromEntries(Object.entries(hp).filter(([, v]) => v !== undefined)) as Hyperparams;
}
