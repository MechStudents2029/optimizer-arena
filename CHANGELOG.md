# Changelog

## Day 2 — 2026-11-25

- Three new surfaces in `src/surfaces.ts`, each with an analytic gradient, minimum, start point and plot window: Rastrigin (121 minima in the plot window, one global), Ackley (flat outer region around a deep funnel; subgradient 0 at the origin), and Booth (convex quadratic, κ = 9, the control case).
- `makeOptimizer(id, lr, hyperparams)` and `race(surface, { hyperparams })` take β, ρ, β₁, β₂ and ε; `DEFAULT_HYPERPARAMS` holds the defaults, so every Day 1 number is unchanged.
- Demo: sliders for η (log scale, with a "tune η per optimizer" toggle), momentum β, RMSProp ρ, Adam β₁, β₂ (log scale in 1 − β₂) and ε (log scale). The race reruns live while dragging, at most once per animation frame.
- Demo: a loss-vs-step chart under the race, plain canvas, log₁₀(f − f*) on the y-axis (linear f on the saddle), drawn in step with the animation.
- Reran `npm run race`: Booth converges for everyone (GD and RMSProp in 14 steps); on Rastrigin all five stop in the local minimum at (1.990, −1.990) with f = 7.96; on Ackley nobody reaches 1e-6, and Adam at η = 1 ends closest (f ≈ 0.14).
- `npm run sweep` (`scripts/sweep.ts`) measures one hyperparameter at a time; results in `docs/hyperparameter-sweeps.md` and the README.
- README: formulas, gradients and failure modes for the new surfaces, and what each slider does to the update rule.
- 79 tests (from 52): gradient checks and minima for the new surfaces, Rastrigin's off-integer local minima found by Newton's method, Ackley's origin and fading slope, Booth's Hessian, hyperparameter plumbing (β = 0 momentum is GD, ρ/β₁/β₂/ε against hand-computed steps, large ε turns Adam into momentum SGD), and race behaviour on the new surfaces.

## Day 1 — 2026-11-24

- Repo scaffold mirroring `gradient`: TypeScript, Vite, Vitest, GitHub Actions CI (typecheck, tests, build), MIT license.
- Loss surfaces in `src/surfaces.ts` with analytic gradients, known minima, start points and plot windows: an ill-conditioned quadratic bowl (κ = 10), Rosenbrock, Beale, the saddle x² − y², and Himmelblau with all four minima.
- Optimizers in `src/optimizers.ts` as pure `step(state, grad) → state` functions: gradient descent, heavy-ball momentum, Nesterov (look-ahead form), RMSProp, and Adam with bias correction.
- `run()` records full trajectories, steps to tolerance, and divergence; `race()` sweeps a learning-rate grid and ranks the optimizers per surface.
- `npm run race` (`scripts/race.ts`) prints the measured results table used in the README and `docs/race-results.md`.
- README teaches each update rule in LaTeX, the surfaces, and what the measured numbers say; `docs/saddle-point.md` derives the saddle escape rates (GD's 349 steps exactly, momentum's growth factor from its eigenvalues).
- 2D canvas demo: log-scaled contour map, animated trajectories, live step count and loss beside each update rule, click to move the start point. Deployed at https://optimizer-arena.vercel.app with the Vercel project connected to this repo.
- 52 tests: finite-difference gradient checks on every surface, hand-computed steps for every optimizer (including Adam's bias correction), convergence on the convex bowl, GD's 2/λ_max stability limit, saddle stalling and escape, and the race ranking.

## Roadmap

- [x] Day 1: surfaces, five optimizers, race script, README math, 2D demo
- [x] Day 2: optimizer hyperparameter controls (β, ρ, β₁, β₂, ε) in the demo, loss-vs-step chart, more surfaces (Rastrigin, Ackley, Booth)
- [ ] Day 3: 3D surface view with trajectories in the browser
- [ ] Day 4: AdaGrad, AdamW, Lion, and learning-rate schedules
