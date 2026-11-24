# Changelog

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
- [ ] Day 2: optimizer hyperparameter controls (β, ρ, β₁, β₂, ε) in the demo, loss-vs-step chart, more surfaces (Rastrigin, Ackley, Booth)
- [ ] Day 3: 3D surface view with trajectories in the browser
- [ ] Day 4: AdaGrad, AdamW, Lion, and learning-rate schedules
