# Which optimizer when

A guide drawn only from what this repo has measured. Every number below is
copied from [race-results.md](race-results.md), [lr-schedules.md](lr-schedules.md),
[hyperparameter-sweeps.md](hyperparameter-sweeps.md) or
[saddle-point.md](saddle-point.md), and all of them are reproducible with
`npm run race` and `npm run sweep` (re-run for this page on 2026-11-28; the
tables matched byte for byte).

The usual setup: best η from `{1e-4, 3e-4, …, 1}` per cell, 10,000-step budget,
goal f − f* < 1e-6 (saddle: f < −1 at a fixed η = 0.01), constant rate unless a
schedule is named. "Off-lattice" means the start nudged by (√2, e)·10⁻³, which
removes Lion's exact-landing coincidences (see
[lr-schedules.md](lr-schedules.md#why-two-tables-per-schedule-lions-lattice)).

**Caveat first.** These are 2D surfaces with exact gradients, no noise, no
mini-batches and no generalisation. They show *mechanisms* (what each update
rule does to a valley, a saddle, a ripple) and are not a benchmark for training
networks. Read "use X" as "on this kind of geometry, X did best here".

## Short version

| situation | pick | measured evidence |
| --- | --- | --- |
| No idea yet, one optimizer for everything | **Adam** (defaults) | Reaches the goal on 6 of 8 surfaces at a constant rate (every surface except Rastrigin and Ackley) and 7 of 8 with cosine (6 from the off-lattice start). Never the fastest by much, never stuck on a smooth surface. |
| Long, curved, narrow valley (Rosenbrock) | **Momentum**, then Adam | Momentum 450 steps (440 off-lattice), Adam 679. GD stops at f = 5.9e-5 and RMSProp at 2.3e-4 after 10,000 steps. |
| Ill-conditioned bowl (κ = 10) | **RMSProp**, Nesterov, AdaGrad | RMSProp 25 (η = 0.3), Nesterov 45, AdaGrad 47, GD 76. |
| Well-conditioned convex problem (Booth, κ = 9) | **plain GD** or RMSProp | GD 14 (η = 0.03), RMSProp 14 (η = 1), Nesterov 21; momentum 88 and Adam 93. |
| Saddle points | anything **with momentum or a normaliser** | Momentum 72, Nesterov 68, RMSProp 75, Adam 80, AdamW 81; GD 349 and AdaGrad 1,440. |
| Many local minima (Rastrigin) or a flat outer region (Ackley) | **a decaying schedule** matters more than the optimizer | No optimizer reaches either at a constant rate. Cosine: GD reaches Rastrigin in 7,406 steps. Step decay: RMSProp reaches both (3,001 and 2,001). |
| Short runs, few steps | **AdaGrad** at a large η | Fastest on Himmelblau (16 steps, η = 1) and 47 on the bowl. |
| You need weight decay | **AdamW**, with a small λ | On Booth, λ = 0 takes 93 steps, 1e-4 takes 705, 1e-2 takes 8,910. |
| Optimizer memory is tight | **Lion**, with a schedule | One state buffer instead of Adam's two; at a constant rate off the lattice it is slow (bowl 3,999 steps), with cosine it reaches 6 of 8 surfaces. |

## Robustness: how many surfaces each one reaches

Cells that reach the goal, out of the 8 surfaces, counted from the tables in
[lr-schedules.md](lr-schedules.md) (standard start / off-lattice start):

| schedule | GD | Momentum | Nesterov | RMSProp | Adam | AdaGrad | AdamW | Lion | total of 64 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| constant | 5 / 5 | 6 / 6 | 6 / 6 | 4 / 4 | 6 / 6 | 6 / 6 | 5 / 5 | 5 / 5 | 43 / 43 |
| step decay | 6 / 6 | 7 / 6 | 6 / 6 | **8** / 7 | 6 / 6 | 5 / 5 | 5 / 5 | 6 / 6 | 49 / 47 |
| cosine | 7 / 7 | 6 / 7 | 6 / 6 | 7 / 7 | 7 / 6 | 5 / 6 | 6 / 6 | 6 / 6 | 50 / 51 |
| warmup + cosine | 7 / 7 | 7 / 6 | 6 / 7 | 7 / 7 | 6 / 6 | 5 / 5 | 3 / 3 | 7 / 7 | 48 / 48 |

What this says:

- **Adding any schedule helps more than switching optimizer.** The best
  constant-rate optimizers reach 6 surfaces; the totals go from 43 to 47–51
  with a schedule.
- **RMSProp is the most schedule-dependent.** Worst at a constant rate (4),
  it is the only optimizer to reach all 8 (step decay, standard start):
  Rosenbrock in 6,001, Rastrigin in 3,001 and Ackley in 2,001 steps. Its
  roughly constant-size steps are what stall it on Rosenbrock; shrinking η
  fixes exactly that.
- **AdamW drops under warmup + cosine** (3 of 8): it misses Booth
  (f = 3.0e-6), Beale (f = 1.4e-6) and Rosenbrock (f = 6.8e-6), each just
  short of 1e-6, because the decay keeps pulling toward the origin.
- **AdaGrad is the least helped** (5 or 6 everywhere): its step already
  shrinks as 1/√(sum of g²), so a schedule shrinks it twice.

## By optimizer

**Gradient descent.** Best when the problem is well conditioned: 14 steps on
Booth, 18 on Himmelblau. It is limited by η < 2/λ_max (0.19 converges and 0.21
diverges on the bowl, a unit test), so on Rosenbrock it crawls (f = 5.9e-5
after 10,000 steps), and on the saddle it grows by only (1 + 2η) per step:
349 steps, derived exactly in [saddle-point.md](saddle-point.md). With cosine
it is the one method that reaches Rastrigin's global minimum from both starts
(7,406 steps).

**Momentum (heavy ball).** The Rosenbrock winner (450) and a fast saddle
escaper (72, from its eigenvalue ≈ 1.107 per step). β matters and the best
value depends on the surface: on the bowl at η = 0.01, β = 0.8 takes 114 steps
against 132 at 0.9 and 1,238 at 0.99; on Rosenbrock at η = 0.001, β = 0.95
takes 637 against 1,386 at 0.9. With warmup + cosine it can use η = 0.1 on
Beale and needs 127 steps instead of 405.

**Nesterov.** Same as momentum but with the gradient taken at the look-ahead
point, which damps overshoot: 45 on the bowl (momentum 132) and 68 on the
saddle. On Rosenbrock it is slower than momentum (1,426 against 450).
Warmup + cosine takes it from 403 to 117 steps on Beale.

**RMSProp.** Per-coordinate scaling undoes conditioning: bowl 25, Booth 14.
At a constant rate it stalls in valleys (Rosenbrock f = 2.3e-4; in the Beale
ρ sweep at η = 0.001 only ρ = 0.999 converges, in 8,675 steps). Use it with a schedule:
under step decay it reaches every surface.

**Adam.** The safe default. Fastest of the non-Lion methods on Beale (205),
second on Rosenbrock (679), and at the bottom of Ackley's funnel under cosine
(9,981 steps). Keep the defaults: on Rosenbrock at η = 0.1, β₁ = 0.9 takes
1,139 steps, β₁ = 0 or 0.5 never get there, and β₂ = 0.9 or 0.99 also miss.
ε barely matters until it is large (bowl: 149 steps for every ε up to 0.01,
167 at ε = 1). Its weak spot is easy convex problems, where it pays for its
averaging: 93 steps on Booth against GD's 14.

**AdaGrad.** The step only shrinks, so the largest rate in the grid (η = 1)
wins for it everywhere it converges. That makes it excellent early
(Himmelblau 16, bowl 47) and poor late: Rosenbrock 8,491 and the saddle 1,440
(√t growth, about 1/(8η²) = 1,250 predicted). Do not combine it with warmup:
on the bowl it goes from 47 to 571, because the sum of g² is built up while η
is still tiny. ε is irrelevant up to 0.01 (47 steps).

**AdamW.** Equal to Adam when the minimum is at the origin (bowl 124 against
132, saddle 81 against 80), and slower everywhere else because decoupled decay
moves the fixed point toward 0: Booth 8,910, Beale 9,497, Himmelblau 4,336, and
Rosenbrock stops at f = 3.4e-6. Booth across λ: 93 (λ = 0), 93 (1e-5), 705
(1e-4), 4,616 (1e-3), 8,910 (1e-2), not reached at 0.1. On the bowl decay
helps: 37 steps at λ = 1. Here there is no overfitting to regularise against,
so this is the pure *cost* of decay; pick λ as small as your regulariser
allows.

**Lion.** Read its standard-start numbers with care: bowl 4, Booth 7, Beale 62
and Rosenbrock 290 are exact landings on the η-lattice (f ≈ 1e-28 or 0). Off
the lattice at a constant rate it needs 3,999 steps on the bowl, 9,376 on
Rosenbrock and 3,211 on Beale, and it never reaches Himmelblau (f = 2.3e-5).
Sign steps have a fixed size η, so it needs η to shrink: with step decay it
reaches Himmelblau off the lattice in 4,001 steps and with cosine in 6,682, and
under warmup + cosine it reaches 7 of 8 surfaces, Ackley included. The default
betas are the best measured (Beale, cosine, η = 0.01, off-lattice start: 5,804
steps at β₁ = 0.9, β₂ = 0.99); β₂ = 0.999 never converges.

## Schedules

- **Cosine** gives the best totals (50 / 51 of 64) and is what lets GD find
  Rastrigin's global minimum (7,406) and RMSProp converge on Rosenbrock
  (8,370).
- **Step decay** is the best for RMSProp (8 of 8) and gets GD and RMSProp to
  the bottom of Ackley soonest (3,001 and 2,001 steps).
- **Warmup + cosine** lets momentum and Nesterov use a 30× larger peak rate on
  Beale (η = 0.1: 127 and 117 steps, against 405 and 403), and lets GD win
  Booth at η = 1 (20 steps), about nine times past the constant-rate
  stability limit 2/λ_max = 1/9 (Booth's Hessian has eigenvalues 18 and 2):
  it is done while the ramped rate is still below 0.04. It
  hurts AdaGrad (bowl 47 → 571) and AdamW (3 of 8).
- **Runs that finish early do not care.** GD, Nesterov, RMSProp and AdaGrad
  give the same counts on the bowl under constant, step and cosine (76, 45,
  25, 47): they are done before the rate has decayed.

## Reproduce

```sh
npm run race                                   # constant rate
npm run race -- --schedule cosine              # or step, warmup-cosine
npm run race -- --schedule cosine --start-off-lattice
npm run sweep                                  # one hyperparameter at a time
```

Or try any row in the [live demo](https://optimizer-arena.vercel.app), for
example
[Lion against Adam on Himmelblau under cosine](https://optimizer-arena.vercel.app/?surface=himmelblau&schedule=cosine&only=adam,adagrad,adamw,lion).
