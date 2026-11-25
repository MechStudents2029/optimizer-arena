# optimizer-arena

[![CI](https://github.com/MechStudents2029/optimizer-arena/actions/workflows/ci.yml/badge.svg)](https://github.com/MechStudents2029/optimizer-arena/actions/workflows/ci.yml)
[![Live demo](https://img.shields.io/badge/demo-optimizer--arena.vercel.app-orange)](https://optimizer-arena.vercel.app)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Gradient descent, momentum (heavy-ball and Nesterov), RMSProp, and Adam racing
on classic loss surfaces (Rosenbrock, Beale, a saddle point, Himmelblau,
Rastrigin, Ackley, Booth), with the update-rule math written next to each one.

Written in TypeScript with no runtime dependencies. Work in progress.

![2D race on Rosenbrock: momentum, Nesterov and Adam reach the minimum while GD and RMSProp are still in the valley](docs/screenshots/rosenbrock.png)

**[Live demo](https://optimizer-arena.vercel.app)** (or `npm run dev` locally) opens the 2D race: a log-scaled contour map of the surface, the
five trajectories drawn step by step, and each optimizer's update rule beside
its live step count and loss. Click the plot to move the start point; the
learning-rate sweep reruns from there in the browser.

## The problem

Training a model means minimising a loss $f(\theta)$ over parameters $\theta$.
All we can cheaply ask is the gradient $\nabla f(\theta)$, the direction of
steepest ascent at the current point. Every optimizer here is a different answer
to one question: **given the gradients seen so far, where should the next step
go, and how far?**

Two-dimensional surfaces make the answers visible. Each one below isolates a
failure mode that real networks hit in millions of dimensions: narrow valleys,
flat plateaus, saddle points, and several minima.

## Update rules

$\eta$ is the learning rate, $g_t = \nabla f(\theta_t)$, and all operations on
vectors are element-wise. Each rule is a pure `step(state, grad) → state` in
[src/optimizers.ts](src/optimizers.ts).

### Gradient descent

$$
\theta_{t+1} = \theta_t - \eta\, g_t
$$

Walk downhill by a fixed multiple of the slope. On a quadratic with Hessian
eigenvalues $\lambda_{\min} \le \dots \le \lambda_{\max}$ it is stable only for
$\eta < 2/\lambda_{\max}$, and the slow direction then shrinks by
$(1 - \eta\lambda_{\min})$ per step, so the condition number
$\kappa = \lambda_{\max}/\lambda_{\min}$ sets the pace. On the bowl below
($\kappa = 10$) the test suite checks the $2/\lambda_{\max} = 0.2$ limit
directly: $\eta = 0.19$ converges, $\eta = 0.21$ blows up.

### Momentum (heavy ball, Polyak 1964)

$$
v_{t+1} = \beta v_t - \eta\, g_t, \qquad \theta_{t+1} = \theta_t + v_{t+1}
$$

Keep a velocity that is a decaying sum of past gradients ($\beta = 0.9$).
Components that keep pointing the same way add up, roughly
$1/(1-\beta) = 10\times$; components that flip sign from step to step cancel.
That damps the zig-zag across a narrow valley and speeds travel along it.

### Nesterov accelerated gradient

$$
v_{t+1} = \beta v_t - \eta\, \nabla f(\theta_t + \beta v_t), \qquad \theta_{t+1} = \theta_t + v_{t+1}
$$

Same velocity, but the gradient is measured at the look-ahead point
$\theta_t + \beta v_t$, where momentum is about to carry you anyway. If that
point is already uphill, the correction arrives one step early, so Nesterov
overshoots less than heavy ball.

### RMSProp (Hinton, 2012)

$$
s_{t+1} = \rho\, s_t + (1-\rho)\, g_t^2, \qquad
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{s_{t+1}} + \epsilon}\, g_t
$$

Divide each coordinate's step by a running RMS of its own gradients
($\rho = 0.9$, $\epsilon = 10^{-8}$). Steep directions get small steps, flat
ones get large steps, so every coordinate moves at a similar speed of about
$\eta$ per step, regardless of the raw gradient scale.

### Adam (Kingma & Ba, 2015)

$$
m_{t+1} = \beta_1 m_t + (1-\beta_1)\, g_t, \qquad
v_{t+1} = \beta_2 v_t + (1-\beta_2)\, g_t^2
$$

$$
\hat m = \frac{m_{t+1}}{1-\beta_1^{\,t+1}}, \qquad
\hat v = \frac{v_{t+1}}{1-\beta_2^{\,t+1}}, \qquad
\theta_{t+1} = \theta_t - \eta\, \frac{\hat m}{\sqrt{\hat v} + \epsilon}
$$

Momentum on the gradient (first moment $m$) plus RMSProp's scaling (second
moment $v$), with $\beta_1 = 0.9$, $\beta_2 = 0.999$. Both moments start at
zero, so early on they are biased toward zero; dividing by $1-\beta^t$ removes
that bias. Without it, the very first step would be
$\eta\,(1-\beta_1)/\sqrt{1-\beta_2} \approx 3.16\,\eta$ in every coordinate;
with it, the first step is exactly $\eta \cdot \operatorname{sign}(g)$. Both
facts are unit tests.

## Surfaces

| surface | formula | minimum | what it tests |
| --- | --- | --- | --- |
| Quadratic bowl | $\tfrac12(x^2 + 10y^2)$ | $f(0,0)=0$ | ill-conditioning ($\kappa = 10$): zig-zag across the steep axis |
| Rosenbrock | $(1-x)^2 + 100(y-x^2)^2$ | $f(1,1)=0$ | a long curved valley whose floor is nearly flat |
| Beale | $(1.5-x+xy)^2 + (2.25-x+xy^2)^2 + (2.625-x+xy^3)^2$ | $f(3,\tfrac12)=0$ | flat plateaus next to very steep walls |
| Saddle | $x^2 - y^2$ | none (unbounded below) | a stationary point that is not a minimum |
| Himmelblau | $(x^2+y-11)^2 + (x+y^2-7)^2$ | four, all $f=0$ | which basin you land in depends on the path |
| Rastrigin | $20 + \sum_{i}\left(x_i^2 - 10\cos 2\pi x_i\right)$ | $f(0,0)=0$ | a local minimum in every unit cell |
| Ackley | $-20e^{-0.2\sqrt{(x^2+y^2)/2}} - e^{(\cos 2\pi x + \cos 2\pi y)/2} + e + 20$ | $f(0,0)=0$ | a nearly flat outer region around one deep funnel |
| Booth | $(x+2y-7)^2 + (2x+y-5)^2$ | $f(1,3)=0$ | nothing: a well-conditioned quadratic, the control case |

Every analytic gradient is checked against central finite differences at the
start point and 25 seeded random points in the plot window.

**Why the saddle matters.** At the origin $\nabla f = 0$, so a method that only
reacts to the current gradient has no reason to leave. Started exactly on the
$x$-axis, gradient descent converges to $(0, 0)$ and the gradient norm drops
below $10^{-12}$: it looks finished but is not at a minimum. Nudged $10^{-3}$
off the axis, plain GD grows $y$ by a factor $(1 + 2\eta)$ per step, which at
$\eta = 0.01$ is only $1.02^{100} \approx 7.2\times$ after 100 steps. Momentum
compounds those small pushes and is past $|y| > 1$ in the same 100 steps.
[docs/saddle-point.md](docs/saddle-point.md) works through the saddle numbers in
the results table from the update rules.

**Rastrigin: many local minima.**

$$
f(x, y) = 20 + \left(x^2 - 10\cos 2\pi x\right) + \left(y^2 - 10\cos 2\pi y\right),
\qquad
\frac{\partial f}{\partial x} = 2x + 20\pi \sin 2\pi x
$$

A shallow bowl $x^2 + y^2$ with a cosine egg-crate of amplitude 10 on top. In
each coordinate the derivative vanishes just inside every integer, at
$x \approx \pm 0.995, \pm 1.990, \pm 2.985, \dots$ (the bowl pulls each dimple
slightly toward the origin), and each of those is a local minimum. Inside
$[-5.12, 5.12]^2$ that gives 11 such points per axis, so $11^2 = 121$ minima
of which only the one at the origin is global. A method that follows the local gradient stops in whichever dimple it
starts in or falls into, and the stopping point has $\nabla f = 0$, so it
looks exactly like convergence. The tests find these dimples with Newton's
method and check that each one is a strict local minimum above $f(0,0)$.

**Ackley: a flat outer region.**

$$
f(x, y) = -20\exp\!\left(-0.2\sqrt{\tfrac12(x^2+y^2)}\right)
- \exp\!\left(\tfrac12(\cos 2\pi x + \cos 2\pi y)\right) + e + 20
$$

With $r = \sqrt{\tfrac12(x^2+y^2)}$, the first term's gradient is
$\frac{2e^{-0.2r}}{r}\,(x, y)$: its magnitude is
$2\sqrt2\,e^{-0.2 r}$, which decays exponentially with distance, so far from
the origin the surface is a near-plateau (at $r = 4$ the funnel term is worth
$-20e^{-0.8} \approx -9$ and its slope is only about $1.3$). There the
strongest signal comes from the cosine ripples, whose slope reaches up to about
$6.5$, and those point to the nearest ripple, not to the centre. Near the origin the funnel is steep and
narrow. The gradient of $\sqrt{x^2+y^2}$ is undefined at the origin itself;
the code returns the subgradient $0$ there, which is also where the minimum
is. A good Ackley run needs steps large enough to hop ripples on the plateau
and small enough to settle at the bottom of the funnel, and no fixed $\eta$
does both.

**Booth: the easy quadratic.**

$$
f(x, y) = (x + 2y - 7)^2 + (2x + y - 5)^2,
\qquad
\nabla^2 f = \begin{pmatrix} 10 & 8 \\ 8 & 10 \end{pmatrix}
$$

Booth is a convex quadratic with Hessian eigenvalues 18 (along $(1,1)$) and 2
(along $(1,-1)$), so $\kappa = 9$ and GD is stable for $\eta < 2/18 \approx 0.11$.
There is one minimum, at $(1, 3)$, and no traps. It is here as a control:
any method that struggles on Booth has a bug or a bad learning rate, not a
hard problem.

## Results

Every optimizer was run on every surface with each learning rate in
`{1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.03, 0.1, 0.3, 1}`; each cell shows the best
one. Numbers come from `npm run race` and are copied into
[docs/race-results.md](docs/race-results.md) unedited.

| surface (start) | GD | Momentum | Nesterov | RMSProp | Adam |
| --- | --- | --- | --- | --- | --- |
| Quadratic bowl (-4, 2) | **76** (η=0.1) | **132** (η=0.01) | **45** (η=0.1) | **25** (η=0.3) | **132** (η=0.3) |
| Rosenbrock (-1.2, 1) | — f=5.9e-5 (η=0.001) | **450** (η=0.003) | **1426** (η=0.001) | — f=2.3e-4 (η=0.001) | **679** (η=0.3) |
| Beale (1, 1.5) | **557** (η=0.03) | **405** (η=0.003) | **403** (η=0.003) | — f=5.7e-6 (η=0.0003) | **205** (η=0.3) |
| Saddle point (1, 0.001) | **349** (η=0.01) | **72** (η=0.01) | **68** (η=0.01) | **75** (η=0.01) | **80** (η=0.01) |
| Himmelblau (-0.5, -0.5) | **18** (η=0.01) | **156** (η=0.003) | **18** (η=0.01) | **39** (η=0.1) | **149** (η=1) |
| Rastrigin (2.2, -1.7) | — f=8.0e+0 (η=0.0003) | — f=8.0e+0 (η=0.0001) | — f=8.0e+0 (η=0.0001) | — f=8.0e+0 (η=0.0001) | — f=8.0e+0 (η=0.0001) |
| Ackley (3.3, -2.6) | — f=3.5e+0 (η=0.1) | — f=2.6e+0 (η=0.03) | — f=9.0e+0 (η=0.0003) | — f=2.3e+0 (η=0.3) | — f=1.4e-1 (η=1) |
| Booth (-6, -4) | **14** (η=0.03) | **88** (η=0.003) | **21** (η=0.03) | **14** (η=1) | **93** (η=0.3) |

Steps to f − f* < 0.000001 (saddle: steps until f < -1 at a fixed η=0.01), best η per cell, budget 10000 steps; "—" means the budget ran out, with the final loss shown.


What the numbers say:

- **Bowl.** Tuned RMSProp wins (25 steps): per-coordinate scaling undoes most
  of the 10:1 conditioning. Nesterov (45) beats plain GD (76) at the same
  $\eta = 0.1$.
- **Rosenbrock.** The valley floor is where methods separate. Momentum gets to
  $10^{-6}$ in 450 steps and Adam in 679; plain GD is still at
  $f \approx 5.9\times10^{-5}$ after 10,000 steps, and RMSProp stalls near
  $2.3\times10^{-4}$ because its roughly constant-size steps keep bouncing across
  the floor instead of shrinking.
- **Beale.** Adam is fastest (205) and the momentum methods (~400) beat GD
  (557). RMSProp only converges at a tiny $\eta$ and runs out of budget.
- **Saddle.** At the same $\eta = 0.01$, GD needs 349 steps to get off the
  saddle; every other method needs 68 to 80.
- **Himmelblau.** From $(-0.5, -0.5)$ all five land in the same minimum,
  $(-2.805, 3.131)$; GD and Nesterov get there in 18 steps.
- **Booth.** The easy control: all five converge. GD and tuned RMSProp take
  14 steps and Nesterov 21; momentum (88) and Adam (93) are slower at their
  best $\eta$ on this well-conditioned bowl ($\kappa = 9$).
- **Rastrigin.** From $(2.2, -1.7)$ nobody gets near the global minimum. All
  five finish in the dimple at $(1.990, -1.990)$, where $f = 7.96$. Every
  small $\eta$ lands there (up to $3\times10^{-3}$ for GD and the momentum
  methods, up to $0.1$ for Adam), and larger rates do not find the centre
  either: they bounce across the egg-crate and end higher (GD at
  $\eta = 0.03$ ends at $f = 43.7$). The $\eta$ in each cell is just the
  first of the tied rates. Started inside the central basin at $(0.3, -0.4)$,
  plain GD does reach 0 (a unit test).
- **Ackley.** Nobody reaches $10^{-6}$ in 10,000 steps. Small steps settle in
  the ripple next to the start (Nesterov's best run, $\eta = 3\times10^{-4}$,
  ends at $(2.98, -2.98)$ with $f = 9.0$). Larger steps cross the ripples:
  momentum at $\eta = 0.03$ gets most of the way in and then stops in the
  ripple at $(0, 0.95)$ ($f = 2.58$); GD at $\eta = 0.1$ never settles (its
  last 500 losses range from 0.04 to 4.5, and the table shows wherever step
  10,000 happens to be). Adam at $\eta = 1$ ends closest: it reaches the
  funnel and hovers there (last 500 losses between 0.139 and 0.147), about
  $0.04$ from the origin, without settling to the bottom.

Learning rates matter as much as the algorithm: plain GD with $\eta = 0.1$,
its best rate on the bowl, diverges on Rosenbrock
within 100 steps (a unit test checks this).

## Deployment

The Vercel project is connected to this repo: `vercel.json` builds the Vite
demo with `npm run build`, pushes to `main` deploy to
[optimizer-arena.vercel.app](https://optimizer-arena.vercel.app), and pull
requests get preview deploys. Links like
[`?surface=saddle`](https://optimizer-arena.vercel.app/?surface=saddle) open a
specific surface.

## Quick start

```sh
npm install
npm test
npm run race   # regenerate the results table
npm run dev
```

## Project structure

```
src/surfaces.ts    loss surfaces, analytic gradients, known minima
src/optimizers.ts  GD, momentum, Nesterov, RMSProp, Adam as pure step functions
src/run.ts         run(surface, optimizer) → trajectory, steps to tolerance, divergence
src/race.ts        learning-rate sweep and ranking used by the script and the demo
scripts/race.ts    prints the results table (npm run race)
web/               the 2D race demo (Vite, canvas, no framework)
test/              Vitest suite
```

## License

MIT
