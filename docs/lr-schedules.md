# Learning-rate schedules, measured

Every table here is `npm run race` output, pasted unedited. The race gives each
optimizer the best peak rate η from `{1e-4, 3e-4, …, 1}`, budget 10,000 steps,
goal f − f* < 1e-6 (saddle: f < −1 at a fixed η = 0.01). A schedule multiplies
η by s(t) over that 10,000-step budget, so the cosine reaches 0 at step 10,000
and the warmup is the first 500 steps. Formulas and derivations are in the
[README](../README.md#learning-rate-schedules); the code is
[src/schedules.ts](../src/schedules.ts).

```sh
npm run race -- --schedule constant|step|cosine|warmup-cosine
npm run race -- --schedule cosine --start-off-lattice
```

## Why two tables per schedule: Lion's lattice

With a constant η and no decay, Lion only ever moves by ±η per coordinate, so
every iterate is on the lattice θ₀ + ηℤ². Several of the standard starts sit a
whole number of grid steps from their minimum: bowl (−4, 2) → (0, 0), Booth
(−6, −4) → (1, 3), Beale (1, 1.5) → (3, 0.5), Rosenbrock (−1.2, 1) → (1, 1).
At the right η, Lion lands on the minimum exactly (f = 0 or ≈ 1e-28), which
says more about the start point than about Lion. `--start-off-lattice` adds
(√2, e)·10⁻³ to every start except the saddle's, which breaks that
coincidence. For the other seven methods the nudge moves smooth-surface
counts only a little (momentum on Rosenbrock 450 → 440, Adam on Booth
93 → 120), but on Rastrigin and Ackley, where the outcome depends on which
ripple a run ends in, it can change the final loss a lot (Adam on Ackley,
f = 0.14 → 8.0, constant rate).

Counts of cells that reach the goal (out of 8 surfaces × 8 optimizers = 64):

| schedule | standard starts | off-lattice starts |
| --- | --- | --- |
| constant | 43 | 43 |
| step decay | 49 | 47 |
| cosine | 50 | 51 |
| warmup + cosine | 48 | 48 |

Lion alone (surfaces reached, of 8): constant 5 → 5, step 6 → 6, cosine 6 → 6,
warmup + cosine 7 → 7. The totals do not move much between the two columns,
but the *step counts* do: Lion on the bowl goes from 4 steps (standard start,
η = 1) to 3,999 (off-lattice, η = 1e-3) under a constant rate.

## Constant

| surface (start) | GD | Momentum | Nesterov | RMSProp | Adam | AdaGrad | AdamW | Lion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Quadratic bowl (-4, 2) | **76** (η=0.1) | **132** (η=0.01) | **45** (η=0.1) | **25** (η=0.3) | **132** (η=0.3) | **47** (η=1) | **124** (η=1) | **4** (η=1) |
| Rosenbrock (-1.2, 1) | — f=5.9e-5 (η=0.001) | **450** (η=0.003) | **1426** (η=0.001) | — f=2.3e-4 (η=0.001) | **679** (η=0.3) | **8491** (η=1) | — f=3.4e-6 (η=0.003) | **290** (η=0.01) |
| Beale (1, 1.5) | **557** (η=0.03) | **405** (η=0.003) | **403** (η=0.003) | — f=5.7e-6 (η=0.0003) | **205** (η=0.3) | **739** (η=1) | **9497** (η=0.03) | **62** (η=0.1) |
| Saddle point (1, 0.001) | **349** (η=0.01) | **72** (η=0.01) | **68** (η=0.01) | **75** (η=0.01) | **80** (η=0.01) | **1440** (η=0.01) | **81** (η=0.01) | **100** (η=0.01) |
| Himmelblau (-0.5, -0.5) | **18** (η=0.01) | **156** (η=0.003) | **18** (η=0.01) | **39** (η=0.1) | **149** (η=1) | **16** (η=1) | **4336** (η=0.3) | — f=4.3e-5 (η=0.001) |
| Rastrigin (2.2, -1.7) | — f=8.0e+0 (η=0.0003) | — f=8.0e+0 (η=0.0001) | — f=8.0e+0 (η=0.0001) | — f=8.0e+0 (η=0.0001) | — f=8.0e+0 (η=0.0001) | — f=8.0e+0 (η=0.1) | — f=8.0e+0 (η=0.003) | — f=8.0e+0 (η=0.001) |
| Ackley (3.3, -2.6) | — f=3.5e+0 (η=0.1) | — f=2.6e+0 (η=0.03) | — f=9.0e+0 (η=0.0003) | — f=2.3e+0 (η=0.3) | — f=1.4e-1 (η=1) | — f=1.9e-2 (η=1) | — f=1.4e-1 (η=1) | — f=9.0e+0 (η=0.0001) |
| Booth (-6, -4) | **14** (η=0.03) | **88** (η=0.003) | **21** (η=0.03) | **14** (η=1) | **93** (η=0.3) | **163** (η=1) | **8910** (η=0.3) | **7** (η=1) |

Off-lattice starts:

| surface (start) | GD | Momentum | Nesterov | RMSProp | Adam | AdaGrad | AdamW | Lion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Quadratic bowl (-4, 2) | **76** (η=0.1) | **132** (η=0.01) | **45** (η=0.1) | **25** (η=0.3) | **132** (η=0.3) | **47** (η=1) | **124** (η=1) | **3999** (η=0.001) |
| Rosenbrock (-1.2, 1) | — f=5.9e-5 (η=0.001) | **440** (η=0.003) | **1427** (η=0.001) | — f=2.3e-4 (η=0.001) | **678** (η=0.3) | **8658** (η=1) | — f=3.3e-6 (η=0.003) | **9376** (η=0.0003) |
| Beale (1, 1.5) | **557** (η=0.03) | **399** (η=0.003) | **402** (η=0.003) | — f=9.0e-6 (η=0.001) | **206** (η=0.3) | **746** (η=1) | **8733** (η=0.1) | **3211** (η=0.001) |
| Saddle point (1, 0.001) | **349** (η=0.01) | **72** (η=0.01) | **68** (η=0.01) | **75** (η=0.01) | **80** (η=0.01) | **1440** (η=0.01) | **81** (η=0.01) | **100** (η=0.01) |
| Himmelblau (-0.5, -0.5) | **18** (η=0.01) | **156** (η=0.003) | **18** (η=0.01) | **39** (η=0.1) | **149** (η=1) | **16** (η=1) | **2205** (η=1) | — f=2.3e-5 (η=0.001) |
| Rastrigin (2.2, -1.7) | — f=8.0e+0 (η=0.0003) | — f=8.0e+0 (η=0.0001) | — f=8.0e+0 (η=0.0001) | — f=8.0e+0 (η=0.0001) | — f=5.3e+0 (η=1) | — f=8.0e+0 (η=0.03) | — f=8.0e+0 (η=0.003) | — f=8.0e+0 (η=0.0001) |
| Ackley (3.3, -2.6) | — f=3.3e+0 (η=0.1) | — f=6.9e+0 (η=0.03) | — f=9.0e+0 (η=0.0001) | — f=2.3e+0 (η=0.3) | — f=8.0e+0 (η=1) | — f=1.9e-2 (η=1) | — f=2.6e+0 (η=1) | — f=9.0e+0 (η=0.0001) |
| Booth (-6, -4) | **14** (η=0.03) | **88** (η=0.003) | **21** (η=0.03) | **14** (η=1) | **120** (η=0.3) | **164** (η=1) | **8909** (η=0.3) | **6998** (η=0.001) |

## Step decay (×0.5 every 1,000 steps)

| surface (start) | GD | Momentum | Nesterov | RMSProp | Adam | AdaGrad | AdamW | Lion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Quadratic bowl (-4, 2) | **76** (η=0.1) | **132** (η=0.01) | **45** (η=0.1) | **25** (η=0.3) | **132** (η=0.3) | **47** (η=1) | **124** (η=1) | **4** (η=1) |
| Rosenbrock (-1.2, 1) | — f=2.5e-3 (η=0.003) | **450** (η=0.003) | **1859** (η=0.001) | **6001** (η=0.003) | **679** (η=0.3) | — f=5.9e-3 (η=1) | — f=1.2e-6 (η=0.1) | **290** (η=0.01) |
| Beale (1, 1.5) | **557** (η=0.03) | **405** (η=0.003) | **403** (η=0.003) | **1008** (η=0.01) | **205** (η=0.3) | **739** (η=1) | **8378** (η=0.3) | **62** (η=0.1) |
| Saddle point (1, 0.001) | **349** (η=0.01) | **72** (η=0.01) | **68** (η=0.01) | **75** (η=0.01) | **80** (η=0.01) | **2026** (η=0.01) | **81** (η=0.01) | **100** (η=0.01) |
| Himmelblau (-0.5, -0.5) | **18** (η=0.01) | **156** (η=0.003) | **18** (η=0.01) | **39** (η=0.1) | **149** (η=1) | **16** (η=1) | **5730** (η=1) | **4002** (η=0.01) |
| Rastrigin (2.2, -1.7) | — f=4.0e+0 (η=0.03) | **5112** (η=0.1) | — f=2.0e+0 (η=0.1) | **3001** (η=1) | — f=8.0e+0 (η=0.003) | — f=8.0e+0 (η=0.1) | — f=8.0e+0 (η=0.1) | — f=8.0e+0 (η=0.0003) |
| Ackley (3.3, -2.6) | **3001** (η=0.1) | — f=2.6e+0 (η=0.03) | — f=9.0e+0 (η=0.0003) | **2001** (η=0.3) | — f=7.1e-4 (η=1) | — f=4.5e-5 (η=1) | — f=3.0e-4 (η=1) | — f=7.2e+0 (η=0.3) |
| Booth (-6, -4) | **14** (η=0.03) | **88** (η=0.003) | **21** (η=0.03) | **14** (η=1) | **93** (η=0.3) | **163** (η=1) | **8384** (η=1) | **7** (η=1) |

Off-lattice starts:

| surface (start) | GD | Momentum | Nesterov | RMSProp | Adam | AdaGrad | AdamW | Lion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Quadratic bowl (-4, 2) | **76** (η=0.1) | **132** (η=0.01) | **45** (η=0.1) | **25** (η=0.3) | **132** (η=0.3) | **47** (η=1) | **124** (η=1) | **1665** (η=0.003) |
| Rosenbrock (-1.2, 1) | — f=2.5e-3 (η=0.003) | **440** (η=0.003) | **1860** (η=0.001) | **6001** (η=0.003) | **678** (η=0.3) | — f=6.0e-3 (η=1) | — f=1.1e-6 (η=0.1) | **3483** (η=0.003) |
| Beale (1, 1.5) | **557** (η=0.03) | **399** (η=0.003) | **402** (η=0.003) | **1008** (η=0.01) | **206** (η=0.3) | **746** (η=1) | **8393** (η=0.3) | **2001** (η=0.003) |
| Saddle point (1, 0.001) | **349** (η=0.01) | **72** (η=0.01) | **68** (η=0.01) | **75** (η=0.01) | **80** (η=0.01) | **2026** (η=0.01) | **81** (η=0.01) | **100** (η=0.01) |
| Himmelblau (-0.5, -0.5) | **18** (η=0.01) | **156** (η=0.003) | **18** (η=0.01) | **39** (η=0.1) | **149** (η=1) | **16** (η=1) | **5730** (η=1) | **4001** (η=0.003) |
| Rastrigin (2.2, -1.7) | — f=9.9e-1 (η=0.03) | — f=2.0e+0 (η=0.03) | — f=8.0e+0 (η=0.0001) | — f=1.0e+0 (η=1) | — f=5.0e+0 (η=1) | — f=8.0e+0 (η=0.03) | — f=8.0e+0 (η=0.1) | — f=8.0e+0 (η=0.0003) |
| Ackley (3.3, -2.6) | **3001** (η=0.1) | — f=6.9e+0 (η=0.03) | — f=9.0e+0 (η=0.0001) | **2001** (η=0.3) | — f=8.0e+0 (η=1) | — f=4.5e-5 (η=1) | — f=2.6e+0 (η=1) | — f=9.0e+0 (η=0.0003) |
| Booth (-6, -4) | **14** (η=0.03) | **88** (η=0.003) | **21** (η=0.03) | **14** (η=1) | **120** (η=0.3) | **164** (η=1) | **8383** (η=1) | **4002** (η=0.01) |

## Cosine annealing

| surface (start) | GD | Momentum | Nesterov | RMSProp | Adam | AdaGrad | AdamW | Lion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Quadratic bowl (-4, 2) | **76** (η=0.1) | **132** (η=0.01) | **45** (η=0.1) | **25** (η=0.3) | **132** (η=0.3) | **47** (η=1) | **124** (η=1) | **4** (η=1) |
| Rosenbrock (-1.2, 1) | — f=4.5e-5 (η=0.003) | **451** (η=0.003) | **1452** (η=0.001) | **8370** (η=0.001) | **682** (η=0.3) | — f=2.7e-5 (η=1) | — f=1.3e-6 (η=0.03) | **6276** (η=0.001) |
| Beale (1, 1.5) | **558** (η=0.03) | **405** (η=0.003) | **404** (η=0.003) | **6076** (η=0.001) | **205** (η=0.3) | **743** (η=1) | **8307** (η=0.3) | **62** (η=0.1) |
| Saddle point (1, 0.001) | **350** (η=0.01) | **72** (η=0.01) | **68** (η=0.01) | **75** (η=0.01) | **80** (η=0.01) | **1474** (η=0.01) | **81** (η=0.01) | **100** (η=0.01) |
| Himmelblau (-0.5, -0.5) | **18** (η=0.01) | **156** (η=0.003) | **18** (η=0.01) | **39** (η=0.1) | **149** (η=1) | **16** (η=1) | **5917** (η=0.3) | **4452** (η=0.001) |
| Rastrigin (2.2, -1.7) | **7406** (η=0.03) | — f=4.0e+0 (η=0.03) | — f=5.0e+0 (η=0.01) | — f=9.9e-1 (η=1) | — f=8.0e+0 (η=0.001) | — f=8.0e+0 (η=0.03) | — f=8.0e+0 (η=0.03) | — f=9.9e-1 (η=1) |
| Ackley (3.3, -2.6) | **9991** (η=0.1) | — f=2.6e+0 (η=0.03) | — f=9.0e+0 (η=0.0001) | **9993** (η=0.3) | **9981** (η=1) | — f=3.6e+0 (η=1) | **9981** (η=1) | — f=8.0e+0 (η=0.3) |
| Booth (-6, -4) | **14** (η=0.03) | **88** (η=0.003) | **21** (η=0.03) | **14** (η=1) | **93** (η=0.3) | **163** (η=1) | **8373** (η=1) | **7** (η=1) |

Off-lattice starts:

| surface (start) | GD | Momentum | Nesterov | RMSProp | Adam | AdaGrad | AdamW | Lion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Quadratic bowl (-4, 2) | **76** (η=0.1) | **132** (η=0.01) | **45** (η=0.1) | **25** (η=0.3) | **132** (η=0.3) | **47** (η=1) | **124** (η=1) | **4608** (η=0.003) |
| Rosenbrock (-1.2, 1) | — f=4.5e-5 (η=0.003) | **441** (η=0.003) | **1452** (η=0.001) | **8370** (η=0.001) | **680** (η=0.3) | — f=2.8e-5 (η=1) | — f=1.2e-6 (η=0.03) | **7755** (η=0.001) |
| Beale (1, 1.5) | **558** (η=0.03) | **399** (η=0.003) | **403** (η=0.003) | **6076** (η=0.001) | **206** (η=0.3) | **749** (η=1) | **8323** (η=0.3) | **1629** (η=0.003) |
| Saddle point (1, 0.001) | **350** (η=0.01) | **72** (η=0.01) | **68** (η=0.01) | **75** (η=0.01) | **80** (η=0.01) | **1474** (η=0.01) | **81** (η=0.01) | **100** (η=0.01) |
| Himmelblau (-0.5, -0.5) | **18** (η=0.01) | **156** (η=0.003) | **18** (η=0.01) | **39** (η=0.1) | **149** (η=1) | **16** (η=1) | **2411** (η=1) | **6682** (η=0.001) |
| Rastrigin (2.2, -1.7) | **7406** (η=0.03) | **8092** (η=0.1) | — f=9.9e-1 (η=0.03) | — f=9.9e-1 (η=1) | — f=8.0e+0 (η=0.001) | — f=8.0e+0 (η=0.01) | — f=8.0e+0 (η=0.03) | — f=8.0e+0 (η=0.0003) |
| Ackley (3.3, -2.6) | **9991** (η=0.1) | — f=6.9e+0 (η=0.03) | — f=9.0e+0 (η=0.0001) | **9993** (η=0.3) | — f=8.0e+0 (η=1) | **9953** (η=1) | **9981** (η=1) | — f=8.0e+0 (η=0.3) |
| Booth (-6, -4) | **14** (η=0.03) | **88** (η=0.003) | **21** (η=0.03) | **14** (η=1) | **120** (η=0.3) | **164** (η=1) | **8372** (η=1) | **2450** (η=0.003) |

## Linear warmup (500 steps) + cosine

| surface (start) | GD | Momentum | Nesterov | RMSProp | Adam | AdaGrad | AdamW | Lion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Quadratic bowl (-4, 2) | **86** (η=1) | **132** (η=1) | **63** (η=1) | **90** (η=1) | **173** (η=1) | **571** (η=1) | **173** (η=1) | **4985** (η=0.001) |
| Rosenbrock (-1.2, 1) | — f=5.8e-5 (η=0.003) | **699** (η=0.003) | **1719** (η=0.001) | **8463** (η=0.001) | **942** (η=0.3) | — f=9.1e-6 (η=1) | — f=6.8e-6 (η=0.1) | **3392** (η=0.001) |
| Beale (1, 1.5) | **797** (η=0.03) | **127** (η=0.1) | **117** (η=0.1) | **6272** (η=0.001) | **316** (η=1) | **3471** (η=1) | — f=1.4e-6 (η=1) | **3756** (η=0.001) |
| Saddle point (1, 0.001) | **598** (η=0.01) | **224** (η=0.01) | **221** (η=0.01) | **303** (η=0.01) | **241** (η=0.01) | **2137** (η=0.01) | **241** (η=0.01) | **316** (η=0.01) |
| Himmelblau (-0.5, -0.5) | **23** (η=0.3) | **150** (η=0.1) | **21** (η=0.3) | **68** (η=1) | **191** (η=1) | **243** (η=1) | **6746** (η=0.3) | **4632** (η=0.001) |
| Rastrigin (2.2, -1.7) | **7538** (η=0.03) | **8996** (η=0.3) | — f=9.9e-1 (η=0.01) | — f=9.9e-1 (η=1) | — f=8.0e+0 (η=0.003) | — f=8.0e+0 (η=0.1) | — f=8.0e+0 (η=0.1) | — f=4.0e+0 (η=1) |
| Ackley (3.3, -2.6) | **9991** (η=0.1) | — f=9.0e+0 (η=0.0001) | — f=3.6e+0 (η=0.1) | **9993** (η=0.3) | — f=9.0e+0 (η=0.0003) | — f=9.0e+0 (η=0.03) | — f=7.2e+0 (η=1) | **9976** (η=0.3) |
| Booth (-6, -4) | **20** (η=1) | **77** (η=0.03) | **24** (η=1) | **117** (η=1) | **195** (η=1) | **1138** (η=1) | — f=3.0e-6 (η=1) | **2674** (η=0.003) |

Off-lattice starts:

| surface (start) | GD | Momentum | Nesterov | RMSProp | Adam | AdaGrad | AdamW | Lion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Quadratic bowl (-4, 2) | **86** (η=1) | **132** (η=1) | **63** (η=1) | **90** (η=1) | **174** (η=1) | **571** (η=1) | **173** (η=1) | **4982** (η=0.001) |
| Rosenbrock (-1.2, 1) | — f=5.8e-5 (η=0.003) | **699** (η=0.003) | **1719** (η=0.001) | **8462** (η=0.001) | **933** (η=0.3) | — f=7.5e-6 (η=1) | — f=6.5e-6 (η=0.1) | **6896** (η=0.001) |
| Beale (1, 1.5) | **797** (η=0.03) | **127** (η=0.1) | **117** (η=0.1) | **6272** (η=0.001) | **317** (η=1) | **3512** (η=1) | — f=1.4e-6 (η=1) | **1445** (η=0.003) |
| Saddle point (1, 0.001) | **598** (η=0.01) | **224** (η=0.01) | **221** (η=0.01) | **303** (η=0.01) | **241** (η=0.01) | **2137** (η=0.01) | **241** (η=0.01) | **316** (η=0.01) |
| Himmelblau (-0.5, -0.5) | **23** (η=0.3) | **150** (η=0.1) | **21** (η=0.3) | **68** (η=1) | **191** (η=1) | **243** (η=1) | **6747** (η=0.3) | **4630** (η=0.001) |
| Rastrigin (2.2, -1.7) | **7538** (η=0.03) | — f=8.0e+0 (η=0.0001) | **9387** (η=0.3) | — f=9.9e-1 (η=1) | — f=8.0e+0 (η=0.003) | — f=8.0e+0 (η=0.1) | — f=8.0e+0 (η=0.1) | — f=2.0e+0 (η=1) |
| Ackley (3.3, -2.6) | **9991** (η=0.1) | — f=9.0e+0 (η=0.0001) | — f=9.0e+0 (η=0.0003) | **9993** (η=0.3) | — f=7.2e+0 (η=1) | — f=9.0e+0 (η=0.03) | — f=8.0e+0 (η=1) | **9606** (η=1) |
| Booth (-6, -4) | **20** (η=1) | **77** (η=0.03) | **24** (η=1) | **117** (η=1) | **195** (η=1) | **1143** (η=1) | — f=2.9e-6 (η=1) | **6122** (η=0.01) |

## What changes, and why

- **Runs that finish early barely notice.** GD, Nesterov, RMSProp and AdaGrad
  on the bowl give the same counts under constant, step and cosine (76, 45,
  25, 47): they are done in under 100 steps, when cosine is still at
  s(76) = ½(1 + cos(π·76/10⁴)) ≈ 0.99986 and step decay has not cut once.
  A schedule only matters for runs that last a good fraction of the budget.
- **Decay fixes the methods that rattle.** RMSProp on Rosenbrock stalls at
  f = 2.3e-4 with a constant rate (its steps are ≈ η per coordinate, so it
  bounces across the valley floor); with step decay it converges in 6,001
  steps and with cosine in 8,370. On Beale it goes from "not reached" to
  1,008 (step) and 6,076 (cosine). Lion off the lattice: Himmelblau goes from
  f = 2.3e-5 (constant) to 4,001 steps (step) and 6,682 (cosine).
- **Annealing can find the global minimum.** On Rastrigin every constant-rate
  run ends in the local dimple at f = 7.96. With cosine, GD at η = 0.03
  reaches the global minimum in 7,406 steps: the large early steps carry it
  across the egg-crate, and the decay lets it settle in whichever basin it is
  in when the rate gets small. Under warmup + cosine GD (7,538) and Nesterov
  (9,387) both get there.
- **Ackley is reached only at the very end.** GD (9,991), RMSProp (9,993) and,
  on the standard start, Adam and AdamW (9,981) cross 1e-6 only in the last
  hundred steps of the cosine, where η_t is tiny: they hovered over the
  funnel all run, and the decay is what finally lets them sit in it.
- **Warmup buys larger peak rates.** With warmup + cosine, momentum and
  Nesterov on Beale win at η = 0.1 (127 and 117 steps) where the constant
  race's best was η = 0.003 (405 and 403): ramping up over 500 steps gets
  them into the valley before the rate is large enough to throw them out. On
  Booth GD wins at a peak η = 1 in 20 steps, nine times past Booth's
  2/λ_max = 1/9 stability limit (Hessian eigenvalues 18 and 2), because it
  converges while η_t = (t+1)/500 is still below 0.04, well inside it. The cost is a slower start, and AdaGrad pays most: on the bowl it goes
  from 47 steps to 571, because G sums the large gradients at the start
  point while η_t is still tiny, and by the time the rate has ramped up √G
  is already large and never shrinks.
- **AdamW still pays for its decay.** The decay term is scaled by η_t too
  (θ ← θ − η_t(… + λθ)), so annealing shrinks the pull toward the origin, but
  not to zero before the end: on Booth it takes over 8,000 steps under every
  schedule (and misses under warmup + cosine, f = 3.0e-6), and under warmup + cosine AdamW reaches the goal on only 3 of 8
  surfaces.
