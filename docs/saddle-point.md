# Escaping the saddle x² − y²

The saddle is the one surface where the race numbers can be predicted with a
pencil, and the predictions match the measured table.

## Gradient descent

With $f = x^2 - y^2$ the gradient is $(2x, -2y)$, so one GD step is linear and
the two coordinates decouple:

$$
x_{t+1} = (1 - 2\eta)\, x_t, \qquad y_{t+1} = (1 + 2\eta)\, y_t
$$

$x$ shrinks toward 0 and $y$ grows, but only if it is not exactly 0. Started on
the $x$-axis ($y_0 = 0$), $y$ stays 0 forever and GD converges to the saddle.

From $y_0 = 10^{-3}$ at $\eta = 0.01$, $y$ grows by $1.02$ per step. The race
counts a method as escaped once $f < -1$, which (with $x \approx 0$ by then)
means $|y| > 1$:

$$
10^{-3} \cdot 1.02^{\,k} > 1 \iff k > \frac{\ln 1000}{\ln 1.02} \approx 348.8
$$

so GD escapes at step **349**, exactly the number in the results table (and a
unit test).

## Heavy-ball momentum

Along $y$ the heavy-ball update is $v_{t+1} = \beta v_t + 2\eta\, y_t$,
$y_{t+1} = y_t + v_{t+1}$, a linear map on $(y, v)$:

$$
\begin{pmatrix} y_{t+1} \\ v_{t+1} \end{pmatrix} =
\begin{pmatrix} 1 + 2\eta & \beta \\ 2\eta & \beta \end{pmatrix}
\begin{pmatrix} y_t \\ v_t \end{pmatrix}
$$

Its eigenvalues solve $\lambda^2 - (1 + \beta + 2\eta)\lambda + \beta = 0$. With
$\beta = 0.9$, $\eta = 0.01$ the larger one is

$$
\lambda = \frac{1.92 + \sqrt{1.92^2 - 3.6}}{2} \approx 1.1070
$$

so after a short warm-up $|y|$ grows about 10.7% per step instead of 2%. That
predicts $\ln 1000 / \ln 1.107 \approx 68$ steps; the measured count is **72**
because the velocity starts at zero and takes a few steps to build up.

## Nesterov

Nesterov evaluates the gradient at $y + \beta v$, which gives the matrix
$\begin{pmatrix} 1 + 2\eta & \beta(1 + 2\eta) \\ 2\eta & \beta(1 + 2\eta) \end{pmatrix}$
with dominant eigenvalue $\approx 1.1138$: predicted $\approx 64$ steps,
measured **68**.

## RMSProp and Adam

Both divide by the gradient's running RMS, so the tiny $y$-gradient
$2 \times 10^{-3}$ produces a step about as large as the big $x$-gradient's.
They do not amplify geometrically the way momentum does; their steps along $y$
stay close to $\eta$, so covering a distance of 1 at $\eta = 0.01$ should take
on the order of 100 steps. They escape a bit sooner, in 75 (RMSProp) and 80
(Adam), because their normalisers lag behind a gradient that keeps growing:

- RMSProp's $s$ starts at 0 with no bias correction, so its first steps are
  $\eta/\sqrt{1-\rho} \approx 3.16\,\eta$, and they settle toward $\eta$ only
  slowly (measured $\Delta y / \eta$: 3.16 at step 1, 1.74 at step 10, 1.10 at
  step 79).
- Adam's first step is exactly $\eta$, but $\hat v$ averages over roughly
  $1/(1-\beta_2) = 1000$ steps of smaller past gradients, so
  $\hat m / \sqrt{\hat v}$ climbs above 1 (measured $\Delta y / \eta$: 1.00 at
  step 1, 1.37 at step 40, 1.57 at step 79).

Like GD, all four still need $y_0 \ne 0$: on the exact axis the $y$-gradient is
0, momentum never picks up any $y$-velocity, and every method stalls.
