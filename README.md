# circular_regression_toolbox

**A MATLAB toolbox for regression of angles** (phase, direction, time-of-day,
compass bearing — anything whose values live on a circle rather than on a
number line).

## Why this is needed

If your outcome is an angle and you reach for `fitlm` / `fitlme` / `fitglm`,
you will get a wrong answer. The reason: linear models treat the response as
a real number, so they think 179° and –179° are 358° apart when they are
actually 2° apart. Every quantity that depends on residual size — the
coefficient estimates, the standard errors, the p-values, the R² — gets
distorted by the wrap-around. The bigger the angular spread of your data,
the worse the distortion gets.

The right approach is a **circular regression**: the response is modeled by a
distribution on the circle (the von Mises distribution, the circular analogue
of the Normal), residual size is measured by `1 − cos(y − ŷ)` rather than
`(y − ŷ)²`, and the wrap-around is handled correctly by construction. There
are well-established circular-regression methods in the statistics literature,
and several mature R packages implement them (`brms`, `bpnreg`, `lme4`'s
sin/cos workaround). **MATLAB has no native support for any of this.** A
MATLAB-based analysis that needs to regress an angle on a covariate is left
with three options: (a) misuse `fitlme` and silently bias the result,
(b) bridge into R for every fit, or (c) hand-code a von Mises GLMM from
scratch. None of those scale.

## What this toolbox provides

A single MATLAB entry point — `circ_fit(table, formula, backend, options)` —
that runs a proper circular regression and returns a uniform output struct
regardless of which underlying estimator did the work. Four estimator
backends are available:

| Backend | Estimator | Where it runs |
|---|---|---|
| `'fitcirc_lme'` (default) | Native EM von Mises GLMM with subject random intercept and cluster-robust standard errors | MATLAB only |
| `'brms'` | Bayesian von Mises GLMM | R (`brms` + Stan) |
| `'lme4'` | Two parallel sin/cos linear mixed models combined via `atan2` | R (`lme4`) |
| `'bpnreg'` | Bayesian projected-normal mixed model | R (`bpnreg`) |

The reason for shipping four is that they make different trade-offs (the
**Backend chooser** table below lists them side by side). If a result is
robust, all four backends will agree; if not, you want to know that.

What the toolbox adds on top of the bare estimators:

- **One formula grammar.** The same Wilkinson string
  (`y ~ Age^2 + sex + (1|Subj_ID)`) drives every backend. No reformatting
  between MATLAB and R conventions.
- **One result struct.** Every backend returns the same field names
  (`Coefficients`, `AgeEffect.pValue`, `GOF.R2_circ_marginal`,
  `GOF.R2_circ_conditional`, `Trajectory`, ...). Downstream code that reads
  the result does not change when you swap backends.
- **An omnibus age-effect test.** A single p-value answering "does age matter
  for this response, at all?" — computed as a joint Wald across every
  age-involving term (polynomial main effects + every age × covariate
  interaction). The right test for the "is there an age effect?" sentence
  in a paper.
- **Order selection.** The polynomial order in the predictor is chosen by
  step-up likelihood-ratio test (or LOO / WAIC for the Bayesian backends),
  capped at a user-set maximum.
- **Marginal and conditional R²** on the angle scale, comparable across
  backends, computed via a Nakagawa–Schielzeth ICC adjustment adapted to the
  von Mises GLMM. The tutorial below explains what each one tells you.
- **Plotting that handles the ±π seam by construction.** `plot_circ_fit`
  draws the trajectory at its value plus copies at ±2π so the curve never
  visibly jumps across the wrap.
- **`fitcirc_lme` itself** — the native MATLAB von Mises GLMM that the
  default backend wraps. Closed-form exact E-step (no Laplace approximation),
  monotone-EM update guarded against likelihood decrease, warm-start across
  polynomial orders so the order-selection LRT is well posed by construction,
  cluster-robust sandwich standard errors keyed on the random-effect grouping
  variable, and a halfcauchy prior on the subject random-phase concentration
  that keeps the marginal log-likelihood finite when subjects collapse onto a
  shared baseline.

For a hands-on introduction with simulated data, see the [Tutorial
section below](#tutorial-simulate-fit-recover) (or run [`tutorial.m`](tutorial.m)
in MATLAB).

## The four methods

The four backends fall into two families based on how they represent the
angular response. The first family — `fitcirc_lme` and `brms` — fits a
**single circular distribution** (the von Mises) directly on the angle.
The second — `lme4` and `bpnreg` — fits **two coupled real-valued models**
(sin/cos or projected-normal x/y) and combines them into an angle at the
end. That structural choice is the main thing that drives when one
backend is better than another, so it's worth understanding before
picking one.

### `fitcirc_lme` — native von Mises mixed-effects model

[**Full technical reference: `docs/methods/fitcirc_lme.md`**](docs/methods/fitcirc_lme.md) — exact model, EM derivation, monotonicity guarantees, cluster-robust sandwich SE construction, GOF formulas.

**What the model says.** Each subject has a personal baseline angle, and
each of their observations sits a random distance away from the
population trend plus their personal offset. Both the per-observation
noise and the per-subject offset follow a **von Mises** distribution —
the circular analogue of the Normal, parameterized by a concentration
`κ` (large `κ` = tightly clustered angles, small `κ` = wide spread). In
symbols:

$$
y_{ij} \sim \text{vonMises}(X_{ij}\beta + \phi_i,\ \kappa),\quad
\phi_i \sim \text{vonMises}(0,\ \kappa_\phi).
$$

**How it's fit.** EM algorithm. The E-step computes, in closed form,
each subject's posterior offset distribution given the data and the
current parameters; no Laplace or Gaussian approximation. The M-step
updates `β` by iteratively reweighted least squares on the angular
residuals (each observation weighted by how confident we are of its
subject's offset), and updates the two concentrations from circular
mean resultant lengths. Each step is backtracked so the marginal
log-likelihood is **monotone non-decreasing** — successive polynomial
orders are warm-started from the previous order's converged solution,
so `LL(order k) ≥ LL(order k−1)` by construction and the order-selection
LRT is well posed.

**Strengths.** Pure MATLAB, no R dependency; closed-form E-step; honest
cluster-robust (Liang–Zeger sandwich) standard errors keyed on the
random-effect grouping variable; small-sample F-correction; fast (one
fit is seconds, not minutes).

**Limits.** Single `(1|group)` random intercept only — no random slopes,
no crossed grouping factors. The population trend is a single circular
point (a von Mises mean), which **cannot sweep more than one revolution
of the response**; for trajectories that wrap a full 2π over the
predictor range, use `lme4` or `bpnreg` instead.

**Paper reporting.** A defensible methods sentence:

> Circular features were modeled with a von Mises mixed-effects
> regression (`fitcirc_lme`) with fixed effects [list them] and a per-
> subject random intercept. Polynomial order in [predictor] was chosen
> by step-up likelihood-ratio test (α = 0.05, max order [k]). Standard
> errors are cluster-robust (Liang & Zeger 1986 sandwich) keyed on
> subject, with a small-sample correction. Single-coefficient tests use
> Student's *t*; the omnibus age-effect test is a joint Wald across all
> age-involving coefficients (polynomial main effects plus every age ×
> covariate interaction).

What to put in the results table per fit: the **selected polynomial
order**, the **omnibus age-effect statistic** (F, df, p), the **per-
coefficient** estimates with cluster-robust SEs and *t*-test p-values,
and both R² flavors — **R²ₘ (marginal)** for "how well does the
population trend alone predict?" and **R²_c (conditional)** for "how
well does it predict if we also know the subject?"

### `brms` — Bayesian von Mises mixed-effects model (R + Stan)

[**Full technical reference: `docs/methods/brms.md`**](docs/methods/brms.md) — model, priors, NUTS sampler, convergence diagnostics, LOO order selection, omnibus age-effect statistic.

**What the model says.** The same von Mises GLMM as `fitcirc_lme`, but
fit in a Bayesian framework with weakly-informative priors on every
parameter. The link function is `tan_half`, which maps the real-valued
linear predictor onto an angle in (−π, π].

**How it's fit.** Hamiltonian Monte Carlo via Stan. Order selection
uses **leave-one-out cross-validation** (`loo` ELPD differences;
accept the larger order when `elpd_diff > 2 × se_diff`), with the
classical LRT reported alongside for cross-checking. Uncertainty is
the full joint posterior — every quantity (trajectory, coefficient,
GOF) comes with a 95% credible interval drawn from the MCMC samples.

**Strengths.** Posterior-based inference, so uncertainty propagation
into derived quantities (predictions, differences between trajectories,
etc.) is principled. LOO + LRT side by side gives a sanity check on
order selection. Robust to small samples in a way frequentist Wald is
not.

**Limits.** Same `tan_half` link bound — **cannot sweep more than one
revolution**. Slow: one fit is several minutes. Requires R + the `brms`
package + a working Stan toolchain, so adds an external dependency.

**Paper reporting.** A defensible methods sentence:

> We fit a Bayesian von Mises mixed-effects regression with `brms`
> (Stan back end, `tan_half` link, weakly-informative defaults).
> Polynomial order in [predictor] was chosen by leave-one-out cross-
> validation (`loo`), accepting the larger order when the ELPD
> difference exceeded twice its standard error; the classical
> likelihood-ratio test gave the same selection. Each parameter is
> reported as the posterior median with a 95% credible interval.

Per fit, report: **selected order**; **posterior median + 95% CrI**
for each coefficient (and for the omnibus age effect, the posterior
probability that all age-involving coefficients are simultaneously
zero, or equivalently the LOO ELPD against the null); **R̂** for each
sampled parameter (should be < 1.01); and the **R²_circ** computed on
posterior-mean predictions.

### `lme4` — sin/cos parallel linear mixed models (R)

[**Full technical reference: `docs/methods/lme4.md`**](docs/methods/lme4.md) — model, atan2 trajectory reconstruction, bootMer parametric bootstrap band, Bonferroni-union joint LRT, where the two-stage decoupling introduces approximation.

**What the model says.** This one is not a single circular model; it is
**two ordinary linear mixed models stacked in parallel**, one on the
sine of the response and one on the cosine. Each model is fit
independently by REML through `lme4::lmer`. The angular trajectory is
reconstructed at evaluation time as `atan2(ŝin, ĉos)`.

**How it's fit.** Two separate `lme4` fits. Order selection uses a
combined sin+cos LRT: a polynomial order is accepted if either
component model accepts it (in practice this is the more conservative
of the two LRT p-values via a Bonferroni union). Uncertainty in the
trajectory comes from `bootMer` — a parametric bootstrap of the joint
sin/cos predictions, recombined through `atan2` per bootstrap replicate
so the band on the reconstructed angle is honest about both components'
uncertainty at once.

**Strengths.** The reconstructed trajectory `atan2(ŝin, ĉos)` **can
sweep a full 2π revolution** of the predictor — useful if the angle
genuinely makes more than a half-circle excursion (e.g. an oscillation
that drifts a complete cycle across age). Uses well-tested `lme4`
machinery; frequentist; supports `lme4`'s full random-effects grammar
(random slopes, crossed factors, etc.).

**Limits.** The two-stage decoupling is the elephant in the room: the
sin and cos parts of an angle are constrained (they live on the unit
circle together), so fitting them as two independent Gaussian models
is structurally wrong. Standard errors and tests are approximate —
fine for exploratory work, less defensible for a strict null result.
The Bonferroni-union joint test is **conservative** (you may miss real
effects).

**Paper reporting.** A defensible methods sentence:

> The sine and cosine of the angle response were fit as parallel
> linear mixed-effects models in `lme4`, with fixed effects [list them]
> and a per-subject random intercept on each component. The combined
> trajectory was reconstructed as `atan2(ŝin, ĉos)` with a 95%
> trajectory band from a parametric bootstrap (`bootMer`). The omnibus
> polynomial-order test is the more conservative of the two component
> LRTs (Bonferroni union). Because the sin/cos pair are treated as
> independent linear responses, reported standard errors are
> approximate; we use this backend as a sensitivity check against
> [primary backend] and report disagreement when it occurs.

Per fit, report: **selected order**, the **two LRT p-values** that
went into the Bonferroni union, the **trajectory bootstrap CI**, and
the two component **R²**s (one each for sin and cos).

### `bpnreg` — Bayesian projected-normal mixed model (R)

[**Full technical reference: `docs/methods/bpnreg.md`**](docs/methods/bpnreg.md) — projected-normal latent model, Gibbs sampler with data augmentation, WAIC order selection, why this differs from the two-stage sin/cos approach.

**What the model says.** An angle is treated as the **angle of a 2-D
Gaussian latent vector projected onto the unit circle**. The latent
`(x, y)` Gaussian has fixed effects and per-subject random intercepts
on each component; the observed angle is `atan2(y, x)`. This is a
proper joint model (unlike sin/cos parallel) because the two components
share a covariance structure that's estimated from the data.

**How it's fit.** MCMC, using `bpnreg::bpnme`. Order selection is by
**WAIC**. Uncertainty is the full posterior, evaluated through the
projection step at every MCMC draw so the band on the reconstructed
angle is honest.

**Strengths.** Like `lme4`, the reconstructed angle **can sweep a full
2π revolution** — the latent Gaussian carries no wrap-around constraint.
Unlike `lme4`, the model is a proper joint likelihood. Bayesian
inference. The projected-normal family has been argued in the circular-
stats literature to be a better physical model than the von Mises for
some classes of angles (anything generated as an angle of a vector).

**Limits.** Requires R + `bpnreg`. WAIC is the only order-selection
criterion exposed (no closed-form Wald test for "is there an age
effect?"); the `AgeEffect.pValue` returned by the toolbox is **derived
from WAIC differences**, not a classical p, so it should be reported as
such. Slow.

**Paper reporting.** A defensible methods sentence:

> We fit a Bayesian projected-normal mixed-effects regression with
> `bpnreg` (latent bivariate Gaussian, projected to the unit circle
> per row). Polynomial order in [predictor] was chosen by WAIC. Each
> parameter is reported as the posterior median with a 95% credible
> interval. The omnibus age-effect statistic is a WAIC-based
> probability that the larger model is preferred over the no-age
> model, not a classical p-value.

Per fit, report: **selected order**; **posterior median + 95% CrI** for
each coefficient and for the trajectory; **WAIC values** for each
polynomial order considered; the per-component (sin/cos) **R²**s on
posterior-mean predictions.

## Which method to use

A quick three-question chooser:

1. **Does your angular trajectory plausibly wrap more than one
   revolution across the predictor range?** (Imagine: would the
   "true" curve, if you could see it, swing more than 180° from one end
   of the predictor to the other?)
   - **No** → `fitcirc_lme` (default) or `brms`. The von Mises family
     gives the right likelihood for a half-circle-or-less trajectory
     and the SEs / posteriors are honest.
   - **Yes** → `lme4` or `bpnreg`. The von Mises mean is a single
     circular point; it physically cannot represent a trajectory that
     wraps further than one revolution. Switch to a two-component
     representation.

2. **Frequentist or Bayesian inference?**
   - **Frequentist** → `fitcirc_lme` (the primary recommendation) for
     ≤ one-revolution trajectories, or `lme4` for full-revolution.
   - **Bayesian** → `brms` for ≤ one-revolution, or `bpnreg` for
     full-revolution.

3. **How important is fit speed?**
   - **Fast** → `fitcirc_lme` (seconds per fit). Useful when you have
     dozens of features × clusters to fit.
   - **OK with minutes** → any backend.

The combination matrix:

|                           | ≤ one revolution | Full revolution |
|---------------------------|------------------|-----------------|
| **Frequentist**           | `fitcirc_lme`    | `lme4`          |
| **Bayesian**              | `brms`           | `bpnreg`        |

### How to report this in a paper

Three patterns, depending on whether the choice of backend is part of
the contribution or just a tool:

**Pattern A — single backend (most papers).** Pick one based on the
chooser above. Use the methods sentence template from that backend's
section. Add **one sentence** stating that the choice was based on the
expected angular range of the trajectory and (if relevant) the
frequentist-vs-Bayesian framing of the inference. Example:

> Because preferred-phase trajectories were not expected to span more
> than half a circle across the lifespan, we used the von Mises mixed-
> effects backend (`fitcirc_lme`) as the primary estimator.

**Pattern B — robustness check (recommended when the result matters).**
Fit two backends (typically the primary + one cross-check). Report
**both** trajectories and **both** age-effect statistics. State that the
inferences agreed if they did, and quantify the difference if they did
not. Example:

> The age effect was significant under the primary von Mises mixed-
> effects fit (joint Wald F(3, 838) = 12.4, p < .001). To check that
> this conclusion did not depend on the von Mises likelihood
> assumption, we refit with a Bayesian projected-normal model
> (`bpnreg`); the posterior probability of any age effect (WAIC-based)
> exceeded 0.99 and the posterior median trajectory differed from the
> frequentist fit by less than [X] radians at every age.

**Pattern C — methods paper / backend comparison.** Fit all four. Show
the trajectories overlaid in one figure (`plot_circ_fit({r1, r2, r3,
r4}, tbl)`). The Methods section names every backend with one sentence
per (using the templates above) and a final sentence explaining why
each appears. The Results section reports the omnibus age-effect
statistic from each backend on a single line so the reader can compare
at a glance.

## Quick start

```matlab
addpath(genpath('circular_regression_toolbox'));

% tbl needs: <response>, <predictor>, Subj_ID (and optionally electrode, sex).
% Response is an angle in radians; the toolbox centers by circular mean
% internally so the input range is irrelevant.
result = circ_fit(tbl, 'Phase ~ 1 + Age^2 + (1|Subj_ID)', 'fitcirc_lme', ...
                  struct('Select', true, 'MaxOrder', 2));

result.SelectedOrder              % polynomial order picked by LRT
result.AgeEffect.pValue           % omnibus joint test: all age-involving terms = 0
result.GOF.R2_circ_marginal       % fixed effects only
result.GOF.R2_circ_conditional    % fixed effects + subject random intercept
result.Trajectory                 % evaluation-grid table (Age × electrode × {mean, lo, hi})
plot_circ_fit(result, tbl);       % triple-line at ±2π so the seam never jumps
```

To overlay multiple backends on the same data (sensitivity / robustness check):

```matlab
r1 = circ_fit(tbl, fml, 'fitcirc_lme', opts);
r2 = circ_fit(tbl, fml, 'brms',        opts);
r3 = circ_fit(tbl, fml, 'lme4',        opts);
r4 = circ_fit(tbl, fml, 'bpnreg',      opts);
plot_circ_fit({r1, r2, r3, r4}, tbl);
```

## Tutorial: simulate, fit, recover

For a hands-on walkthrough, run [`tutorial.m`](tutorial.m). It generates one
synthetic dataset with known parameters and walks through every piece of
the result against the truth that produced it. Takes a few seconds.

```matlab
addpath(genpath('/path/to/circular_regression_toolbox'));
run(fullfile('/path/to/circular_regression_toolbox', 'tutorial.m'));
```

Inline walkthrough (same content as the script, with prose between blocks):

### 1. Simulate a von Mises GLMM by hand

We pretend we are running a study with 100 subjects, 8 observations each.
The response is an angle (radians) that depends on age via a shifted
parabola, plus a per-subject baseline offset and within-subject noise.

```matlab
clear; close all; rng(0);

n_subj = 100;  n_per = 8;
ages_subj = linspace(8, 80, n_subj)';
Subj_ID = repelem((1:n_subj)', n_per);
Age     = repelem(ages_subj, n_per);

% True fixed effects (the population trend with age)
beta_intercept = 0.30;
beta_age       = -0.025;
beta_age2      =  0.00050;
age_centered   = Age - mean(Age);
mu_fixed       = beta_intercept + beta_age*age_centered + beta_age2*age_centered.^2;

% True concentrations
kappa_phi = 6;     % between-subject (lower = more subject heterogeneity)
kappa_eps = 10;    % within-subject  (higher = tighter cluster around the curve)

% Draw subject offsets and per-row residual noise from von Mises
phi_subj = circ_vmrnd(0, kappa_phi, [n_subj, 1]);
eps_row  = circ_vmrnd(0, kappa_eps, [numel(Age), 1]);

% Wrap the resulting angles to (-pi, pi]
wrap  = @(x) ((x + pi) - 2*pi*floor((x + pi)/(2*pi))) - pi;
Phase = wrap(mu_fixed + phi_subj(Subj_ID) + eps_row);

T = table(Subj_ID, Age, Phase);
```

`circ_vmrnd(mu, kappa, sz)` is the Best–Fisher (1979) rejection sampler
shipped with the toolbox; use `rng(seed)` first if you want a draw you can
reproduce exactly.

### 2. Fit

```matlab
result = circ_fit(T, ...
    'Phase ~ 1 + Age + Age^2 + (1|Subj_ID)', ...
    'fitcirc_lme', ...
    struct('Select', true, 'MaxOrder', 3));
```

We pass `MaxOrder = 3` to test whether the step-up LRT will be fooled into
accepting a cubic term we did not put in the truth. With `Select = true`
the toolbox starts at intercept-only and adds polynomial terms one by one,
stopping when the LRT no longer accepts the next term at α = 0.05.

### 3. Read the result against truth

```matlab
fprintf('Selected polynomial order: %d   (truth = 2)\n', result.SelectedOrder);
fprintf('Omnibus Age test p-value:  %.3g\n',             result.AgeEffect.pValue);
fprintf('R2_circ marginal:          %.3f\n',             result.GOF.R2_circ_marginal);
fprintf('R2_circ conditional:       %.3f\n',             result.GOF.R2_circ_conditional);
fprintf('MAE_angular:               %.3f rad\n',         result.GOF.MAE_angular);
disp(result.Coefficients);
```

Expected output (the exact numbers depend on the RNG seed):

```
Selected polynomial order: 2   (truth = 2)
Omnibus Age test p-value:  ~0
R2_circ marginal:          ~0.15
R2_circ conditional:       ~0.65
MAE_angular:               ~0.32 rad
```

The selected order matches the truth, the omnibus age test is decisively
significant, and the gap between marginal and conditional R² tells you that
subject heterogeneity is a large component of the response variance — which
matches the modest `kappa_phi = 6` we set up.

**Why the Age coefficients in `result.Coefficients` don't match `beta_age`
and `beta_age2` directly.** The toolbox fits in an orthogonal-polynomial
reparameterization for numerical conditioning (`Age_op1`, `Age_op2` instead
of `Age`, `Age^2`). The fitted curve and the joint Age test are identical
to what the raw `[Age, Age^2]` basis would give; the per-coefficient values
just live in a rotated basis.

### 4. Marginal vs conditional R² in plain words

- **R²ₘ (marginal)** — "how well can I predict a brand-new subject from their
  Age alone?" Fixed effects only; the subject random intercept is set to zero.
- **R²_c (conditional)** — "how well can I predict if I have already measured
  this subject and know their personal baseline?" Fixed effects + the per-
  subject offset.
- The gap `R²_c − R²ₘ` is the share of the response variance explained by the
  subject random intercept on top of what the fixed effects (Age, etc.) explain.
  It is the most direct quantitative report on between-subject heterogeneity.

For circular models the toolbox computes both via the Nakagawa–Schielzeth
(2013) ICC adjustment adapted to a von Mises GLMM. Each variance component
is the circular variance `V = 1 − I₁(κ)/I₀(κ)` of the relevant concentration
parameter (`κ_φ` for subjects, `κ` for residuals); the ICC is then
`V_α / (V_α + V_ε)`, and `R²_c = R²ₘ + ICC·(1 − R²ₘ)`.

### 5. Plot the fit

```matlab
plot_circ_fit(result, T);
```

The plotter draws the trajectory mean and CI band against the raw points.
The angular axis is repeated above and below at ±2π so the curve never
"jumps" at the ±π seam — a visualization trick documented in `plot_circ_fit.m`.

### 6. Things to try

- Change `kappa_phi` to `1` (large subject variation) or `30` (subjects very
  alike). Watch how the gap between R²ₘ and R²_c grows or shrinks.
- Set `beta_age2 = 0` and re-run. `SelectedOrder` should drop to 1 — the LRT
  no longer accepts a curvature term.
- Add a covariate: simulate a binary `sex` factor with its own effect, append
  `+ sex` to the formula, refit, and inspect `result.Coefficients`.
- Swap the backend to `'brms'`, `'lme4'`, or `'bpnreg'` (requires R + the
  named package) and overlay the trajectories with
  `plot_circ_fit({r1, r2, r3, r4}, T);`.

## The uniform result schema

Every backend returns a struct validated by `make_circ_result`. Full
field-by-field reference is in [`docs/result_schema.md`](docs/result_schema.md);
the highlights:

**Required of every backend**
- `Backend`, `Formula`, `ResponseName`, `Order`, `ThetaShift`
- `Trajectory` — table `{Age, electrode, sex, mean, lo, hi}`, unwrapped per electrode
- `GOF` — `{R2_circ, R2_circ_marginal, R2_circ_conditional, R2_adj, MAE_angular, LogLikelihood, AIC, BIC}`.
  `R2_circ` is an alias of `R2_circ_marginal` kept for backward compatibility.
  `R2_circ_conditional` lifts the marginal value by the per-subject random-
  intercept variance via the Nakagawa–Schielzeth (2013) ICC adjustment (see
  the Tutorial section above). AIC/BIC are NaN for the Bayesian backends.
- `AgeEffect` — `{pValue, stat, df, Method}`, a single omnibus "any age effect" test
- `OrderTable` — per-order log-likelihood, $R^2_\text{circ}$, criterion value, and which row was selected
- `SelectedOrder`, `SelectCriterion` (`'LRT'` | `'LRT-sincos'` | `'LOO'` | `'WAIC'`)
- `Diagnostics`, `Converged`

**Optional** — populated only when the backend exposes a single linear-predictor coefficient vector (i.e. `fitcirc_lme` and `brms`):
- `Coefficients` (table `{Name, Estimate, SE, pValue}`, Wilkinson grammar)
- `Beta`, `cov_b`, `ContrastIndex`, `CoefficientNames`, `NumCoefficients`

The split is structural, not stylistic: `lme4` fits two sin/cos models behind
`atan2` and `bpnreg` carries two posterior coefficient sets, so neither has a
uniform per-coefficient table.

`R2_circ` and `MAE_angular` are the only metrics directly comparable across
backends (they're computed identically on the angle scale).
`LogLikelihood`/`AIC`/`BIC` are within-backend only (different likelihood
families).

## Backend cheat-sheet

The detailed write-up of each method, the chooser logic, and paper-
reporting templates live in [The four methods](#the-four-methods) and
[Which method to use](#which-method-to-use). At-a-glance table:

| Backend         | Model                                 | Order selection                                       | Random effects               | Uncertainty                    | Wraps full revolution? | Dependencies                        |
|-----------------|---------------------------------------|-------------------------------------------------------|------------------------------|--------------------------------|------------------------|-------------------------------------|
| **fitcirc_lme** | von Mises GLMM, exact EM              | LRT                                                   | Single `(1|group)` intercept | Cluster-robust Wald (sandwich) | No                     | MATLAB only                         |
| **brms**        | Bayesian vM-GLMM, `tan_half` link     | LOO (`elpd_diff > 2·se_diff`); LRT reported alongside | brms-side                    | Posterior (Stan)               | No                     | R + `brms` + `loo` + Stan toolchain |
| **lme4**        | Sin/cos parallel LMEs                 | Combined sin+cos LRT                                  | lme4-side `(1|Subj_ID)`      | Wald + optional `bootMer` band | Yes                    | R + `lme4`                          |
| **bpnreg**      | Bayesian projected-normal mixed model | WAIC                                                  | bpnreg-side                  | Posterior                      | Yes                    | R + `bpnreg`                        |

Full side-by-side comparison: [`docs/backends.md`](docs/backends.md). The
"Age effect" significance statistic is, for every backend, an omnibus test
that **all** age-involving terms (`Age`, `Age^k`, `Age:cat`, `Age^k:cat`)
are simultaneously zero — one number per model.

## Function index

Detailed per-function docs in [`docs/functions.md`](docs/functions.md).

| Function | One-liner |
|---|---|
| [`circ_fit`](docs/functions.md#circ_fit) | Main entry point: dispatches to a backend and returns the uniform result struct. |
| [`fit_circ_method`](docs/functions.md#fit_circ_method) | Lower-level dispatcher with cluster-bootstrap / subject-subsample bagging around `fitcirc_lme`. |
| [`circ_fit_fitcirc`](docs/functions.md#circ_fit_fitcirc) | `fitcirc_lme` backend adapter: orthogonal-polynomial basis, order selection by LRT, trajectory + CI, GOF. |
| [`fitcirc_lme`](docs/functions.md#fitcirc_lme) | The core estimator: von Mises GLMM with subject random intercept, fit by exact EM, cluster-robust SEs. |
| [`circ_fit_config`](docs/functions.md#circ_fit_config) | Process-global config (default backend, sampler opts, etc.). |
| [`circ_center`](docs/functions.md#circ_center) | Canonical preprocessing: subtract the circular mean to place the seam in the data gap. |
| [`circ_shift_min_var`](docs/functions.md#circ_shift_min_var) | Legacy alternative: variance-minimizing seam placement. |
| [`circ_gof`](docs/functions.md#circ_gof) | Cross-backend goodness-of-fit: $R^2_\text{circ}$, adjusted $R^2$, mean absolute angular error. |
| [`circ_vmrnd`](circ_vmrnd.m) | Best–Fisher 1979 rejection sampler for the von Mises distribution. Used by the tutorial and the parameter-recovery tests. |
| [`ortho_poly_basis`](docs/functions.md#ortho_poly_basis) | R-style orthonormal polynomial basis with a transform that can be reapplied at new $x$. |
| [`make_circ_result`](docs/functions.md#make_circ_result) | Result-struct factory + validator (single source of truth for the schema). |
| [`read_circ_result`](docs/functions.md#read_circ_result) | Read the R worker's output contract from disk and assemble a `circ_result`. |
| [`write_circ_contract`](docs/functions.md#write_circ_contract) | Write `data.csv` / `eval_grid.csv` / `meta.json` for the R worker. |
| [`plot_circ_fit`](docs/functions.md#plot_circ_fit) | Plot one or more results with the triple-line trick for seam-free display. |
| [`tutorial.m`](tutorial.m) | Self-contained simulate-and-recover walkthrough; see the [Tutorial section](#tutorial-simulate-fit-recover). |

## Repository layout

```
circular_regression_toolbox/
├── README.md                  (this file)
├── tutorial.m                 (runnable simulate-and-recover walkthrough)
├── docs/
│   ├── functions.md           (per-function reference)
│   ├── backends.md            (the four backends side by side)
│   └── result_schema.md       (field-by-field of the circ_result struct)
├── circ_fit.m                 (MATLAB-side dispatcher)
├── circ_fit_fitcirc.m         (fitcirc_lme adapter)
├── circ_fit_config.m          (process-global config factory)
├── fit_circ_method.m          (legacy / cboot / sub80 resample bagging)
├── fitcirc_lme.m              (the core EM von Mises GLMM)
├── circ_center.m              (canonical preprocessing)
├── circ_shift_min_var.m       (legacy variance-minimizing shift)
├── circ_gof.m                 (R^2_circ + MAE)
├── circ_vmrnd.m               (von Mises rejection sampler; used by tests + tutorial)
├── ortho_poly_basis.m         (R-style orthonormal polynomial basis)
├── make_circ_result.m         (schema factory)
├── read_circ_result.m         (R worker output → MATLAB result struct)
├── write_circ_contract.m      (MATLAB table → R worker input contract)
├── plot_circ_fit.m            (triple-line plotter)
├── R/                         (R-side worker for brms/lme4/bpnreg)
│   ├── circ_fit.R             (entry: Rscript circ_fit.R <work_dir> <backend>)
│   ├── circ_fit_common.R      (shared helpers: unwrap, R²_circ, name remap, IO)
│   ├── circ_fit_brms_impl.R
│   ├── circ_fit_lme4_impl.R
│   └── circ_fit_bpnreg_impl.R
├── utils/
│   ├── build_model_formula.m  (Wilkinson formula builder)
│   └── get_LLR.m              (chi-square LRT helper)
└── tests/                     (parity / recovery / pipeline tests)
```

`R/` is the brms / lme4 / bpnreg backend. The MATLAB side writes a small
contract (`data.csv`, `eval_grid.csv`, `meta.json`) into a working directory
via `write_circ_contract`, shells out to `Rscript R/circ_fit.R <work_dir>
<backend>`, and reads the per-backend output files
(`<backend>_predictions.csv`, `<backend>_stats.json`,
`<backend>_order_table.csv`, optional `<backend>_coefs.csv` /
`_cov_b.csv`) back via `read_circ_result`. Each `*_impl.R` does the
backend-native fit, order sweep, and writes the same output contract.

`utils/build_model_formula.m` and `utils/get_LLR.m` are tiny, stable helpers
copied here so the toolbox is self-contained.

## Dependencies

- **MATLAB** R2020b or later (tables, `categorical`, anonymous functions with
  struct capture).
- **R 4.x** *only if you want the brms / lme4 / bpnreg backends*. With:
  - `brms`, `loo`, `readr`, `jsonlite`  (brms)
  - `lme4`                              (lme4)
  - `bpnreg`                            (bpnreg)
- `Rscript` on the system `PATH`. Override via `opts.RscriptPath` if needed
  (default is `/usr/local/bin/Rscript`).

No R is required for the default `fitcirc_lme` backend or any MATLAB-side
function.

## Installation

### As a Git submodule

```sh
git submodule add git@github.com:preraulab/circular_regression_toolbox.git path/to/circular_regression_toolbox
git submodule update --init --recursive
```

In MATLAB:

```matlab
addpath(genpath('path/to/circular_regression_toolbox'));
```

### Standalone

```sh
git clone git@github.com:preraulab/circular_regression_toolbox.git
```

## Configuration

`circ_fit_config` is a process-global config (set once, read by `circ_fit`):

```matlab
circ_fit_config('set', struct( ...
    'Backend',  'fitcirc_lme', ...      % default backend
    'MaxOrder', 2, ...                  % polynomial-order cap for selection
    'Select',   true, ...               % order selection on
    'Chains',   4, 'Iter', 2000, ...    % brms sampler options
    'Band',     true));                 % lme4 bootMer CI band
```

Options can also be passed inline as the last argument to `circ_fit`.

## Tests

From inside MATLAB, with the toolbox on the path:

```matlab
test_circ_fit_schema({'fitcirc_lme','lme4'});                       % fast
test_circ_fit_schema({'fitcirc_lme','brms','lme4','bpnreg'});       % full
sim_circ_compare();                                                 % overlay on a wrapping sim
```

`test_circ_fit_schema` asserts every backend returns the same required-tier
fields, that `AgeEffect.pValue` is populated, and that the figure-side
trajectory wiring (`get_model_fit` → `Trajectory` interp) works for each.
The full per-test inventory is in [`docs/functions.md`](docs/functions.md#tests).

## Conventions

- **Centering.** Inputs are centered by their circular mean
  $\theta_\text{shift} = \text{atan2}\!\bigl(\overline{\sin y},\ \overline{\cos y}\bigr)$
  before fitting; $\theta_\text{shift}$ is recorded on the result and added
  back for predictions. This is the recommended fix for the von Mises
  seam-multimodality problem.
- **Unwrapping.** Each backend's stored trajectory is unwrapped per electrode
  along the predictor — the plotter then doesn't need break-at-jumps logic.
- **Coefficient names.** Wilkinson grammar (`(Intercept)`, `Age`, `Age^2`,
  `Age^2:electrode`), uniform across backends; the R workers remap their
  native names accordingly. Internally, `circ_fit_fitcirc` uses an
  orthogonal-polynomial reparameterization with names like `Age_op1`,
  `Age_op2` for numerical conditioning; the fitted curve and the joint
  Wald test are unchanged by this reparameterization (Wald is invariant under
  nonsingular linear reparameterization of the tested block).

## Known issues

These are documented behaviors that have not been fixed, and that downstream
callers should be aware of:

- **Cluster-robust sandwich SEs underperform with very small clusters.** The
  cluster-robust ("sandwich") SEs in `fitcirc_lme` are correct asymptotically
  but can under-cover when each subject contributes only two observations.
  For primary inference with very small clusters, prefer the resample-based
  options in `circ_fit_fitcirc`.
- **brms chains forced sequential.** When `Rscript` is launched from
  MATLAB's `system()`, parallel chain workers fail to initialize rstan, so
  `circ_fit_brms_impl.R` forces `cores = 1`. Slower than parallel sampling,
  but works in every launch environment.
- **`fitcirc_lme` supports a single `(1|group)` random-intercept only.** No
  random slopes, no crossed grouping factors. For more complex random-effects
  structures, switch to a Bayesian backend.
- **bpnreg has no frequentist p for `AgeEffect`.** WAIC is used for both
  order selection and the age-effect test; `AgeEffect.Method` will read
  `'bpnreg'` and `pValue` is derived from WAIC differences rather than a
  classical test.
- **Default `RscriptPath` is `/usr/local/bin/Rscript`** and is not auto-detected.
  Override via `opts.RscriptPath` if your R install is elsewhere (e.g. Homebrew
  on Apple Silicon at `/opt/homebrew/bin/Rscript`).

## Citation

If you use this toolbox in a paper, please cite the host paper this code was
extracted from (TBD).

## License

TBD.
