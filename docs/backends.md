# Backends side-by-side

The toolbox exposes four estimators behind one signature. This page lays out
what each fits, what random-effects structure each supports, how uncertainty
is computed, what the speed and dependency cost is, and which fields of the
unified result struct each populates.

Pick by the criteria in the [chooser](#chooser) at the bottom of this page.

---

## fitcirc_lme — native MATLAB

> Full technical reference: [`docs/methods/fitcirc_lme.md`](methods/fitcirc_lme.md).

**Model.** True von Mises generalized linear mixed model:

$$
y_{ij}\mid \beta, \phi_i, \kappa \sim \text{vonMises}(X_{ij}\beta + \phi_i,\ \kappa),
\qquad
\phi_i \sim \text{vonMises}(0,\ \kappa_\phi)
$$

Both the response noise and the subject offset are von Mises. That pairing
makes each subject's offset $\phi_i$ exactly von Mises given the data and
parameters — no Laplace approximation. The exact marginal likelihood is

$$
\log p(y_i) = \log I_0(K_{\text{post},i}) - n_i \log\bigl(2\pi\,I_0(\kappa)\bigr) - \log I_0(\kappa_\phi).
$$

**Fitter.** EM with a closed-form E-step. The M-step runs a weighted
circular regression (IRLS) for $\beta$, turns the $\rho_i$-weighted mean
resultant length of the residuals into a concentration for $\kappa$, and
finds $\kappa_\phi$ by an exact one-dimensional maximization that includes a
weakly-informative prior. The prior keeps $\kappa_\phi$ finite when the
subject offsets collapse toward 0 (which would otherwise send
$\kappa_\phi \to \infty$ and overflow the likelihood); set it through
`KappaPhiPrior` / `KappaPhiPriorScale`, with an optional hard ceiling
`KappaPhiMax` (default `inf`).

**Random effects supported.** Exactly one `(1|group)` random-intercept term.
No random slopes, no crossed grouping factors. The grouping variable is
named in the formula via `(1|Subj_ID)`.

**Uncertainty.** Cluster-robust ("sandwich") SEs on $\beta$ with each
subject as one cluster, built from the same $\rho_i$-weighted score and
using the positive-definite expected-information bread. Rescaled by the
small-sample factor $\frac{m}{m-1}\cdot\frac{n-1}{n-p}$; inference uses
Student's $t$ on $n_\text{subj} - 1$ degrees of freedom. The joint Wald
age-effect test is computed on the full `ContrastIndex.x_age` block
(polynomial main effects plus every age-interaction).

**Order selection (when `Select` is on).** Step up the polynomial order
while each nested LRT $p < 0.05$; stop at the first non-significant step.

**Speed.** Fast. Pure MATLAB, no compilation. EM typically converges in
$O(10)$ iterations on lifespan-size data.

**Dependencies.** None beyond base MATLAB (R2020b+).

**Options used.** `Select`, `MaxOrder`, `Order`, `x_col`, `feature`,
`categorical_varnames`, `xcol_categorical_interactions`, `Resample`, `B`,
`KeepFrac`, `eval_ages`.

**Known limitations.** EM marginal LL can be non-monotone in order when
$\kappa_\phi$ posterior collapses near 0 (symptom: `AgeEffect.pValue = NaN`
in spite of a visible $R^2_\text{circ}$ jump). Cluster-robust SE undercovers
with very small $n_j$ (e.g. two-obs-per-subject). The fitter does not
represent a mean that wraps a full revolution (vM mean is a single point).

---

## brms — Bayesian von Mises GLMM, Stan

> Full technical reference: [`docs/methods/brms.md`](methods/brms.md).

**Model.** Same vM-GLMM as `fitcirc_lme`, fit via Stan with a `tan_half` link.
brms parameterizes the linear predictor with a polynomial in standardized
Age (z-scored before fitting; predictions de-standardized on output).

**Fitter.** Stan HMC (NUTS) through `brms::brm`. Chains run sequentially —
the impl forces `cores = 1` because parallel chain workers fail to spawn
under `Rscript` launched from MATLAB's `system()` (`rstan` cannot
initialize). Slower than parallel sampling, but works everywhere.

**Random effects supported.** Whatever brms supports (anything Stan can
express). The MATLAB-side dispatcher writes a `(1|Subj_ID)` term by default;
override by hand-writing the worker invocation if you need a richer
structure.

**Uncertainty.** Full posterior. `Coefficients.SE` is the posterior SD,
`pValue` is the two-sided posterior probability mass beyond zero.
`Trajectory.lo` / `Trajectory.hi` are the 2.5% / 97.5% posterior quantiles.

**Order selection (when `Select` is on).** Step up while
$\Delta\text{elpd}_\text{loo} > 2\,\text{SE}(\Delta\text{elpd}_\text{loo})$.
A chi-square LRT $p$ is reported alongside for cross-reference.
`SelectCriterion` reads `'LOO'`.

**Speed.** Slow. Compiled Stan models cached as `.rds` inside the work
directory; the auto-tag is data-dependent so different data slices get
separate caches. First fit per (slice, model) is dominated by compile time;
subsequent fits use the cached model.

**Dependencies.** R, `brms`, `loo`, `readr`, `jsonlite`, plus a working
Stan toolchain (C++ compiler, `rstan`).

**Options used.** All of the above plus `Chains` (default 4), `Iter`
(default 2000), `Warmup` (default 1000), `Seed` (default 1), `AdaptDelta`
(default 0.95). `WorkDir` controls where caches live. `BrmsFallback`
(default true) falls back to `fitcirc_lme` on failure.

**Known limitations.** Stan compile time on first run. Cannot represent a
mean that wraps a full revolution. `cores = 1` slows sampling.

---

## lme4 — sin/cos parallel LMEs

> Full technical reference: [`docs/methods/lme4.md`](methods/lme4.md).

**Model.** Two parallel Gaussian LMMs: one on $\sin y$, one on $\cos y$.
Reconstructed angle is $\text{atan2}(\widehat{\sin y},\ \widehat{\cos y})$.
This is the frequentist projected-Gaussian approach.

**Fitter.** `lme4::lmer`, REML disabled (`REML = FALSE`).

**Random effects supported.** Anything `lme4` supports. The MATLAB
dispatcher writes `(1|Subj_ID)` by default.

**Uncertainty.** Combined-LL likelihood-ratio test for the age-effect block.
`Trajectory` band, when requested (`opts.Band = true`), comes from `bootMer`
parametric bootstrap; otherwise `lo == hi == mean`.

**Order selection (when `Select` is on).** Combined sin+cos LRT, treating
the LL as the sum of the two component LLs and the parameter count as the
sum of fixed-effects parameters. `SelectCriterion` reads `'LRT-sincos'`.

**Speed.** Fast (LMM, no MCMC). `bootMer` band can be slow when enabled.

**Dependencies.** R, `lme4`.

**Options used.** `Select`, `MaxOrder`, `Order`, `eval_ages`, `Band`.

**Known limitations.** Trajectory mean *can* wrap a full revolution — this
is the right choice when your data does. No single $\beta$ vector
(two coefficient sets behind `atan2`), so the per-coefficient tier of the
result schema is **not** populated.

---

## bpnreg — Bayesian projected-normal mixed model

> Full technical reference: [`docs/methods/bpnreg.md`](methods/bpnreg.md).

**Model.** `bpnreg::bpnme` projected-normal mixed model with a polynomial
in raw Age. Like lme4's sin/cos but Bayesian and modeling the projected
bivariate normal directly.

**Fitter.** `bpnreg::bpnme` (Gibbs sampler internal to the package).

**Random effects supported.** Whatever bpnreg supports; default is
subject-level.

**Uncertainty.** Posterior. `Trajectory.lo` / `Trajectory.hi` are posterior
quantiles. No frequentist $p$ for `AgeEffect` — `AgeEffect.Method` reads
`'bpnreg'` and the $p$-value is derived from WAIC differences rather than a
classical test.

**Order selection (when `Select` is on).** Step up by **WAIC**.
`SelectCriterion` reads `'WAIC'`.

**Speed.** Moderate (Gibbs, no Stan compile).

**Dependencies.** R, `bpnreg`.

**Options used.** `Select`, `MaxOrder`, `Order`, `eval_ages`, plus shared
sampler options where applicable.

**Known limitations.** Like lme4: trajectory mean *can* wrap a revolution
(this is fine). Two posterior coefficient sets ($\beta^{(1)}, \beta^{(2)}$
for the two projected components), so no single $\beta$ vector — the
per-coefficient tier is **not** populated. WAIC-based age-effect $p$ is not
a classical hypothesis test.

---

## Result-field coverage

Which result-struct fields each backend fills.
$\checkmark$ = populated, $\circ$ = empty / NaN by design.

| Field                  | fitcirc_lme | brms | lme4 | bpnreg |
|------------------------|:-----------:|:----:|:----:|:------:|
| `Backend`              | ✓ | ✓ | ✓ | ✓ |
| `Formula`              | ✓ | ✓ | ✓ | ✓ |
| `ResponseName`         | ✓ | ✓ | ✓ | ✓ |
| `Order`                | ✓ | ✓ | ✓ | ✓ |
| `ThetaShift`           | ✓ | ✓ | ✓ | ✓ |
| `Trajectory`           | ✓ | ✓ | ✓ | ✓ |
| `GOF.R2_circ`          | ✓ | ✓ | ✓ | ✓ |
| `GOF.MAE_angular`      | ✓ | ✓ | ✓ | ✓ |
| `GOF.LogLikelihood`    | ✓ | ✓ | ✓ | ✓ |
| `GOF.AIC` / `BIC`      | ✓ | ✓ (may be NaN) | ✓ | ✓ (may be NaN) |
| `AgeEffect.pValue`     | ✓ (Wald) | ✓ (Bayes / LRT) | ✓ (LRT-sincos) | ✓ (WAIC-derived) |
| `OrderTable`           | ✓ | ✓ | ✓ | ✓ |
| `SelectedOrder`        | ✓ | ✓ | ✓ | ✓ |
| `SelectCriterion`      | `'LRT'` | `'LOO'` | `'LRT-sincos'` | `'WAIC'` |
| `Diagnostics`          | ✓ (Kappa, KappaPhi, ConvergedIn) | ✓ (rhat_max, divergent) | ✓ | ✓ |
| `Converged`            | ✓ | ✓ | ✓ | ✓ |
| `Coefficients`         | ✓ | ✓ | ○ | ○ |
| `Beta`                 | ✓ | ✓ | ○ | ○ |
| `cov_b`                | ✓ | ✓ | ○ | ○ |
| `ContrastIndex`        | ✓ | ✓ | ○ | ○ |
| `CoefficientNames`     | ✓ | ✓ | ○ | ○ |
| `NumCoefficients`      | ✓ | ✓ | ○ | ○ |
| `NumObservations`      | ✓ | ✓ | ✓ | ✓ |
| `NumSubjects`          | ✓ | ✓ | ✓ | ✓ |
| `DFE`                  | ✓ | ✓ ($n_\text{subj}-1$) | ○ | ○ |
| `WorkDir`              | ○ | ✓ | ✓ | ✓ |
| `Raw`                  | ✓ (fitcirc_lme handle) | ○ | ○ | ○ |

`R2_circ` and `MAE_angular` are the only metrics directly comparable across
backends — they're computed identically on the angle scale. `LL` / `AIC` /
`BIC` live on different likelihood scales between families and are within-
backend only.

---

## Chooser <a id="chooser"></a>

1. **Does your trajectory stay within ~one arc** (less than about half a
   revolution over the predictor range)?
   - Yes → use `fitcirc_lme`. Fast, exact EM, cluster-robust inference,
     gives you the per-coefficient tier with Wald p-values.
   - No → skip to step 2.

2. **Does your trajectory wrap a full revolution?**
   - Yes → use `lme4` (frequentist) or `bpnreg` (Bayesian). Both represent
     the mean as two components that can sweep through $2\pi$. Neither
     populates the per-coefficient tier, but `Trajectory`, `GOF`, and
     `AgeEffect` are all there.
   - No → step 3.

3. **Do you want Bayesian inference plus an LRT cross-check?**
   - Yes → use `brms`. Posterior CIs, LOO-based order selection with a chi-
     square LRT reported alongside, and the same per-coefficient tier as
     `fitcirc_lme`.
   - No → use `fitcirc_lme`.

4. **Robustness panel for a paper.** Run all four with the same `opts`
   and `plot_circ_fit({r1,r2,r3,r4}, tbl)`. `R2_circ` and `MAE_angular`
   are the directly comparable metrics across backends.
