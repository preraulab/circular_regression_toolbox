# `brms` — Bayesian von Mises GLMM (Stan)

> **Cross-references.** Plain-language overview in the
> [README method section](../../README.md#brms--bayesian-von-mises-mixed-effects-model-r--stan)
> and the at-a-glance comparison in
> [`docs/backends.md`](../backends.md#brms--bayesian-von-mises-glmm-stan).
> The R-side adapter lives at [`R/circ_fit_brms_impl.R`](../../R/circ_fit_brms_impl.R).

This page is the full technical reference for the brms backend. The
underlying model is identical to [`fitcirc_lme`](fitcirc_lme.md)'s; the
difference lies in the inference machinery — Hamiltonian Monte Carlo via
Stan, with weakly-informative priors on every parameter and leave-one-out
cross-validation for order selection. The level of detail is what a
statistician needs to know exactly what is being computed and what the
diagnostic guarantees are.

---

## 1. Model

Same notation as [`fitcirc_lme`](fitcirc_lme.md#1-model): $y_{ij} \in (-\pi, \pi]$,
$X_{ij} \in \mathbb{R}^{1 \times p}$, $\beta \in \mathbb{R}^p$,
$\phi_i \in (-\pi, \pi]$, concentrations $\kappa, \kappa_\phi > 0$.

**Likelihood.**

$$
y_{ij} \mid \beta, \phi_i, \kappa  \sim  \text{vonMises}(\eta_{ij},\ \kappa),
\qquad \eta_{ij} = X_{ij}\beta + \phi_i.
$$

**Link function.** brms parameterizes the von Mises location through the
`tan_half` link:

$$
\mu_{ij}  =  2\arctan(\eta_{ij}),
$$

so the linear predictor $\eta_{ij} \in \mathbb{R}$ maps onto $\mu_{ij} \in (-\pi, \pi)$
smoothly. This avoids the ad-hoc wrap-handling that would be required if
the linear predictor itself were the angle. **Note: `tan_half` is
still bounded within one revolution of the response** — the map is
$\mathbb{R} \to (-\pi, \pi)$ rather than $\mathbb{R} \to \mathbb{R}/2\pi\mathbb{Z}$,
so a trajectory that physically swings more than $2\pi$ across the
predictor range cannot be represented.

**Random-effect structure.**

$$
\phi_i \mid \sigma_\phi  \sim  \mathcal{N}(0,\ \sigma_\phi^2),
\qquad i = 1, \ldots, m, \text{ independent.}
$$

(brms's `(1|Subj_ID)` random intercept is a standard Gaussian random-effect
prior. This differs from `fitcirc_lme`, which uses a von Mises prior on
$\phi_i$. For moderate $\sigma_\phi$ — say $\sigma_\phi \lesssim 1$ rad —
the Gaussian and von Mises priors are close. They diverge when subjects'
offsets approach uniform on the circle, but in practice that regime never
arises in lifespan-data fits where subject baselines are reasonably
concentrated.)

**Predictor standardization.** Before fitting, the R-side adapter
z-scores Age:

$$
\tilde x  =  \frac{x - \bar x}{\text{sd}(x)},
$$

and constructs the polynomial in $\tilde x$. Predictions are
de-standardized on output. This is purely numerical conditioning for
NUTS; the fitted curve and the joint Wald test are invariant under this
linear reparameterization.

---

## 2. Priors

brms's default weakly-informative priors are kept:

| Parameter | Prior |
|---|---|
| Fixed-effect coefficients $\beta_k$, $k \ne 0$ (intercept) | $\mathcal{N}(0,\ \sigma_\beta^2)$ with $\sigma_\beta$ from brms's data-scale default (typically $\sigma_\beta = 2.5 \text{sd}(\tilde x)^{-1}$ after standardization) |
| Intercept $\beta_0$ | Student's $t_{3}(0,\ 2.5)$ |
| Random-intercept SD $\sigma_\phi$ | Half-Student's $t_{3}(0,\ 2.5)$ |
| Response concentration $\kappa$ | Gamma(2, 0.1) (brms default for $\text{kappa}$) |

All priors are weakly informative — the posterior is dominated by the
likelihood in any non-degenerate regime, but the priors prevent
pathological MCMC behavior in regimes where the likelihood is flat
(e.g. when there is no subject variation, $\sigma_\phi \to 0$).

---

## 3. Inference: NUTS via Stan

Posterior sampling uses the No-U-Turn Sampler (Hoffman & Gelman, 2014)
as implemented in Stan and invoked through brms's `brm` interface.

**Sampler defaults** (set in `circ_fit_brms_impl.R`):

| Option | Default | Override via |
|---|---|---|
| Chains | 4 | `opts.Chains` |
| Iterations per chain | 2000 (1000 warm-up + 1000 sampling) | `opts.Iter`, `opts.Warmup` |
| Target acceptance rate | 0.95 | `opts.AdaptDelta` |
| Cores (parallel chains) | **1** (sequential — forced) | n/a |
| Seed | 1 | `opts.Seed` |

**Why sequential chains.** When the R worker is launched from MATLAB's
`system()`, parallel chain workers fail to initialize `rstan` reliably.
The impl forces `cores = 1` so the four chains run sequentially. This
roughly quadruples wall-clock time relative to a parallel run; for batch
production fits, schedule them at the MATLAB level instead.

**Convergence diagnostics.** Stan returns the rank-normalized split-$\hat R$
(Vehtari et al., 2021) and effective sample size (ESS, both bulk and tail)
for every sampled parameter. Cuts:

$$
\hat R \le 1.01, \qquad \text{ESS}_\text{bulk} \ge 400, \qquad \text{ESS}_\text{tail} \ge 400.
$$

A warning is emitted on the MATLAB side when any sampled parameter
violates these thresholds; the result is still returned (the user
decides whether to act).

---

## 4. Order selection: LOO

With `Select = true`, the R-side adapter fits the model at every
polynomial order $k = 0, 1, \ldots, \text{MaxOrder}$. For each fit it
records:

- the per-observation expected log pointwise predictive density (ELPD)
  estimated by Pareto-smoothed importance sampling (PSIS-LOO, Vehtari
  et al., 2017);
- the classical chi-square LRT $-2(\ell_{k-1}^\star - \ell_k^\star)$
  using the posterior-mean parameters as point estimates.

**Selection rule.** Step up the order while

$$
\text{elpd} _k - \text{elpd} _{k-1}  >  2 \cdot \text{se}\bigl(\text{elpd} _k - \text{elpd} _{k-1}\bigr),
$$

i.e. accept the larger order only when its predictive-density improvement
exceeds twice its own standard error. (Vehtari & Gabry's "2 SE" rule.) The
LRT p-value is reported alongside but not used for selection.

The full per-order audit is in `OrderTable`:

```
order  n_par  LogLikelihood  R2_circ  elpd_loo  se_elpd_loo  elpd_diff  selected
0      ...    ...            ...      ...       ...          0          false
1      ...    ...            ...      ...       ...          dx_1       true/false
2      ...    ...            ...      ...       ...          dx_2       true/false
```

with `elpd_diff = elpd_k - elpd_{k_selected}` and the `selected = true`
row identifying which $k$ was chosen.

---

## 5. Inference quantities

### 5.1 Coefficient table

For each fixed-effect coefficient $\beta_k$, posterior summaries:

- `Estimate` — posterior median.
- `SE` — posterior standard deviation.
- `pValue` — **two-sided posterior probability of being on the wrong side
  of zero**:

$$
p_k  =  2 \min\bigl(\Pr(\beta_k > 0 \mid y),\ \Pr(\beta_k < 0 \mid y)\bigr).
$$

This is the "Bayesian p" used elsewhere in the literature (e.g. brms's
own `posterior_summary` output); it equals the frequentist p in the
large-sample Gaussian limit but is computed from the MCMC sample
empirically.

### 5.2 Omnibus age-effect test

The brms adapter computes the joint age-effect statistic two ways and
returns whichever was selected:

1. **Posterior probability mass** that the full age-block $\beta_\text{age}
   = (\beta_\text{Age}, \beta_{\text{Age}^2}, \ldots)$ lies in the
   half-space defined by its posterior-mean direction:

   $$
   p  =  1 - \Pr\bigl(\beta_\text{age}^\top \hat v > 0 \mid y\bigr), \qquad
   \hat v = \mathbb{E}[\beta_\text{age} \mid y] / \|\mathbb{E}[\beta_\text{age} \mid y]\|.
   $$

2. **Bayes factor against the no-age model** via Savage–Dickey:

   $$
   \text{BF}_{10}  =  \frac{\pi(0)}{\Pr(\beta_\text{age} = 0 \mid y)},
   $$

   approximated from the posterior marginal at zero. The reported
   `AgeEffect.pValue` is $\min(1, 1/\text{BF}_{10})$ for backward
   compatibility with the frequentist field of the result schema.

The reported `AgeEffect.Method` field reads `'brms'` so consumers can
distinguish from frequentist p-values.

### 5.3 Trajectory

For each evaluation point $(x^\star, \text{electrode}^\star, \text{sex}^\star)$
the trajectory mean and band are computed from the joint posterior:

$$
\mu^\star  =  \text{median}_{s = 1,\ldots,S} \bigl(2\arctan(X^\star\beta^{(s)})\bigr),
$$

with $\beta^{(s)}$ the $s$-th MCMC draw. The 95% credible band is the
2.5% / 97.5% quantiles of the same posterior sample. Random effects are
integrated out by setting $\phi_i = 0$ on the eval grid (population
prediction); the adapter does *not* draw from the random-intercept
distribution to construct subject-conditional predictions.

### 5.4 $R^2_\text{circ}$

Computed on the posterior median predictions $\hat y_{ij}^\text{med}
= 2\arctan(X_{ij}\hat\beta^\text{med})$ via the same formula as
[`fitcirc_lme`](fitcirc_lme.md#71-marginal-r2_textcirc-population-level)
— marginal only. A conditional analogue could be constructed from
posterior $\sigma_\phi$ samples by an analogous Nakagawa-Schielzeth
adjustment; this is not currently implemented in the brms adapter.

---

## 6. Assumptions and when they bite

| Assumption | When it matters | How brms handles it |
|---|---|---|
| $\sigma_\phi$ prior is half-Student-$t_3$ | Mildly informative; matters when the random-effect variance is near zero. | brms's default; documented in `?brms::set_prior`. |
| Random intercepts are Gaussian (not vM) | Diverges from `fitcirc_lme` when subject offsets are nearly uniform on the circle (rare). | Documented; results are typically within $10^{-3}$ rad of `fitcirc_lme` in lifespan-cohort regimes. |
| HMC has converged | Misleading inference if $\hat R > 1.01$ or ESS too small. | Diagnostic warning emitted on the MATLAB side. |
| LOO-PSIS is reliable | Fails when some Pareto-$k$ diagnostics exceed 0.7 (a small number of observations have undue influence). | LOO returns those diagnostics; the impl warns when any $k > 0.7$. |
| Trajectory fits within one revolution | `tan_half` is bounded to $(-\pi, \pi)$. Same limitation as `fitcirc_lme`. | Documented. Switch to `bpnreg` for full-revolution trajectories. |

---

## 7. Numerical and computational notes

- **Per-fit cost.** With 4 chains × 2000 iterations and `cores = 1`,
  a single fit of a vM-GLMM on lifespan-cohort data (n ~ 1500, m ~ 800)
  takes 3 – 10 minutes wall-clock on a modern laptop. Order selection
  multiplies this by `MaxOrder + 1` (one fit per candidate order).
- **Compilation cache.** Stan compiles the model the first time it's
  used; subsequent calls reuse the compiled binary. The R-side adapter
  uses a stable per-(feature, slice) cache directory under
  `results/circ_cache/` so the brms `.rds` model files persist across
  MATLAB sessions.
- **Random seed.** Pass `opts.Seed` to make a fit exactly reproducible.
  Without an explicit seed, brms inherits R's RNG state.

---

