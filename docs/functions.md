# Function reference

Per-function documentation for every public-facing MATLAB function in the
toolbox. Helpers in `utils/` are summarized in an appendix at the end.
Tests are listed at the very end.

Source files referenced below are at
`/Users/Mike/code/projects/trends_v_individual/stats/circular_regression/<file>`.

Conventions used throughout:
- Angles are in **radians** on $(-\pi,\pi]$.
- The seam refers to the $\pm\pi$ boundary.
- Tables follow MATLAB `table` conventions.
- A row marked "default" in an inputs table means the function fills the
  field if the caller leaves it empty or omits it.

---

## circ_fit <a id="circ_fit"></a>

**Purpose.** Unified dispatcher: pick a backend and run a circular regression,
returning the common result struct.

**When to use.** This is the entry point. Almost all calling code should go
through `circ_fit`, not the backend-specific routines.

**Signature.**
```matlab
result = circ_fit(tbl, formula, backend, opts)
```

**Inputs.**

| Name | Type | Meaning | Default |
|---|---|---|---|
| `tbl` | `table` | Data table. Must contain the response column, the predictor column (default `Age`), and `Subj_ID`. May optionally contain `electrode` and `sex`. NaN rows are dropped per backend. | — |
| `formula` | char | Wilkinson string, e.g. `'Phase ~ 1 + Age^2 + (1|Subj_ID)'`. The polynomial order in the formula is the **max** order when `opts.Select == true`, otherwise the single order to fit. Used to seed `opts.feature` / `opts.Order` if not provided directly. | — |
| `backend` | char | One of `'fitcirc_lme'`, `'brms'`, `'lme4'`, `'bpnreg'`. Case-insensitive. | `'fitcirc_lme'` |
| `opts` | struct | Options bag. Common fields: `Select`, `MaxOrder`, `Order`, `x_col`, `feature`, `categorical_varnames`, `xcol_categorical_interactions`, `Resample`, `B`, `KeepFrac`, `eval_ages`, `Chains`, `Iter`, `Warmup`, `Seed`, `AdaptDelta`, `Band`, `WorkDir`, `RscriptPath`, `BrmsFallback`. See the per-backend docs in `backends.md` for which fields each backend reads. | `struct()` |

**Outputs.**

| Name | Type | Semantics |
|---|---|---|
| `result` | struct | The uniform schema produced by `make_circ_result`. See `result_schema.md`. |

**Side effects.** When the backend is an R backend (`brms`/`lme4`/`bpnreg`):
- Creates a work directory (default `./results/circ_cache/<auto-tag>/`).
- Writes `data.csv`, `eval_grid.csv`, `meta.json` to that directory.
- Shells out to `Rscript R/circ_fit.R <work_dir> <backend>`.
- Reads the per-backend output files (`<backend>_predictions.csv`,
  `<backend>_stats.json`, `<backend>_order_table.csv`, optional
  `<backend>_coefs.csv` / `_cov_b.csv`) back.
- brms additionally caches Stan model objects (`.rds`) inside the work
  directory; the auto-tag is data-dependent so different data slices get
  separate caches.

**Notes.**
- On R-backend failure (missing toolchain, nonzero exit, missing outputs) and
  `opts.BrmsFallback = true` (the default), warns and falls back to
  `fitcirc_lme`.
- `opts.RscriptPath` defaults to `/usr/local/bin/Rscript`. Override if R is
  elsewhere.
- `opts.eval_ages` defaults to `(7:80)'` — the lifespan range the toolbox
  was originally written for. Override for other ranges.
- For the `fitcirc_lme` backend specifically, `opts.Resample` chooses
  between `'none'` (default), `'cboot'` (cluster bootstrap), and `'sub80'`
  (subject subsample); see `circ_fit_fitcirc` below.

**Example.**
```matlab
opts = struct('Select', true, 'MaxOrder', 2);
result = circ_fit(tbl, 'Phase ~ 1 + Age^2 + (1|Subj_ID)', 'fitcirc_lme', opts);
fprintf('order=%d  R2=%.3f  ageP=%.2e\n', ...
    result.SelectedOrder, result.GOF.R2_circ, result.AgeEffect.pValue);
```

---

## fit_circ_method <a id="fit_circ_method"></a>

**Purpose.** Lower-level dispatcher around `fitcirc_lme` that adds two
resampling-based inference modes (cluster bootstrap and subject subsample).
Returns a `fitcirc_lme` model object — *not* the unified result schema.

**When to use.** When you want a raw `fitcirc_lme` object (no schema
wrapping, no order selection, no trajectory build) but with subject-level
resampled inference. Most callers should prefer `circ_fit` with
`opts.Resample`.

**Signature.**
```matlab
obj = fit_circ_method(tbl, formula, Name, Value, ...)
```

**Inputs.**

| Name | Type | Meaning | Default |
|---|---|---|---|
| `tbl` | `table` | Data table with response, predictors, grouping var. | — |
| `formula` | char | Wilkinson string with a single `(1|<group>)` term. | — |
| `Method` | char | `'legacy'` (single fit), `'cboot'` (cluster bootstrap), `'sub80'` (subject subsample without replacement). | `'legacy'` |
| `B` | scalar | Number of resamples for `cboot`/`sub80`. | `60` |
| `KeepFrac` | scalar | Fraction of subjects kept for `sub80`. | `0.8` |
| `AutoShift` | logical | Pass through to `fitcirc_lme`. When bagging, the shift is computed once on the full data and frozen across bootstraps so $\beta$ is comparable. | `true` |
| `Verbose` | logical | Print progress. | `false` |

**Outputs.**

| Name | Type | Semantics |
|---|---|---|
| `obj` | `fitcirc_lme` | Fitted model. For `cboot`/`sub80`, `Beta`, `Coefficients.Estimate`, `cov_b`, `SE`, `tStat`, `pValue` are replaced with bagged values (componentwise-median $\beta$, empirical covariance across resamples). `LogLikelihood`, `DFE`, `ConvergenceHistory` etc. remain from the base fit on the full data. |

**Side effects.** None.

**Notes.**
- Under `'cboot'`, each picked subject is renumbered to a fresh unique ID so
  duplicates count as independent clusters in the random-intercept model.
- $\beta$ aggregation is the **componentwise median** across successful
  resamples; covariance is the empirical $\text{cov}$ across them
  (MLE-style, `cov(..., 1)`). $t$, $p$ are then $\beta_\text{bag} / \text{SE}$
  on $n_\text{subj} - 1$ degrees of freedom.
- If all resamples fail, the function warns and returns the legacy fit.

**Example.**
```matlab
mdl = fit_circ_method(tbl, 'Phase ~ 1 + Age^2 + (1|Subj_ID)', ...
                      'Method', 'cboot', 'B', 200, 'Verbose', true);
disp(mdl.Coefficients);
```

---

## circ_fit_fitcirc <a id="circ_fit_fitcirc"></a>

**Purpose.** `fitcirc_lme` backend adapter: orthogonal-polynomial design,
optional LRT order selection, optional resampled inference, trajectory + 95%
CI on an evaluation grid, cross-backend GOF, and an omnibus age-effect Wald
test — all wrapped into the unified `circ_result` schema.

**When to use.** Called automatically by `circ_fit` when
`backend == 'fitcirc_lme'`. Call directly if you want the schema-wrapped
result without parsing a Wilkinson formula.

**Signature.**
```matlab
result = circ_fit_fitcirc(tbl, opts)
```

**Inputs.**

| Name | Type | Meaning | Default |
|---|---|---|---|
| `tbl` | `table` | Data table; needs the response column (`opts.feature`), `opts.x_col`, `Subj_ID`. Optional `electrode`, `sex`, plus anything in `opts.categorical_varnames`. NaN rows in numeric used columns are dropped. | — |
| `opts.x_col` | char | Predictor column. | `'Age'` |
| `opts.feature` | char | Response column. | `''` (caller must supply) |
| `opts.categorical_varnames` | cellstr | Additional categorical main effects. | `{}` |
| `opts.xcol_categorical_interactions` | logical | Mask over `categorical_varnames`: which interact with the `x_col` polynomial block. | `[]` |
| `opts.Select` | logical | If true, scan polynomial orders 0..`MaxOrder` and select by LRT. | `false` |
| `opts.MaxOrder` | scalar | Maximum polynomial order considered when selecting. | `2` |
| `opts.Order` | scalar | Single order to fit when `Select == false`. | `MaxOrder` |
| `opts.Resample` | char | `'none'` / `'cboot'` / `'sub80'`. `'legacy'` aliases to `'none'`. | `'none'` |
| `opts.B` | scalar | Resample count for `cboot`/`sub80`. | `60` |
| `opts.KeepFrac` | scalar | Fraction of subjects retained for `sub80`. | `0.8` |
| `opts.eval_ages` | column vector | Grid for the trajectory. | `(7:80)'` |

**Outputs.**

| Name | Type | Semantics |
|---|---|---|
| `result` | struct | `circ_result` (see `result_schema.md`). Populates both required tier and optional tier (per-coefficient). |

**Side effects.** None on disk. Adds transient columns `<x_col>_op1`, …,
`<x_col>_opK` to its working copy of `tbl`.

**Notes.**
- **Orthogonal polynomial basis.** Internally uses
  `ortho_poly_basis(x, MaxOrder)` to build columns
  `<x_col>_op1`, …, `<x_col>_opK`. The raw `Age^k` power basis is replaced
  because raw columns are O(1)..O($10^5$) and cross-correlated > 0.99 over a
  typical lifespan range, which makes IRLS inside the EM jitter. The fitted
  curve and the joint Wald test are unchanged by this reparameterization
  (Wald is invariant under nonsingular linear reparameterization; see the
  inline citation to Lafontaine & White 1986 for why we keep the
  reparameterization strictly linear).
- **Warm-start chain across orders.** At each order, the EM is fit twice —
  once cold, once warm-started from the previous order's converged $\beta$
  (matched by coefficient name; new columns start at 0). The higher-LL fit
  wins. This handles the case where cold IRLS at higher orders converges to a
  worse local optimum than the lower order (which would otherwise cause the
  LRT to incorrectly reject the higher order).
- **LRT order selection rule.** Step up while each nested LRT
  $p < 0.05$; stop at the first non-significant step. Documented inline as
  matching the lme/blme path in `get_best_iterative_order.m`.
- **Trajectory CI.** $\eta = X\beta$ on the evaluation grid; standard error
  $\sqrt{\text{diag}(X\,\Sigma_\beta\, X^\top)}$ with $\Sigma_\beta$
  the cluster-robust sandwich; 95% CI is $\eta \pm 1.96\,\text{SE}$. The
  trajectory is unwrapped per electrode along the predictor.
- **Age-effect test.** Joint Wald that the entire `ContrastIndex.x_age` block
  (polynomial main effects + every age-interaction) is zero. Method
  reports as `'Wald'`.

**Example.**
```matlab
opts = struct('feature','Phase','x_col','Age','Select',true,'MaxOrder',2, ...
              'categorical_varnames',{{'electrode','sex'}}, ...
              'xcol_categorical_interactions',[true false]);
result = circ_fit_fitcirc(tbl, opts);
```

---

## fitcirc_lme <a id="fitcirc_lme"></a>

**Purpose.** The core estimator. A von Mises generalized linear mixed model
with one subject-level random intercept on the circle, fit by exact EM with
cluster-robust sandwich standard errors.

**When to use.** When you want the raw fitted model object (no order
selection, no schema wrapping) — for diagnostic plots, custom contrasts,
embedding in another estimator. Most users should reach `circ_fit` instead.

**Signature.**
```matlab
mdl = fitcirc_lme(tbl, formula, Name, Value, ...)
```

**Model.**

$$
\begin{aligned}
y_{ij}\mid \beta, \phi_i, \kappa &\sim \text{vonMises}(X_{ij}\beta + \phi_i,\ \kappa) \\
\phi_i &\sim \text{vonMises}(0,\ \kappa_\phi)
\end{aligned}
$$

Both the response noise and the subject offset are von Mises. Because of
that pairing, each subject's offset $\phi_i$ has an exact von Mises
distribution given the data and the parameters — no Gaussian (Laplace)
approximation and no need to unwrap angles onto the real line — and the
subject-to-subject spread is described directly on the circle by the
concentration $\kappa_\phi$ (large $\kappa_\phi$ = subjects are alike).

**Algorithm — high level.** EM with a closed-form E-step.

**E-step** (per subject $i$, given $\beta$, $\kappa$, $\kappa_\phi$). Compute
$S_i = \sum_j \sin(y_{ij} - X_{ij}\beta)$,
$C_i = \sum_j \cos(y_{ij} - X_{ij}\beta)$,
then $K_{i,\text{post}} = \sqrt{(\kappa C_i + \kappa_\phi)^2 + (\kappa S_i)^2}$
and $\mu_{i,\text{post}} = \text{atan2}(\kappa S_i,\ \kappa C_i + \kappa_\phi)$.
The mean resultant length $\rho_i = A(K_{i,\text{post}}) = I_1/I_0 \in [0,1)$
measures how sure we are of subject $i$'s offset (near 1 = very sure).

**M-step.**

- $\beta$: a circular regression (iteratively reweighted least squares)
  with offset $\mu_{i,\text{post}}$, weighting each observation by its
  subject's $\rho_i$.
- $\kappa$: the mean resultant length of the residual angles (each
  weighted by $\rho_i$), converted to a concentration.
- $\kappa_\phi$: maximize the exact one-dimensional objective
  $n_{\text{subj}}\bigl(R_\phi\,k - \log I_0(k)\bigr) + \log\mathrm{prior}(k)$,
  where $R_\phi = \mathrm{mean}_i\,\rho_i\cos\mu_{i,\text{post}}$ measures
  how tightly the estimated offsets bunch around 0. The prior keeps
  $\kappa_\phi$ finite in the case $R_\phi \to 1$ (offsets all collapse to
  0), which would otherwise drive $\kappa_\phi \to \infty$ and overflow the
  log-likelihood. See `KappaPhiPrior`.

**Marginal log-likelihood** (exact, no approximation):
$\log p(y_i) = \log I_0(K_{i,\text{post}}) - n_i \log(2\pi\,I_0(\kappa)) - \log I_0(\kappa_\phi)$,
reported unpenalized so it is comparable across nested models.

**Inference.** Cluster-robust ("sandwich") SEs on $\beta$ with each subject
as one cluster, built from the same $\rho_i$-weighted score. The bread is
the expected (Fisher) information $\kappa A(\kappa)\sum_{ij}\rho_i X_{ij}'X_{ij}$,
which is always positive-definite. Rescaled by
$\frac{m}{m-1}\cdot\frac{n-1}{n-p}$; tests use Student's $t$ with
$n_{\text{subj}} - 1$ degrees of freedom.

**Inputs.**

| Name | Type | Meaning | Default |
|---|---|---|---|
| `tbl` | `table` | Must contain the response (matching the LHS of the formula) and the grouping variable named inside the `(1|<group>)` term. | — |
| `formula` | char | Wilkinson formula. Must contain exactly one `(1|<group>)` term. Continuous predictors are added through `fitlme`'s design machinery. | — |
| `MaxIter` | scalar | Max EM iterations. | `100` |
| `Tol` | scalar | Relative-LL convergence tolerance. | `1e-5` |
| `Verbose` | logical | Print per-iteration progress. | `false` |
| `InitKappa` | scalar | Starting $\kappa$. | `4` |
| `InitKappaPhi` | scalar | Starting $\kappa_\phi$. | `4` |
| `InitSigma` | scalar | Deprecated; converted to `InitKappaPhi` via $A(\kappa_\phi) = e^{-\sigma^2/2}$. | `[]` |
| `KappaPhiPrior` | char | Weakly-informative prior on $\kappa_\phi$ that keeps its estimate finite: `'halfcauchy'`, `'halfnormal'`, or `'none'`. | `'halfcauchy'` |
| `KappaPhiPriorScale` | scalar | Scale of that prior on the $\kappa_\phi$ axis. Larger = weaker pull = more subject spread allowed. | `8` |
| `KappaPhiMax` | scalar | Optional hard upper limit on $\kappa_\phi$, applied after the prior. `inf` leaves the prior in charge; a finite value clamps $\kappa_\phi$ at that ceiling. | `inf` |
| `AutoShift` | logical | Rotate $y$ by the variance-minimizing shift (`circ_shift_min_var`) so wrapping doesn't split the response near $\pm\pi$. Recorded in `ThetaShift` and absorbed into the intercept after fitting. | `false` |
| `ThetaShift` | scalar | Explicit shift; overrides `AutoShift`. | `[]` |
| `InitBeta` | numeric | Warm-start $\beta$. Names matched against new design column names; absent columns start at 0. | `[]` |
| `InitBetaNames` | cellstr | Coefficient names for `InitBeta`. | `{}` |

**Outputs.** A `fitcirc_lme` object with properties:

| Property | Type | Semantics |
|---|---|---|
| `Formula`, `ResponseName`, `GroupingVar` | char | Echo of the parsed formula. |
| `Coefficients` | table | `{Name, Estimate, SE, tStat, pValue}` with cluster-robust SEs. |
| `Beta` | $p \times 1$ | Fixed-effects vector (intercept absorbs `ThetaShift`). |
| `Kappa` | scalar | Response noise concentration. |
| `KappaPhi` | scalar | Prior concentration on subject phase. |
| `SigmaPhi` | scalar | Equivalent circular SD, $\sqrt{-2\log A(\kappa_\phi)}$. |
| `PhiHat`, `PhiRho`, `PhiKappaPost` | $n_\text{subj} \times 1$ | Posterior mean direction, mean resultant length, concentration per subject. |
| `LogLikelihood` | scalar | Exact marginal LL at the converged params (unpenalized, so comparable across nested models). |
| `LogPrior` | scalar | Log $\kappa_\phi$ prior at the converged params; the penalized objective the M-step climbs is `LogLikelihood + LogPrior`. |
| `AIC`, `BIC` | scalar | Based on $p + 2$ free parameters. |
| `NumObservations`, `NumSubjects`, `NumCoefficients` | scalar | Counts. |
| `ConvergedIn`, `ConvergenceHistory` | scalar, vector | EM iteration count and per-iter LL trace. |
| `DesignNames` | cellstr | Design column names (matches `fitlme`). |
| `SubjectIDs` | column | Levels of the grouping variable in order. |
| `X_design` | $n \times p$ | Design matrix. |
| `TrainingData` | `table` | Internal copy with the shift applied. |
| `cov_b` | $p \times p$ | Cluster-robust sandwich covariance. |
| `DFE` | scalar | $n_\text{subj} - 1$. |
| `ContrastIndex` | struct | `.x_main` (polynomial main effects), `.x_x_<cat>` per interaction block, `.x_age` (union — used by the omnibus age-effect test). |
| `Rsquared` | struct | `.Ordinary`, `.Adjusted` (angle-scale $R^2 = 1 - \text{SSE}_\text{circ}/\text{SST}_\text{circ}$). |
| `ThetaShift` | scalar | Shift applied to the response before fitting; absorbed into the intercept post-fit. |

**Methods.**

- `predict(newdata, 'Conditional', false)` — predicted angle $\widehat{y} = \text{wrap}(X_\text{new}\beta)$.
  With `'Conditional', true`, adds the subject random intercept $\widehat\phi_i$ from
  `PhiHat` (subject must be among `SubjectIDs`).
- `coefTest(R)` — joint Wald test of $H_0:\ R\beta = 0$ using `cov_b`. `R` can be a contrast
  matrix, a vector of indices, or a name from `ContrastIndex` (e.g. `'x_age'`).
  Returns `struct(Fstat, pValue, df1, df2)`.

**Side effects.** None.

**Notes.**
- After fitting, the shift is baked back into the `(Intercept)` coefficient
  so $X\beta$ is on the original (unshifted) angle scale. Variance and CI
  half-widths are translation-invariant, so `cov_b` and joint Wald tests are
  unchanged.
- The fitter supports a single `(1|group)` random-intercept term only — no
  random slopes, no crossed grouping factors.
- Cluster-robust sandwich SEs can under-cover with very small per-cluster
  $n_j$ (e.g. $n_j = 2$). For primary inference with very small clusters,
  prefer the resample-based options through `circ_fit_fitcirc`.
- The `KappaPhiPrior` guards a boundary: when a variance component is zero
  the likelihood-ratio statistic does not follow the usual $\chi^2$
  (Stram & Lee 1994), and here the troublesome boundary is *zero* subject
  spread ($\kappa_\phi \to \infty$). A weakly-informative prior keeps the
  log-likelihood finite and smooth so model comparisons stay well defined.
  This mirrors Gelman's (2006) weakly-informative prior for a group-level
  standard deviation, here placed on $\kappa_\phi$ because the boundary of
  concern is the opposite one (zero spread rather than large spread).

**Example.**
```matlab
mdl = fitcirc_lme(tbl, 'Phase ~ 1 + Age + Age^2 + (1|Subj_ID)', ...
                  'AutoShift', true, 'Verbose', true);
disp(mdl);
ageBlock = mdl.coefTest('x_age');
fprintf('Any-age effect: F=%.2f, df=%d, p=%.2e\n', ...
        ageBlock.Fstat, ageBlock.df1, ageBlock.pValue);
```

**References.**
- Stram & Lee (1994), *Biometrics* 50(4):1171–1177. Variance components on
  the boundary of the parameter space.
- Gelman (2006), *Bayesian Analysis* 1(3):515–533. Weakly-informative
  priors for group-level scale parameters.

---

## circ_fit_config <a id="circ_fit_config"></a>

**Purpose.** Process-global configuration object (`persistent` variable
behind the function). Set once at the top of a pipeline; consumed by
`circ_fit` and by the stats stage of the lifespan pipeline without threading
options through every intermediate signature.

**When to use.** At the entry point of any batch pipeline that fits many
circ models with the same backend / sampler settings.

**Signature.**
```matlab
cfg = circ_fit_config()                    % get current config
cfg = circ_fit_config('get')               % same
circ_fit_config('set', struct(...))        % merge fields into the current cfg
circ_fit_config('reset')                   % restore defaults
```

**Inputs.**

| Name | Type | Meaning |
|---|---|---|
| action | char | `'get'` (default), `'set'`, `'reset'`. |
| struct (for `'set'`) | struct | Fields to merge into the current config. |

**Default config fields.**

| Field | Default | Meaning |
|---|---|---|
| `Backend` | `'fitcirc_lme'` | Default backend used by `circ_fit`. |
| `MaxOrder` | `2` | Polynomial-order cap. |
| `Select` | `true` | Run internal order selection. |
| `Resample` | `'none'` | `'none'`/`'cboot'`/`'sub80'` (fitcirc_lme only). |
| `Method` | `'legacy'` | Legacy alias for `Resample` (`'legacy'` → `'none'`). |
| `B` | `60` | Resample count. |
| `KeepFrac` | `0.8` | Subject-keep fraction for `sub80`. |
| `Chains`, `Iter`, `Warmup`, `Seed`, `AdaptDelta` | `4`, `2000`, `1000`, `1`, `0.95` | brms Stan sampler settings. |
| `Band` | `true` | lme4 CI band via `bootMer`. |
| `BrmsFallback` | `true` | If true, R-backend failures fall back to `fitcirc_lme`. |

**Outputs.**

| Name | Type | Semantics |
|---|---|---|
| `cfg` | struct | Returned for `'get'`, `'set'`, `'reset'`. |

**Side effects.** Mutates a `persistent` variable inside the function. The
state survives across calls within the same MATLAB session but resets with
`clear functions` or session restart.

**Example.**
```matlab
circ_fit_config('set', struct('Backend','brms','MaxOrder',3,'Chains',4));
% ... later, in any downstream function ...
cfg = circ_fit_config();
result = circ_fit(tbl, fml, cfg.Backend, cfg);
```

---

## circ_center <a id="circ_center"></a>

**Purpose.** Canonical preprocessing for every circ-regression backend:
subtract the circular mean so the seam sits in the data's natural gap.

**When to use.** Whenever you're about to fit a circular regression — done
automatically by all backends; expose it directly only when you need the
shift value yourself.

**Signature.**
```matlab
[theta_shift, y_shifted] = circ_center(y)
```

**Inputs.**

| Name | Type | Meaning |
|---|---|---|
| `y` | vector | Angles in radians (any range; NaN entries are excluded from the mean). |

**Outputs.**

| Name | Type | Meaning |
|---|---|---|
| `theta_shift` | scalar | Circular mean of $y$, $\text{atan2}(\overline{\sin y},\,\overline{\cos y})$, on $(-\pi,\pi]$. |
| `y_shifted` | vector | $\text{wrap}(y - \theta_\text{shift})$ on $(-\pi,\pi]$ (only computed if requested). |

**Side effects.** None.

**Notes.**
- After fitting on `y_shifted`, predictions are unshifted by adding
  $\theta_\text{shift}$ back and re-wrapping. Non-intercept coefficients are
  invariant under the shift; the intercept absorbs it.
- Compare `circ_shift_min_var`, which picks the seam location by minimizing
  the linear variance of the wrapped angles instead — retained for the legacy
  comparison harness.

**Example.**
```matlab
[shift, y0] = circ_center(tbl.Phase);
tbl.Phase = y0;
```

---

## circ_shift_min_var <a id="circ_shift_min_var"></a>

**Purpose.** Legacy alternative to `circ_center`: find the circular shift
that minimizes the *linear* variance of the wrapped angles, then apply it.

**When to use.** Multi-modal data where the largest empirical gap (rather
than the antipode of the circular mean) is the right seam location, or
backward-compatibility with old pipelines.

**Signature.**
```matlab
[theta_star, y_shifted] = circ_shift_min_var(y)
```

**Inputs.**

| Name | Type | Meaning |
|---|---|---|
| `y` | vector | Angles in radians. NaN entries excluded. |

**Outputs.**

| Name | Type | Meaning |
|---|---|---|
| `theta_star` | scalar | Shift on $(-\pi,\pi]$, optimized over a 0.5° grid (721 candidates). |
| `y_shifted` | vector | $\text{wrap}(y - \theta^*)$. |

**Side effects.** None.

**Notes.**
- For unimodal data, equivalent to the antipode of the circular mean.
- Used internally by `fitcirc_lme` when called with `'AutoShift', true`.

**Example.**
```matlab
[shift, y0] = circ_shift_min_var(tbl.Phase);
```

---

## circ_gof <a id="circ_gof"></a>

**Purpose.** Circular goodness-of-fit metrics shared by all backends. The
only directly cross-backend-comparable fit metrics (LL/AIC/BIC are on
different likelihood scales).

**When to use.** Any time you need $R^2_\text{circ}$ or mean absolute angular
error — including inside custom predict-and-score code.

**Signature.**
```matlab
gof = circ_gof(y, yhat)
gof = circ_gof(y, yhat, n_par)
```

**Inputs.**

| Name | Type | Meaning |
|---|---|---|
| `y` | vector | Observed angles (radians). |
| `yhat` | vector | Fitted/predicted angles at matching rows. |
| `n_par` | scalar | Number of fixed-effect parameters, for adjusted $R^2$. Optional. |

**Outputs.** Struct with fields:

| Field | Type | Meaning |
|---|---|---|
| `R2_circ` | scalar | $1 - \text{SSE}_\text{circ} / \text{SST}_\text{circ}$, where $\text{SSE}_\text{circ} = \sum (1 - \cos(\text{wrap}(y - \widehat y)))$ and $\text{SST}_\text{circ} = \sum (1 - \cos(\text{wrap}(y - \bar y_\text{circ})))$. |
| `R2_adj` | scalar | $1 - (1 - R^2)\,(n-1)/(n - n_\text{par})$ when `n_par` is given, else `NaN`. |
| `MAE_angular` | scalar | $\overline{\lvert\text{wrap}(y - \widehat y)\rvert}$. |

**Side effects.** None.

**Notes.**
- NaN rows in either $y$ or $\widehat y$ are dropped before scoring.
- This formula matches the one used inside the R workers
  (`circ_fit_common.R`), so `R2_circ` is comparable across all four backends.

**Example.**
```matlab
yhat = predict(mdl);
g = circ_gof(tbl.Phase, yhat, mdl.NumCoefficients);
fprintf('R2_circ=%.3f (adj=%.3f), MAE=%.2f rad\n', g.R2_circ, g.R2_adj, g.MAE_angular);
```

---

## ortho_poly_basis <a id="ortho_poly_basis"></a>

**Purpose.** R-style orthogonal polynomial basis of degree $k$ over a sample,
with a transform that can be reapplied at new $x$. Mirrors
`stats::poly(x, k)` (the centered, orthonormalized convention).

**When to use.** Whenever a polynomial-in-$x$ design is used in an iterative
fitter. Replaces the raw `[x, x^2, ..., x^k]` design to avoid the
conditioning + collinearity pathology in iterative estimators.

**Signature.**
```matlab
[P, info] = ortho_poly_basis(x, k)            % build new basis from x
 P        = ortho_poly_basis(x, k, info)      % apply existing basis to new x
```

**Inputs.**

| Name | Type | Meaning |
|---|---|---|
| `x` | vector | Sample values. |
| `k` | scalar | Maximum polynomial degree. |
| `info` | struct (optional) | Basis transform from a previous build; required for the "apply at new $x$" path. |

**Outputs.**

| Name | Type | Meaning |
|---|---|---|
| `P` | $n \times k$ | Orthogonal-polynomial design. `P(:,j)` is orthogonal to `P(:,j')` for $j \ne j'$ and to the constant column over the sample used to build the basis. |
| `info` | struct | `{means, R, k}` — the QR transform needed to reapply at new $x$. |

**Side effects.** None.

**Notes.**
- Build: $M = [x, x^2, ..., x^k]$; centered $M_c = M - \overline M$;
  $[Q, R] = \text{qr}(M_c, 0)$; $P = Q$; `info = {mean, R, k}`.
- Apply: $M_e$ from new $x$; $P_e = (M_e - \text{info.means})\,/\,\text{info.R}$.
- Per the inline docstring, the fitted curve, joint Wald tests, $R^2$,
  residuals and prediction intervals are unchanged by switching to this
  basis. Per-coefficient Wald p-values gain a standalone marginal
  interpretation.
- References (from the source file): Chambers & Hastie eds. (1992)
  *Statistical Models in S*; Kennedy & Gentle (1980) *Statistical Computing*.

**Example.**
```matlab
% Build at training x
[P_train, info] = ortho_poly_basis(tbl.Age, 3);
% Apply at evaluation grid
P_eval = ortho_poly_basis((7:80)', 3, info);
```

---

## make_circ_result <a id="make_circ_result"></a>

**Purpose.** Single source of truth for the result schema. Takes a struct
of raw fields from a backend, validates the required tier, fills the
optional tier with defaults, and returns the canonical `circ_result`.

**When to use.** From every backend adapter (`circ_fit_fitcirc`,
`read_circ_result`) — and from any custom estimator you bolt on that should
slot into the rest of the toolbox.

**Signature.**
```matlab
result = make_circ_result(s)
```

**Inputs.**

| Name | Type | Meaning |
|---|---|---|
| `s` | struct | Raw backend output. Must contain every field in the required tier. Optional-tier fields are filled with defaults if missing. |

**Outputs.**

| Name | Type | Meaning |
|---|---|---|
| `result` | struct | The validated, default-filled `circ_result`. See `result_schema.md`. |

**Validation performed.** Required-tier presence (errors on missing/empty);
type checks on `Trajectory` (table with columns `Age, mean, lo, hi`),
`OrderTable` (table), `GOF`/`AgeEffect`/`Diagnostics` (struct); presence of
GOF subfields `R2_circ, MAE_angular, LogLikelihood, AIC, BIC` and
AgeEffect subfields `pValue, Method`.

**Side effects.** None.

**Notes.**
- The required vs. optional split is structural: backends without a single
  $\beta$ vector (lme4, bpnreg) can't populate `Coefficients`/`Beta`/`cov_b`
  uniformly, so those fields are optional and remain empty for those
  backends.

**Example.**
```matlab
s = struct( ...
    'Backend',     'mybackend', ...
    'Formula',     fml, ...
    'ResponseName','Phase', ...
    'Order',        2, ...
    'ThetaShift',   0.0, ...
    'Trajectory',   Traj, ...
    'GOF',          GOF, ...
    'AgeEffect',    AgeEffect, ...
    'OrderTable',   OrderTable, ...
    'SelectedOrder',2, ...
    'SelectCriterion','LRT', ...
    'Diagnostics',  struct(), ...
    'Converged',    true);
result = make_circ_result(s);
```

---

## read_circ_result <a id="read_circ_result"></a>

**Purpose.** Reverse of `write_circ_contract`: read the R worker's output
files from a work directory and assemble a `circ_result`.

**When to use.** Called automatically by `circ_fit` after a successful R
backend run. Call directly to re-read a cached result without rerunning the
fit.

**Signature.**
```matlab
result = read_circ_result(work_dir, backend, meta)
```

**Inputs.**

| Name | Type | Meaning |
|---|---|---|
| `work_dir` | char | Directory containing the R worker's output files. |
| `backend` | char | `'brms'`, `'lme4'`, or `'bpnreg'`. |
| `meta` | struct | Metadata returned by `write_circ_contract` (formula, feature, theta_shift, x_col, …). |

**Files read.**

- `<backend>_predictions.csv` — `Age, electrode, sex, mean, lo, hi` (unwrapped).
- `<backend>_stats.json` — `LL, R2_circ, mae_angular, AIC, BIC, n_obs, n_subj, chosen_order, select_criterion, age_effect{pValue,stat,df,method}, diagnostics`.
- `<backend>_order_table.csv` — `order, n_par, LogLikelihood, R2_circ, criterion_value, selected`.
- `<backend>_coefs.csv` + `<backend>_cov_b.csv` — brms only, populates the optional per-coefficient tier.

**Outputs.**

| Name | Type | Meaning |
|---|---|---|
| `result` | struct | `circ_result`. `Converged` is `true` unless `Diagnostics.rhat_max >= 1.05`, `Diagnostics.divergent > 0`, or `Diagnostics.converged == false`. |

**Side effects.** Reads files; does not write.

**Notes.**
- For backends other than brms, no per-coefficient files are written, so the
  optional tier (`Coefficients`, `Beta`, `cov_b`, `ContrastIndex`,
  `NumCoefficients`, `DFE`) is left empty.

**Example.**
```matlab
result = read_circ_result('/path/to/work', 'brms', meta);
```

---

## write_circ_contract <a id="write_circ_contract"></a>

**Purpose.** Write the `data.csv` / `eval_grid.csv` / `meta.json` contract
that the R worker reads. Centralizes circular-mean shifting, NaN-row
dropping, single-level-factor dropping, and full Wilkinson formula assembly.

**When to use.** Called automatically by `circ_fit` for R backends. Use
directly to set up a work directory and inspect what the R worker will see.

**Signature.**
```matlab
meta = write_circ_contract(T, feature, order, results_dir, opts)
```

**Inputs.**

| Name | Type | Meaning | Default |
|---|---|---|---|
| `T` | `table` | Data table with the predictor (`opts.x_col`), the response (`feature`), `Subj_ID`, and optionally `electrode`, `sex`. | — |
| `feature` | char | Response column name. | — |
| `order` | scalar | Max (`Select == true`) or single polynomial order. | — |
| `results_dir` | char | Output directory. Created if missing. | — |
| `opts.Shift` | char | `'circmean'` (default; uses `circ_center`) or `'minvar'` (uses `circ_shift_min_var`). | `'circmean'` |
| `opts.x_col` | char | Predictor name. | `'Age'` |
| `opts.eval_ages` | vector | Prediction grid. | `(7:80)'` |
| `opts.backend` | char | Echoed into `meta.backend`. | `''` |
| `opts.select` | logical | Whether the R worker should sweep orders. | `true` |
| `opts.max_order` | scalar | Max order in the sweep. | `order` |
| `opts.chains` / `iter` / `warmup` / `seed` / `adapt_delta` / `band` | various | Sampler options echoed into `meta`. | brms defaults |
| `opts.dump` | char | Provenance path stored in `meta.dump`. | `''` |

**Outputs.**

| Name | Type | Meaning |
|---|---|---|
| `meta` | struct | Everything written to `meta.json`: `{formula, x_col, feature, order, has_electrode, has_sex, theta_shift, shift_kind, backend, select, max_order, chains, iter, warmup, seed, adapt_delta, band, dump}`. |

**Files written.**

- `<results_dir>/data.csv` — columns `y` (shifted response), `<x_col>`,
  optional `electrode`, optional `sex`, `Subj_ID`.
- `<results_dir>/eval_grid.csv` — the prediction grid (ages × electrode
  levels × sex 0).
- `<results_dir>/meta.json` — `jsonencode(meta)`.

**Notes.**
- Drops rows with NaN in any used numeric column.
- Drops single-level factors automatically and logs to stdout.
- The response is renamed to `y` and pre-shifted by `theta_shift` before
  writing; predictions are unshifted (`y + theta_shift`) downstream.
- Full formula assembled inline: `y ~ <x_col>^<order> [* electrode] [+ sex] + (1|Subj_ID)`.

**Example.**
```matlab
meta = write_circ_contract(tbl, 'Phase', 2, '/tmp/circ', ...
                           struct('backend','brms','chains',4,'iter',2000));
```

---

## plot_circ_fit <a id="plot_circ_fit"></a>

**Purpose.** Plot one or more `circ_result` trajectories with the triple-line
trick — each curve drawn at $y$, $y + 2\pi$, $y - 2\pi$ — so the $\pm\pi$
seam never produces a visible jump. Overlays multiple backends if given a
cell array.

**When to use.** Whenever you want to visualize a fit. The "compare four
backends on the same axes" pattern is the canonical use.

**Signature.**
```matlab
ax = plot_circ_fit(result, tbl)
ax = plot_circ_fit({r1, r2, ...}, tbl, opts)
```

**Inputs.**

| Name | Type | Meaning | Default |
|---|---|---|---|
| `results` | struct or cell of structs | One or more `circ_result`s. | — |
| `tbl` | `table` | Data table for the scatter (needs `opts.x_col` + the response). May be empty. | — |
| `opts.ax` | axes handle | Axes to draw into. | new figure |
| `opts.feature` | char | Response column. | `results{1}.ResponseName` |
| `opts.x_col` | char | Predictor column. | `'Age'` |
| `opts.plot_CI` | logical | Draw CI patches. | `true` |
| `opts.scatter` | logical | Draw the data scatter. | `true` |
| `opts.colors` | $N \times 3$ | RGB per result. | `lines(N)` |
| `opts.labels` | cellstr | Legend labels per result. | each result's `Backend` |

**Outputs.**

| Name | Type | Meaning |
|---|---|---|
| `ax` | axes handle | The axes drawn into. |

**Side effects.** Creates a figure (and axes) if `opts.ax` is not supplied.

**Notes.**
- Trajectories are plotted in the original (unshifted) angle frame, matching
  the raw scatter.
- The data scatter is also triple-plotted at $\pm 2\pi$ at low alpha so the
  visual continuity matches the curves.
- Electrode level 0 is drawn solid, level 1 dashed (if `electrode` is in the
  trajectory).
- $y$-limits fixed to $[-\pi,\pi]$ with ticks at $-\pi, 0, \pi$.
- CI patches are drawn only when `Trajectory.hi != Trajectory.mean` (i.e.
  when the backend supplied a band).

**Example.**
```matlab
ax = plot_circ_fit({r1, r2, r3, r4}, tbl, ...
    struct('feature','Phase','x_col','Age','plot_CI',true));
title(ax, 'Phase ~ Age — four-backend overlay');
```

---

## Appendix A: utils/

Small, stable helpers copied here so the toolbox is self-contained.

| File | One-liner |
|---|---|
| `utils/build_model_formula.m` | Build a Wilkinson formula string from `{order, x_col, feature, categorical_varnames, xcol_categorical_interactions}` — the (`1|Subj_ID`) random-intercept term is appended by the caller. Used by both the circular dispatcher and the linear LME path so they assemble identical formulas. |
| `utils/get_LLR.m` | Chi-square likelihood-ratio-test $p$-value from two log-likelihoods and their residual degrees of freedom: $p = 1 - \chi^2_\text{cdf}(-2(\text{LL}_1 - \text{LL}_2),\ \text{DFE}_1 - \text{DFE}_2)$. |

## Appendix B: R/

R-side worker for the brms / lme4 / bpnreg backends. Invoked from MATLAB as
`Rscript R/circ_fit.R <work_dir> <backend>`.

| File | Role |
|---|---|
| `R/circ_fit.R` | Entry point: read `data.csv` / `eval_grid.csv` / `meta.json`, dispatch to the per-backend impl, finalize and write outputs. |
| `R/circ_fit_common.R` | Shared helpers: `wrap_pi`, unwrap, $R^2_\text{circ}$, the Wilkinson-name remap that puts each backend's native coefficient names back into `(Intercept)` / `Age^k` / `Age^k:cat` form, IO helpers. |
| `R/circ_fit_brms_impl.R` | brms Stan vM-GLMM, `tan_half` link, LOO order selection (step up while `elpd_diff > 2 * se_diff`) plus a chi-square LRT reported alongside. Only backend that writes per-coefficient files (`<backend>_coefs.csv`, `<backend>_cov_b.csv`). |
| `R/circ_fit_lme4_impl.R` | Parallel sin/cos `lmer` fits with combined-LL LRT for order selection and the age-effect test. No single-$\beta$ vector. |
| `R/circ_fit_bpnreg_impl.R` | `bpnreg::bpnme` projected-normal mixed model with WAIC for order selection. No single-$\beta$ vector. |

## Appendix C: tests <a id="tests"></a>

One line per test file in `tests/`. None are deeply documented here —
read the file's header for the full picture.

| Test file | Covers |
|---|---|
| `tests/test_circ_fit_schema.m` | Schema parity + wiring smoke test: every backend produces the required-tier fields, `AgeEffect.pValue` populated, `Trajectory` unwrapped, downstream `get_model_fit` wiring works. |
| `tests/sim_circ_compare.m` | Fits and overlays all four backends on a noisy synthetic wrapping-trajectory dataset. (Demo, not a strict assertion test.) |
| `tests/test_fitcirc_lme_autoshift.m` | Round-trip and seam-crossing test for `AutoShift`: predictions and CI half-widths match between shifted and unshifted fits. |
| `tests/test_fitcirc_lme_recovery.m` | Parameter recovery on simulated data: $\beta$, $\kappa$, $\kappa_\phi$ recovered to tolerance. |
| `tests/test_joint_test.m` | Wald block-test sanity check: `x_main` joint Wald is true-positive on signal, true-null on shuffled covariate. |
| `tests/test_resample_compare.m` | Compares legacy / cboot / sub80 on leverage-subject contamination. |
