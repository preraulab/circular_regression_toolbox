# The `circ_result` struct

Every backend in the toolbox returns a struct validated by
`make_circ_result`. This page documents every field — name, type, units,
which backends populate it, and what semantic guarantees the field carries.

There are two tiers:

- **Required tier** — present and non-empty on every backend.
- **Optional tier** — populated only when the backend exposes a single
  linear-predictor coefficient vector (currently `fitcirc_lme` and `brms`).
  For `lme4` and `bpnreg`, optional-tier fields are filled with the
  defaults from `make_circ_result` (empty `[]` / `{}` / `NaN` / empty
  struct) and should be ignored.

---

## Required tier

| Field | Type | Units | Populated by | Meaning |
|---|---|---|---|---|
| `Backend` | char | — | all | `'fitcirc_lme'`, `'brms'`, `'lme4'`, or `'bpnreg'`. |
| `Formula` | char | — | all | Wilkinson formula actually fit. For `fitcirc_lme` this is the orthogonal-polynomial form (`Age_op1 + Age_op2 + ...`); for the R backends it is the polynomial form (`Age^k`). |
| `ResponseName` | char | — | all | Name of the response column in the input table. |
| `Order` | scalar | — | all | Selected (or fixed) polynomial order in the predictor. |
| `ThetaShift` | scalar | radians | all | Circular-mean (or min-variance) shift applied to the response before fitting. For `fitcirc_lme` the shift is absorbed into the intercept post-fit, so $X\beta$ is on the original (unshifted) scale; `ThetaShift` is then carried as metadata only. For R backends, the response was pre-shifted before writing `data.csv` and predictions were unshifted on read-back; same metadata role. |
| `Trajectory` | `table` | predictor + radians | all | Population/marginal prediction on an evaluation grid. Columns: `Age` (predictor; raw scale), `electrode` (numeric: 0, optionally 1 — duplicated rows per level), `sex` (numeric; 0 when sex is a covariate), `mean`, `lo`, `hi` (response in radians; `lo`/`hi` = 95% CI / 95% credible interval / `mean` if no band). **Unwrapped per electrode** — `abs(diff(mean)) < pi` is guaranteed within each electrode level. |
| `GOF` | struct | — | all | Goodness-of-fit. See subfields below. |
| `AgeEffect` | struct | — | all | Single omnibus "any age effect" test. See subfields below. |
| `OrderTable` | `table` | — | all | Per-order summary used by the selection rule. Columns: `order` (scalar), `n_par` (scalar), `LogLikelihood` (scalar), `R2_circ` (scalar), `criterion_value` (scalar; LRT $p$ / LOO $\Delta\text{elpd}$ / WAIC / etc.), `selected` (logical; exactly one row is `true`). |
| `SelectedOrder` | scalar | — | all | The order in `OrderTable` flagged `selected`. |
| `SelectCriterion` | char | — | all | `'LRT'` (fitcirc_lme), `'LOO'` (brms), `'LRT-sincos'` (lme4), `'WAIC'` (bpnreg), or `'none'`. |
| `Diagnostics` | struct | — | all | Backend-specific. See subsection below. |
| `Converged` | logical | — | all | Whether the fitter converged. For Bayesian backends, combines diagnostic checks (`rhat_max < 1.05`, `divergent == 0`, etc.). |

### `GOF` subfields

| Subfield | Type | Units | Meaning |
|---|---|---|---|
| `R2_circ` | scalar | — | $1 - \text{SSE}_\text{circ} / \text{SST}_\text{circ}$ with circular dispersion $\sum(1 - \cos(\text{resid}))$. **Comparable across backends.** |
| `R2_adj` | scalar | — | $1 - (1 - R^2)(n-1)/(n - n_\text{par})$ (only populated by `circ_fit_fitcirc`; absent for R backends). |
| `MAE_angular` | scalar | radians | $\overline{\lvert\operatorname{wrap}(y - \widehat y)\rvert}$. **Comparable across backends.** |
| `LogLikelihood` | scalar | log-units | Exact marginal LL (fitcirc_lme) / posterior log-density at the mean (brms) / sum of sin- and cos-component LLs (lme4) / marginal LL approximation (bpnreg). **Within-backend only** — likelihood families differ. |
| `AIC` | scalar | — | $-2\,\text{LL} + 2k$. `NaN` for Bayesian backends if not natively computed. |
| `BIC` | scalar | — | $-2\,\text{LL} + k\log n$. `NaN` for Bayesian backends if not natively computed. |

### `AgeEffect` subfields

| Subfield | Type | Units | Meaning |
|---|---|---|---|
| `pValue` | scalar | — | Single $p$ for "all age-involving terms simultaneously zero". For Bayesian backends this is interpreted as either a posterior tail probability (brms) or a WAIC-derived analog (bpnreg). |
| `stat` | scalar | — | Test statistic ($F$ for fitcirc_lme Wald; $\chi^2$ for LRT; $\Delta$WAIC for bpnreg). May be `NaN` when not applicable. |
| `df` | scalar | — | Numerator degrees of freedom. May be `NaN` for Bayesian backends. |
| `Method` | char | — | `'Wald'` (fitcirc_lme), `'LRT'` / `'LRT-sincos'` (lme4), `'LOO'` / `'Bayes'` (brms), `'bpnreg'` (bpnreg). |

### `Diagnostics`

Schema-free struct. Common subfields:

| Backend | Typical fields |
|---|---|
| `fitcirc_lme` | `ConvergedIn` (EM iterations), `Resample` (`'none'`/`'cboot'`/`'sub80'`), `Kappa`, `KappaPhi`. |
| `brms` | `rhat_max`, `divergent`, plus brms-native diagnostics passed through `<backend>_stats.json`. |
| `lme4` | lme4-native convergence flags. |
| `bpnreg` | bpnreg-native convergence/sampler diagnostics. |

---

## Optional tier (per-coefficient inference)

Populated only by `fitcirc_lme` and `brms`. For `lme4` and `bpnreg`, these
fields are filled with the defaults from `make_circ_result` (empty `[]` /
`{}` / `NaN` / empty struct).

| Field | Type | Units | Populated by | Meaning |
|---|---|---|---|---|
| `Coefficients` | `table` | — | fitcirc_lme, brms | Columns: `Name` (Wilkinson grammar — `'(Intercept)'`, `'Age'`, `'Age^2'`, `'Age^2:electrode'`, …), `Estimate` (radians for the intercept, radian-per-unit-predictor for slopes), `SE`, `pValue`. For `fitcirc_lme`, `SE` is the cluster-robust sandwich SE and `pValue` is two-sided Student-$t$ on $n_\text{subj}-1$ df. For `brms`, `SE` is the posterior SD and `pValue` is the two-sided posterior tail probability. |
| `CoefficientNames` | cellstr | — | fitcirc_lme, brms | Convenience copy of `Coefficients.Name`. |
| `Beta` | $p \times 1$ | radians / radians-per-unit | fitcirc_lme, brms | Coefficient vector aligned with `CoefficientNames`. For `fitcirc_lme`, the intercept has absorbed `ThetaShift`. |
| `cov_b` | $p \times p$ | — | fitcirc_lme, brms | Covariance of $\beta$. Cluster-robust sandwich for `fitcirc_lme`; posterior covariance for `brms`. |
| `ContrastIndex` | struct | — | fitcirc_lme, brms (partial) | Index map into `Beta` for joint Wald tests. `.x_main` is the vector of indices of polynomial main-effect columns. `.x_x_<cat>` is the index vector for each interaction block. `.x_age` is the union (used by the omnibus age-effect test). For `brms`, only `.x_main` is built by `read_circ_result`; for `fitcirc_lme`, all blocks are built inside the model constructor. |
| `NumCoefficients` | scalar | — | fitcirc_lme, brms | Number of fixed-effect parameters $p$. |
| `NumObservations` | scalar | — | all (when known) | Total observations used in the fit (after NaN drop). |
| `NumSubjects` | scalar | — | all (when known) | Number of distinct subject groupings used. |
| `DFE` | scalar | — | fitcirc_lme, brms | Error degrees of freedom — $n_\text{subj} - 1$ for both. |
| `WorkDir` | char | — | brms, lme4, bpnreg | Directory holding the R worker IO contract; useful for caching / debugging. Empty for `fitcirc_lme`. |
| `Raw` | object | — | fitcirc_lme | The native fitter handle (a `fitcirc_lme` instance). Empty for R backends — re-instantiating their native objects from a serialized form would require the full R worker process and is out of scope. |

---

## Trajectory unwrapping guarantee

Each backend unwraps its `Trajectory.mean` per electrode along the predictor
**before writing the trajectory into the result**. The downstream plotter
(`plot_circ_fit`) therefore does no break-at-jumps logic. The contract
enforced by `test_circ_fit_schema` is: within each electrode level,
`max(abs(diff(Trajectory.mean))) < pi`.

If you write a new backend, your trajectory build must include the unwrap
step or `test_circ_fit_schema` will fail.

---

## Comparing across backends

The only fields whose values can be directly compared between backends:

- `GOF.R2_circ` and `GOF.MAE_angular` — computed identically on the angle
  scale by every backend (see `circ_gof.m` and `R/circ_fit_common.R`).
- `SelectedOrder` — the same number means the same polynomial degree in
  every backend.
- `Trajectory.mean` — all on the same radian scale, in the original
  (unshifted) frame, with the same evaluation grid.

Fields **not** directly comparable across backends:

- `GOF.LogLikelihood`, `GOF.AIC`, `GOF.BIC` — different likelihood families.
- `AgeEffect.stat` / `df` — different test statistics.
- `Coefficients.Estimate` — comparable in sign and order of magnitude, but
  `fitcirc_lme` uses an orthogonal-polynomial basis internally (column
  names like `Age_op1`) while `brms` uses a raw polynomial in standardized
  Age (column names like `polyAge_z2EQ1`), so direct numeric comparison of
  the slopes requires mapping back through the basis transformation.
