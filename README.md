# circ_fit_toolbox

A reusable MATLAB toolbox for **regression of circular (angular) outcomes** on
linear predictors, with optional subject random effects, a uniform result
schema across four estimator backends, and plotting that handles the $\pm\pi$
seam by construction.

The toolbox solves a single recurring problem: phase / angle / direction
outcomes don't behave well in standard linear-mixed-model pipelines — the
$\pm\pi$ seam creates phantom discontinuities, `fitlme` ignores periodicity,
and the existing R packages (`brms`, `bpnreg`, `lme4`'s sin/cos approach) each
speak a different dialect. `circ_fit` wraps four estimators (the in-house
`fitcirc_lme` von-Mises GLMM, Bayesian `brms`, frequentist sin/cos `lme4`, and
Bayesian projected-normal `bpnreg`) behind one MATLAB-side dispatcher, a
single Wilkinson-formula signature, and a single output struct schema.

## Quick start

```matlab
addpath(genpath('circ_fit_toolbox'));

% tbl needs: <response>, <predictor>, Subj_ID (and optionally electrode, sex).
% Response is an angle in radians; the toolbox centers by circular mean
% internally so range is irrelevant.
result = circ_fit(tbl, 'Phase ~ 1 + Age^2 + (1|Subj_ID)', 'fitcirc_lme', ...
                  struct('Select', true, 'MaxOrder', 2));

result.SelectedOrder      % polynomial order picked by LRT
result.AgeEffect.pValue   % omnibus joint test: all age-involving terms = 0
result.GOF.R2_circ        % angular R^2, comparable across backends
result.Trajectory         % age × electrode × {mean, lo, hi} eval-grid table
plot_circ_fit(result, tbl);   % triple-line at ±2π so the seam never jumps
```

To overlay multiple backends on the same data (sensitivity / robustness check):

```matlab
r1 = circ_fit(tbl, fml, 'fitcirc_lme', opts);
r2 = circ_fit(tbl, fml, 'brms',        opts);
r3 = circ_fit(tbl, fml, 'lme4',        opts);
r4 = circ_fit(tbl, fml, 'bpnreg',      opts);
plot_circ_fit({r1, r2, r3, r4}, tbl);
```

## The uniform result schema

Every backend returns a struct validated by `make_circ_result`. Full
field-by-field reference is in [`docs/result_schema.md`](docs/result_schema.md);
the highlights:

**Required of every backend**
- `Backend`, `Formula`, `ResponseName`, `Order`, `ThetaShift`
- `Trajectory` — table `{Age, electrode, sex, mean, lo, hi}`, unwrapped per electrode
- `GOF` — `{R2_circ, MAE_angular, LogLikelihood, AIC, BIC}` (AIC/BIC are NaN for the Bayesian backends)
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

## Backend chooser

Full side-by-side comparison in [`docs/backends.md`](docs/backends.md). Quick
chooser:

| Backend         | Model                                 | Order selection                | Random effects               | Uncertainty           | Wraps full revolution? | Dependencies                |
|-----------------|---------------------------------------|--------------------------------|------------------------------|-----------------------|------------------------|-----------------------------|
| **fitcirc_lme** | von Mises GLMM, exact EM              | LRT                            | Single `(1|group)` intercept | Cluster-robust Wald (Liang–Zeger) | No                     | MATLAB only                 |
| **brms**        | Bayesian vM-GLMM, `tan_half` link     | LOO (`elpd_diff > 2·se_diff`); LRT reported alongside | brms-side  | Posterior (Stan)      | No                     | R + `brms` + `loo` + Stan toolchain |
| **lme4**        | Sin/cos parallel LMEs                 | Combined sin+cos LRT           | lme4-side `(1|Subj_ID)`      | Wald + optional `bootMer` band | Yes                    | R + `lme4`                  |
| **bpnreg**      | Bayesian projected-normal mixed model | WAIC                           | bpnreg-side                  | Posterior              | Yes                    | R + `bpnreg`                |

The "Age effect" significance statistic is, for every backend, an omnibus
test that **all** age-involving terms (`Age`, `Age^k`, `Age:cat`, `Age^k:cat`)
are simultaneously zero — one number per model.

### When to switch backends

- **Trajectory stays within ~one arc** (less than about half a revolution):
  `fitcirc_lme` is the natural choice. The seam is handled by circular-mean
  centering and the von Mises likelihood; sandwich SEs give correct
  cluster-robust inference.
- **Trajectory wraps a full revolution**: use `lme4` or `bpnreg`. Both
  represent the mean via two components (sin/cos or projected-normal) that
  can sweep through $2\pi$. The vM-based estimators (`fitcirc_lme`, `brms`)
  cannot: a von Mises mean is a single circular point and the `tan_half` link
  is bounded to one revolution.
- **You want Bayesian inference plus LRT-equivalent comparisons**: `brms`
  reports LOO and chi-square LRT side-by-side per polynomial step.

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
| [`ortho_poly_basis`](docs/functions.md#ortho_poly_basis) | R-style orthonormal polynomial basis with a transform that can be reapplied at new $x$. |
| [`make_circ_result`](docs/functions.md#make_circ_result) | Result-struct factory + validator (single source of truth for the schema). |
| [`read_circ_result`](docs/functions.md#read_circ_result) | Read the R worker's output contract from disk and assemble a `circ_result`. |
| [`write_circ_contract`](docs/functions.md#write_circ_contract) | Write `data.csv` / `eval_grid.csv` / `meta.json` for the R worker. |
| [`plot_circ_fit`](docs/functions.md#plot_circ_fit) | Plot one or more results with the triple-line trick for seam-free display. |

## Repository layout

```
circ_fit_toolbox/
├── README.md                  (this file)
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
git submodule add git@github.com:preraulab/circ_fit_toolbox.git path/to/circ_fit_toolbox
git submodule update --init --recursive
```

In MATLAB:

```matlab
addpath(genpath('path/to/circ_fit_toolbox'));
```

### Standalone

```sh
git clone git@github.com:preraulab/circ_fit_toolbox.git
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
  $\theta_\text{shift} = \operatorname{atan2}\!\bigl(\overline{\sin y},\ \overline{\cos y}\bigr)$
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

- **EM marginal-LL non-monotonicity at the random-effects concentration
  boundary.** When `fitcirc_lme`'s `KappaPhi` posterior collapses near 0, the
  unconstrained M-step drives `KappaPhi` toward infinity. The fitter applies
  a default cap (`KappaPhiMax = 8`) to prevent boundary blow-up, but the
  marginal LL can still be non-monotone in polynomial order in pathological
  clusters. The symptom is `AgeEffect.pValue = NaN` together with a visible
  $R^2_\text{circ}$ jump in the order table. `circ_fit_fitcirc` reports
  $R^2_\text{circ}$ alongside log-likelihood in the order table so this is
  visible to downstream audits. See the inline note in `circ_fit_fitcirc.m`
  citing Stram & Lee (1994).
- **Cluster-robust sandwich SEs underperform with very small clusters.** The
  Liang–Zeger sandwich used in `fitcirc_lme` is correct asymptotically but
  can under-cover when each subject contributes only two observations.
  Documented in the source header.
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
