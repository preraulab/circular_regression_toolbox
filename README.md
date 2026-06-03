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
addpath(genpath('/path/to/circ_fit_toolbox'));
run(fullfile('/path/to/circ_fit_toolbox', 'tutorial.m'));
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

## Backend chooser

Full side-by-side comparison in [`docs/backends.md`](docs/backends.md). Quick
chooser:

| Backend         | Model                                 | Order selection                | Random effects               | Uncertainty           | Wraps full revolution? | Dependencies                |
|-----------------|---------------------------------------|--------------------------------|------------------------------|-----------------------|------------------------|-----------------------------|
| **fitcirc_lme** | von Mises GLMM, exact EM              | LRT                            | Single `(1|group)` intercept | Cluster-robust Wald (sandwich) | No                     | MATLAB only                 |
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
| [`circ_vmrnd`](circ_vmrnd.m) | Best–Fisher 1979 rejection sampler for the von Mises distribution. Used by the tutorial and the parameter-recovery tests. |
| [`ortho_poly_basis`](docs/functions.md#ortho_poly_basis) | R-style orthonormal polynomial basis with a transform that can be reapplied at new $x$. |
| [`make_circ_result`](docs/functions.md#make_circ_result) | Result-struct factory + validator (single source of truth for the schema). |
| [`read_circ_result`](docs/functions.md#read_circ_result) | Read the R worker's output contract from disk and assemble a `circ_result`. |
| [`write_circ_contract`](docs/functions.md#write_circ_contract) | Write `data.csv` / `eval_grid.csv` / `meta.json` for the R worker. |
| [`plot_circ_fit`](docs/functions.md#plot_circ_fit) | Plot one or more results with the triple-line trick for seam-free display. |
| [`tutorial.m`](tutorial.m) | Self-contained simulate-and-recover walkthrough; see the [Tutorial section](#tutorial-simulate-fit-recover). |

## Repository layout

```
circ_fit_toolbox/
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
