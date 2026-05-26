# circ_fit_toolbox

A reusable MATLAB toolbox for **regression of circular (angular) outcomes** —
e.g. a phase trajectory across age — with a uniform interface across four
estimator backends so they can be swapped in/out without changing the calling
code, and plotting that handles the ±π seam by construction.

The core fitter is **`fitcirc_lme`**: a von Mises generalized linear mixed
model fit by EM, with a subject random intercept and cluster-robust (Liang–Zeger
sandwich) standard errors. Around it sits a thin dispatcher (`circ_fit`) that
exposes the same input signature and same output schema for several alternative
estimators — useful for sensitivity checks and for choosing the right tool when
your data wraps a meaningful fraction of the circle.

## Why this exists

Mixing circular outcomes with the usual linear-mixed-model workflow is awkward:
- the ±π seam creates phantom discontinuities in fits and plots,
- standard `fitlme` ignores the periodicity,
- and existing R packages (`brms`, `bpnreg`, `lme4`'s sin/cos approach) each
  speak a different dialect for inputs, outputs, model comparison, and plotting.

This toolbox solves three things at once:

1. **One signature, one output schema** across four backends so a paper can
   show results under any one of them (or all of them as a robustness check).
2. **Circular-mean centering as the canonical preprocessing**, so the seam sits
   in a data gap by default — the practical fix Stan developers recommend for
   the seam-multimodality problem in von Mises mixed models.
3. **A schema-native plotter** that draws each trajectory as three offset lines
   (`y`, `y+2π`, `y−2π`) plus matching CI bands — eliminating seam jumps from
   the figure entirely, without relying on ad-hoc break-at-jumps logic.

## Quick start

```matlab
% MATLAB
addpath(genpath('circ_fit_toolbox'));

% A table with the response, a numeric predictor, and a grouping factor:
%   tbl.Phase, tbl.Age, tbl.Subj_ID   (optional: tbl.electrode, tbl.sex)
result = circ_fit(tbl, 'Phase ~ 1 + Age^2 + (1|Subj_ID)', 'fitcirc_lme', ...
                  struct('Select', true, 'MaxOrder', 2));

% Inspect
result.SelectedOrder           % polynomial order picked by LRT
result.AgeEffect.pValue        % omnibus joint test that age (all terms) is zero
result.GOF.R2_circ             % circular R^2
result.Trajectory              % age x electrode x {mean, lo, hi} eval-grid table

% Plot (triple-line at ±2π so the seam never jumps)
plot_circ_fit(result, tbl);
```

To overlay multiple backends on the same data (e.g. for the paper's robustness
panel):

```matlab
r1 = circ_fit(tbl, fml, 'fitcirc_lme', opts);   % native vM-GLMM
r2 = circ_fit(tbl, fml, 'brms',        opts);   % Stan vM-GLMM, LOO selection
r3 = circ_fit(tbl, fml, 'lme4',        opts);   % sin/cos LMEs (projected-Gaussian)
r4 = circ_fit(tbl, fml, 'bpnreg',      opts);   % Bayesian projected-normal
plot_circ_fit({r1, r2, r3, r4}, tbl);
```

## The uniform result schema

Every backend returns a struct validated by `make_circ_result` with:

**Required of every backend**
- `Backend`, `Formula`, `ResponseName`, `Order`, `ThetaShift`
- `Trajectory` — table `{Age, electrode, sex, mean, lo, hi}`, unwrapped per electrode
- `GOF` — `{R2_circ, MAE_angular, LogLikelihood, AIC, BIC}` (AIC/BIC NaN for Bayesian)
- `AgeEffect` — `{pValue, stat, df, Method}` — a single "any age effect" test
- `OrderTable` — per-order LL / R2 / criterion / `selected`
- `SelectedOrder`, `SelectCriterion` (`LRT` | `LRT-sincos` | `LOO` | `WAIC`)
- `Diagnostics`, `Converged`

**Optional (populated only when the backend has a single linear-predictor coef vector — i.e. fitcirc_lme + brms)**
- `Coefficients` (table `{Name, Estimate, SE, pValue}`, Wilkinson grammar), `Beta`, `cov_b`, `ContrastIndex`, `CoefficientNames`, `NumCoefficients`

The two-tier split is intentional: lme4 (two sin/cos models behind `atan2`) and
bpnreg (two posterior coefficient sets) have no single-β vector, so a uniform
per-coefficient table is structurally impossible for them — they still return
trajectories, GOF, and a backend-appropriate age-effect test.

`R2_circ` and `MAE_angular` are the only metrics directly comparable across
backends (they're computed identically on the angle scale).  `LL`/`AIC`/`BIC`
are within-backend only (different likelihood families).

## Backends

| Backend | Model | Order selection | Notes |
|---|---|---|---|
| **fitcirc_lme** | von Mises GLMM with subject random intercept φ_i ~ vM(0, κ_φ), EM, sandwich SEs | LRT | Default. Pure MATLAB. Supports a `Resample` option (`'cboot'`, `'sub80'`) for cluster-bootstrap inference. |
| **brms** | Bayesian vM-GLMM, `tan_half` link | LOO (`elpd_diff > 2·se_diff`), with LRT reported alongside | Requires R + `brms` + `loo`. Slower (Stan compile/sample). Cannot represent a mean that wraps a full revolution — see "When to switch backends" below. |
| **lme4** | Sin/cos parallel LMEs (frequentist projected-Gaussian) | Combined sin+cos LRT | Requires R + `lme4`. CI band via `bootMer` (optional). Mean *can* wrap. |
| **bpnreg** | Bayesian projected-normal mixed model | WAIC | Requires R + `bpnreg`. Mean *can* wrap. No frequentist p for `AgeEffect` (uses ΔWAIC). |

The single "Age effect" significance statistic is, for every backend, an
omnibus test that **all** age-involving terms (`Age`, `Age^k`, `Age:cat`,
`Age^k:cat`) are simultaneously zero — not just the main polynomial. This
answers *"does age affect this outcome at all, in either condition"* with one
number per model.

### When to switch backends

- **Trajectory stays within ~one arc** (< ~½ revolution): `fitcirc_lme` is the
  natural choice. The seam is handled by centering and the von Mises
  likelihood; sandwich SEs give correct cluster-robust inference.
- **Trajectory wraps a full revolution**: use `lme4` or `bpnreg` — both
  represent the mean via two components (sin/cos or projected-normal) that
  can sweep through 2π. The vM-based estimators (`fitcirc_lme`, `brms`) cannot:
  von Mises mean is a single circular point and `tan_half` is bounded to one
  revolution.
- **You want Bayesian inference + LRT-equivalent comparisons**: `brms` reports
  LOO and chi-square LRT side-by-side per polynomial step.

## Repository layout

```
circ_fit_toolbox/
├─ circ_fit.m, circ_fit_fitcirc.m, circ_fit_config.m
├─ fitcirc_lme.m            (the core EM von-Mises GLMM)
├─ fit_circ_method.m        (legacy/cboot/sub80 resample bagging)
├─ make_circ_result.m, read_circ_result.m, write_circ_contract.m
├─ circ_center.m            (canonical preprocessing: subtract circular mean)
├─ circ_shift_min_var.m     (variance-minimizing seam placement; legacy)
├─ circ_gof.m               (R²_circ + MAE)
├─ plot_circ_fit.m          (triple-line plotter with band)
├─ R/                       (single Rscript entry + per-backend impls)
│   ├─ circ_fit.R           (entry: Rscript circ_fit.R <work_dir> <backend>)
│   ├─ circ_fit_common.R    (shared helpers: unwrap, R²_circ, name remap, IO)
│   ├─ circ_fit_brms_impl.R
│   ├─ circ_fit_lme4_impl.R
│   └─ circ_fit_bpnreg_impl.R
├─ utils/
│   ├─ build_model_formula.m   (Wilkinson formula builder, stable)
│   └─ get_LLR.m               (chi-square LRT helper)
└─ tests/
    ├─ test_circ_fit_schema.m  (backend parity + wiring smoke test)
    ├─ sim_circ_compare.m      (overlay all four backends on a wrapping sim)
    └─ test_fitcirc_lme_*.m    (recovery / autoshift / etc.)
```

`utils/build_model_formula.m` and `utils/get_LLR.m` are tiny, stable helpers
copied here so the toolbox is self-contained.

## Dependencies

- **MATLAB** R2020b or later (uses tables, `categorical`, anonymous functions
  with struct capture).
- **R 4.x** *only if you want the brms / lme4 / bpnreg backends*. With:
  - `brms`, `loo`, `readr`, `jsonlite`  (for brms)
  - `lme4`                              (for lme4)
  - `bpnreg`                            (for bpnreg)
- `Rscript` on the system `PATH`.  Override via `opts.RscriptPath` if needed.

No R is required to use the default `fitcirc_lme` backend or any MATLAB-side
function.

## Installation

### As a Git submodule (intended use)

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

You can also pass options inline as the last argument to `circ_fit`.

## Tests

```matlab
test_circ_fit_schema({'fitcirc_lme','lme4'});                       % fast
test_circ_fit_schema({'fitcirc_lme','brms','lme4','bpnreg'});       % full
sim_circ_compare();                                                 % overlay on a wrapping sim
```

`test_circ_fit_schema` asserts every backend returns the same required-tier
fields, that `AgeEffect.pValue` is populated, and that the figure-side
trajectory wiring (`get_model_fit` → `Trajectory` interp) works for each.

## Conventions

- **Centering.** Inputs are centered by their circular mean (`θ_shift =
  atan2(mean sin y, mean cos y)`) before fitting; `θ_shift` is recorded on the
  result and added back for predictions. This is the recommended fix for the
  von Mises seam-multimodality problem (Stan Discourse / Bürkner et al.).
- **Unwrapping.** Each backend's stored trajectory is unwrapped per electrode
  along the predictor — the plotter then doesn't need break-at-jumps logic.
- **Coefficient names.** Wilkinson grammar (`(Intercept)`, `Age`, `Age^2`,
  `Age^2:electrode`), uniform across backends; the R workers remap their
  native names accordingly.

## Citation

If you use this toolbox in a paper, please cite the host paper this code was
extracted from (TBD).

## License

TBD.
