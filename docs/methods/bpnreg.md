# `bpnreg` — Bayesian projected-normal mixed model (R)

> **Cross-references.** Plain-language overview in the
> [README method section](../../README.md#bpnreg--bayesian-projected-normal-mixed-model-r)
> and the at-a-glance comparison in
> [`docs/backends.md`](../backends.md#bpnreg--bayesian-projected-normal-mixed-model).
> The R-side adapter lives at [`R/circ_fit_bpnreg_impl.R`](../../R/circ_fit_bpnreg_impl.R).

This page is the full technical reference for the `bpnreg` backend.
Unlike the von Mises backends, an angle here is modeled as the **angle
of a bivariate Gaussian latent vector projected onto the unit circle**.
That representation can sweep a full revolution (like `lme4`), and
unlike `lme4` it is a proper joint likelihood model. The level of
detail is what a statistician needs to know exactly what is being
computed and how the projection structure differs from the two-stage
sin/cos approach.

---

## 1. Model

For each observation $i, j$ introduce a latent bivariate Gaussian vector

$$
\mathbf{r}_{ij} \;=\; \begin{pmatrix} r_{ij}^{(1)} \\ r_{ij}^{(2)} \end{pmatrix} \;\in\; \mathbb{R}^2,
$$

with the observed angle defined as

$$
y_{ij} \;=\; \text{atan2}\!\bigl(r_{ij}^{(2)},\ r_{ij}^{(1)}\bigr) \;\in\; (-\pi, \pi].
$$

The latent vector has fixed effects on each component, plus a per-
subject random intercept on each component:

$$
\mathbf{r}_{ij} \;=\; \begin{pmatrix} X_{ij}\beta^{(1)} \\ X_{ij}\beta^{(2)} \end{pmatrix}
\;+\; \begin{pmatrix} \phi_i^{(1)} \\ \phi_i^{(2)} \end{pmatrix}
\;+\; \boldsymbol\varepsilon_{ij},
$$

with

$$
\begin{pmatrix} \phi_i^{(1)} \\ \phi_i^{(2)} \end{pmatrix} \;\sim\; \mathcal{N}_2(\mathbf{0},\ \Sigma_\phi),
\qquad
\boldsymbol\varepsilon_{ij} \;\sim\; \mathcal{N}_2(\mathbf{0},\ \Sigma_\varepsilon),
$$

independent over $i$ and over $(i, j)$ respectively. The two covariance
matrices

$$
\Sigma_\phi \;=\; \begin{pmatrix} \sigma_{\phi,11}^2 & \sigma_{\phi,12} \\ \sigma_{\phi,12} & \sigma_{\phi,22}^2 \end{pmatrix},
\qquad
\Sigma_\varepsilon \;=\; \begin{pmatrix} \sigma_{\varepsilon,11}^2 & \sigma_{\varepsilon,12} \\ \sigma_{\varepsilon,12} & \sigma_{\varepsilon,22}^2 \end{pmatrix}
$$

are estimated freely. **Note the off-diagonal terms.** This is the key
structural difference from [`lme4`](lme4.md): the two latent components
have an estimated covariance, so the projected angle's noise structure
is a proper 2-D model rather than two decoupled 1-D models. The two
components of the random intercept are likewise allowed to covary
through $\sigma_{\phi,12}$.

**Identifiability.** The projection $y = \text{atan2}(r^{(2)}, r^{(1)})$ is
invariant under positive scaling of $\mathbf{r}$, so the model is
identified up to a scale on $(\beta^{(1)}, \beta^{(2)}, \Sigma_\phi^{1/2}, \Sigma_\varepsilon^{1/2})$.
`bpnreg` resolves this by fixing $\sigma_{\varepsilon,11}^2 = 1$.

**Density on the angle.** The marginal density of $y_{ij}$ after
integrating out the latent magnitude $R_{ij} = \|\mathbf{r}_{ij}\|$ is
the projected-normal density (Wang & Gelfand 2013):

$$
p(y \mid \mu, \Sigma) \;=\; \frac{\phi_2(\mathbf{0} \mid \mu, \Sigma)}{}
\Bigl[ 1 + \frac{D(\mu, y)\,\Phi\!\bigl(D(\mu, y)\bigr)}{\phi_1\!\bigl(D(\mu, y)\bigr)} \Bigr],
$$

with the projection function

$$
D(\mu, y) \;=\; \frac{\mathbf{u}_y^\top \Sigma^{-1} \mu}{\sqrt{\mathbf{u}_y^\top \Sigma^{-1} \mathbf{u}_y}},
\qquad
\mathbf{u}_y \;=\; \begin{pmatrix} \cos y \\ \sin y \end{pmatrix}.
$$

`bpnreg` evaluates this density at every MCMC iteration; the user need
not see it.

---

## 2. Priors

`bpnreg` uses weakly-informative defaults; the toolbox does not override
them.

| Parameter | Prior |
|---|---|
| $\beta^{(1)}_k, \beta^{(2)}_k$ | Wide Gaussian, $\mathcal{N}(0, 10^2)$ |
| $\Sigma_\phi$ | Wishart with low DF |
| $\sigma_{\varepsilon,22}^2$ | Wide half-Cauchy |
| $\sigma_{\varepsilon,12}$ | Uniform on the support induced by positive-definiteness of $\Sigma_\varepsilon$ |

See `?bpnreg::bpnme` for the exact specification. The toolbox's adapter
does not currently accept user-supplied priors; this is a clean follow-on
if priors elicitation becomes important for a specific application.

---

## 3. Inference: Gibbs sampler

`bpnreg::bpnme` uses a **data-augmented Gibbs sampler** that imputes the
latent magnitudes $R_{ij}$ at each iteration. Given $R_{ij}$, the latent
vector $\mathbf{r}_{ij} = R_{ij} \mathbf{u}_{y_{ij}}$ is fully observed,
so the conditional updates for $\beta^{(1,2)}, \phi_i^{(1,2)}, \Sigma_\phi,
\Sigma_\varepsilon$ are standard conjugate-Gaussian Gibbs updates. The
$R_{ij}$ conditional given everything else is a one-dimensional truncated
distribution that `bpnreg` samples from via inverse-CDF.

**Sampler defaults** (set in `circ_fit_bpnreg_impl.R`):

| Option | Default | Override via |
|---|---|---|
| Iterations | 2000 (1000 burn-in + 1000 sampling) | `opts.Iter`, `opts.Warmup` |
| Thinning | 1 | not currently exposed |
| Seed | 1 | `opts.Seed` |

Convergence is checked by Gelman-Rubin $\hat R$ on each sampled
parameter (computed from the single chain by splitting it into
independent halves, as `bpnreg` does not natively run multiple chains).
The toolbox's adapter emits a warning when any $\hat R > 1.05$.

---

## 4. Order selection: WAIC

Polynomial order selection uses the **Watanabe-Akaike Information
Criterion** (WAIC; Watanabe 2010):

$$
\text{WAIC} \;=\; -2\bigl(\text{lppd} - p_{\text{WAIC}}\bigr),
$$

with the log pointwise predictive density

$$
\text{lppd} \;=\; \sum_{i, j} \log \frac{1}{S} \sum_{s=1}^{S} p\!\bigl(y_{ij} \mid \theta^{(s)}\bigr)
$$

and the effective number of parameters

$$
p_{\text{WAIC}} \;=\; \sum_{i, j} \text{Var}_{s = 1, \ldots, S}\!\bigl[\log p\!\bigl(y_{ij} \mid \theta^{(s)}\bigr)\bigr].
$$

The selection rule is the standard "lower WAIC wins" with a one-SE
buffer:

$$
k_\text{selected} \;=\; \arg\min_k \text{WAIC}_k, \quad \text{but stop stepping up when}\quad \text{WAIC}_k - \text{WAIC}_{k_\text{selected}} > \text{SE}\bigl(\text{WAIC}_k - \text{WAIC}_{k_\text{selected}}\bigr).
$$

The R-side adapter sweeps $k = 0, 1, \ldots, \texttt{MaxOrder}$ and
records each WAIC.

The full per-order audit is in `OrderTable`:

```
order  n_par  LogLikelihood  R2_circ  WAIC  pWAIC  WAIC_diff  selected
```

---

## 5. Inference quantities

### 5.1 Coefficient table

Each fixed-effect coefficient appears twice: once as the latent-$r^{(1)}$
coefficient $\beta^{(1)}_k$ and once as the latent-$r^{(2)}$ coefficient
$\beta^{(2)}_k$. The MATLAB-side coefficient table reports both, with
posterior median, posterior SD, and the 2-sided posterior probability
of being on the wrong side of zero:

$$
p_k^{(c)} \;=\; 2 \min\bigl(\Pr(\beta_k^{(c)} > 0 \mid y),\ \Pr(\beta_k^{(c)} < 0 \mid y)\bigr), \quad c \in \{1, 2\}.
$$

These are component-level summaries, not joint angular statements; see
Section 5.3 for the joint omnibus test.

### 5.2 Trajectory

For each evaluation point, the angular trajectory is

$$
\hat\mu^\star \;=\; \text{median}_{s = 1, \ldots, S}\,\text{atan2}\!\bigl(X^\star\beta^{(2,s)},\ X^\star\beta^{(1,s)}\bigr).
$$

The 95% credible band is the 2.5% / 97.5% quantiles across $s$, with
the same wrap-handling logic as the `lme4` bootstrap (sliding $\pi$-wide
window before quantiling).

### 5.3 Omnibus age-effect test

There is **no classical Wald or LRT** for the age effect in `bpnreg` —
the projection nonlinearity makes the linear hypothesis $\beta_\text{age}^{(1)}
= \beta_\text{age}^{(2)} = 0$ not the same scientific question as "is
there an age effect on the angle." The adapter reports the
**WAIC-difference probability**:

$$
\text{AgeEffect.pValue} \;=\; \Pr\!\bigl(\text{WAIC}_{k_\text{selected}} < \text{WAIC}_0 \mid y\bigr),
$$

approximated by the proportion of MCMC iterations at which the larger-
model WAIC is lower than the no-age-model WAIC. The reported
`AgeEffect.Method` field reads `'bpnreg'` so consumers can distinguish
from frequentist p-values.

For the host paper's "is there an age effect?" sentence, the right way
to report is "the WAIC-based probability that the age model is preferred
over the no-age model is X" rather than to recast as a frequentist p.

### 5.4 $R^2_\text{circ}$

Computed on the posterior median angular predictions
$\hat y_{ij}^\text{med} = \text{atan2}(X_{ij}\hat\beta^{(2),\text{med}},
X_{ij}\hat\beta^{(1),\text{med}})$ via the same circular sum-of-squares
formula as [`fitcirc_lme`](fitcirc_lme.md#71-marginal-r2_textcirc-population-level).
Marginal only; conditional analogue not implemented for this backend.

---

## 6. Assumptions and when they bite

| Assumption | When it matters | Consequence |
|---|---|---|
| Latent magnitude $R$ can be anything in $[0, \infty)$ | Always — the latent representation is unconstrained in magnitude. When the latent vector passes near the origin, the angle becomes very noisy. | The projected-normal density correctly down-weights regions near the origin; this is a feature, not a bug. |
| Latent noise is Gaussian (not heavy-tailed) | When the data contain genuine angular outliers, the Gaussian model puts too little mass on them and the fit is pulled toward the bulk. | Documented; switch to a Student-$t$ projected model (not currently supported) if outliers dominate. |
| `bpnreg`'s single Gibbs chain has mixed and converged | Inference is misleading if the chain is stuck. | Single-chain $\hat R$ on split halves; warning at $\hat R > 1.05$. |
| WAIC is well-behaved | Fails when individual observations have very large posterior log-density variance. | Diagnostic exposed via `pWAIC`; large $p_{\text{WAIC}}$ (relative to the number of free parameters) is a warning sign. |

---

## 7. When to prefer `bpnreg` over `lme4`

Both backends share the "full-revolution" capability, but they differ
in two important ways:

| Aspect | `lme4` | `bpnreg` |
|---|---|---|
| Joint likelihood | No (two-stage decoupled) | Yes (proper joint Gaussian on latent vector) |
| Noise covariance | Diagonal (forced) | Free 2x2 |
| Inference | Frequentist (Wald, bootstrap) | Bayesian (posterior) |
| Joint omnibus test | Bonferroni-union LRT (conservative) | WAIC-difference probability (not a p) |
| Fit speed | Seconds | Minutes |

Prefer `bpnreg` when the sin and cos noises are correlated (which the
true projection mechanism implies), when posterior uncertainty matters,
or when the angular response visibly clusters near a preferred axis
(asymmetric noise). Prefer `lme4` for fast iteration and frequentist
familiarity.

---

## 8. Numerical and computational notes

- **Per-fit cost.** A single `bpnreg::bpnme` run on lifespan-cohort data
  (n ~ 1500, m ~ 800) takes 5 – 15 minutes wall-clock with the default
  2000 iterations. Order selection sweeps multiply this.
- **Single chain.** `bpnreg` does not natively run multi-chain
  diagnostics. The adapter splits the single chain in two for a
  $\hat R$ check; for production runs you may want to call `bpnreg::bpnme`
  multiple times with different seeds and pool externally.
- **Latent magnitudes.** `bpnreg` does not return the imputed $R_{ij}$
  draws by default. If those are needed for posterior-predictive checks,
  pass `return_R = TRUE` to the underlying R call (not currently
  exposed through `opts`).

---

## 9. References

- Mulder, K., & Klugkist, I. (2017). bpnreg: Bayesian Projected Normal Regression Models for Circular Data. R package version 2.0.x.
- Presnell, B., Morrison, S. P., & Littell, R. C. (1998). Projected multivariate linear models for directional data. *Journal of the American Statistical Association* **93**, 1068–1077.
- Wang, F., & Gelfand, A. E. (2013). Directional data analysis under the general projected normal distribution. *Statistical Methodology* **10**, 113–127.
- Watanabe, S. (2010). Asymptotic equivalence of Bayes cross validation and widely applicable information criterion in singular learning theory. *J. Machine Learning Research* **11**, 3571–3594.
- Nuñez-Antonio, G., & Gutiérrez-Peña, E. (2014). A Bayesian model for longitudinal circular data based on the projected normal distribution. *Computational Statistics & Data Analysis* **71**, 506–519.
