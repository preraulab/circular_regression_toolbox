# `lme4` — sin/cos parallel linear mixed-effects models (R)

> **Cross-references.** Plain-language overview in the
> [README method section](../../README.md#lme4--sincos-parallel-linear-mixed-models-r)
> and the at-a-glance comparison in
> [`docs/backends.md`](../backends.md#lme4--sincos-parallel-lmes).
> The R-side adapter lives at [`R/circ_fit_lme4_impl.R`](../../R/circ_fit_lme4_impl.R).

This page is the full technical reference for the `lme4` backend. The
underlying model is **not** a single circular distribution; it is two
ordinary Gaussian linear mixed-effects models, one on $\sin y$ and one on
$\cos y$, fit independently by REML and recombined into an angular
trajectory via $\text{atan2}$ at evaluation time. The level of detail is
what a statistician needs to know exactly what is being computed and
where the two-stage decoupling introduces approximation.

The two-stage decoupling is the structural compromise of the backend.
It is what lets the reconstructed angle sweep more than one revolution
(which the von Mises mean cannot do), but it costs an honest joint
likelihood — the sin and cos components of an angle are constrained
($\sin^2 + \cos^2 = 1$) and the two LMEs do not enforce that constraint.
What this means in practice for inference is spelled out in Section 5.

---

## 1. Model

Let $y_{ij} \in (-\pi, \pi]$ be the angular response. Define the two
real-valued projections

$$
s_{ij} \;=\; \sin y_{ij}, \qquad c_{ij} \;=\; \cos y_{ij}.
$$

For each projection, fit an independent linear mixed-effects model in
$\sin$/$\cos$ space:

$$
\begin{aligned}
s_{ij} &\;=\; X_{ij}\beta^{(s)} + \phi_i^{(s)} + \varepsilon_{ij}^{(s)}, &\quad \phi_i^{(s)} &\;\sim\; \mathcal{N}(0,\ \sigma^{2(s)}_\phi), \quad \varepsilon_{ij}^{(s)} \;\sim\; \mathcal{N}(0,\ \sigma^{2(s)}_\varepsilon), \\
c_{ij} &\;=\; X_{ij}\beta^{(c)} + \phi_i^{(c)} + \varepsilon_{ij}^{(c)}, &\quad \phi_i^{(c)} &\;\sim\; \mathcal{N}(0,\ \sigma^{2(c)}_\phi), \quad \varepsilon_{ij}^{(c)} \;\sim\; \mathcal{N}(0,\ \sigma^{2(c)}_\varepsilon).
\end{aligned}
$$

The same design matrix $X$ drives both fits; the coefficient vectors
$\beta^{(s)}, \beta^{(c)} \in \mathbb{R}^p$ are estimated independently.
Per-subject random intercepts $\phi_i^{(s)}, \phi_i^{(c)}$ are drawn
independently across the two components.

The R-side adapter calls `lme4::lmer` directly:

```R
fit_s <- lme4::lmer(sin_y ~ <RHS> + (1 | Subj_ID), data = tbl, REML = TRUE)
fit_c <- lme4::lmer(cos_y ~ <RHS> + (1 | Subj_ID), data = tbl, REML = TRUE)
```

where `<RHS>` is the polynomial-in-Age + categorical block parsed from
the Wilkinson formula.

**What this model does not assert.** The two LMEs treat $s$ and $c$ as
independent. In reality $s$ and $c$ are functionally dependent
($s^2 + c^2 = 1$) and the noise structures are correlated. The
two-stage fit is statistically equivalent to assuming the noise is a
2-D Gaussian with diagonal covariance and free variances $\sigma^{2(s)}_\varepsilon \ne \sigma^{2(c)}_\varepsilon$,
which is a misspecification. The projected-normal alternative
(`bpnreg`) makes the noise structure explicit; see
[`bpnreg.md`](bpnreg.md).

---

## 2. Trajectory reconstruction

At each evaluation point $x^\star \in \mathbb{R}$ (and category settings
$\text{cat}^\star$), the angular trajectory is

$$
\hat\mu^\star \;=\; \text{atan2}\!\bigl(X^\star\hat\beta^{(s)},\ X^\star\hat\beta^{(c)}\bigr) \;\in\; (-\pi, \pi].
$$

**Why this can wrap a full revolution.** As $x^\star$ varies, the pair
$(X^\star\hat\beta^{(s)}, X^\star\hat\beta^{(c)})$ traces a planar curve
in $\mathbb{R}^2$. If that curve completes one or more full loops around
the origin, $\text{atan2}$ correctly tracks the angular trajectory all
the way around. The von Mises mean (`fitcirc_lme`, `brms`) is a single
circular point, so it cannot represent a trajectory that loops; the
sin/cos parallel representation can.

**Magnitude of the predicted vector.** The reconstructed angle ignores
the predicted vector's magnitude:

$$
\hat R^\star \;=\; \sqrt{(X^\star\hat\beta^{(s)})^2 + (X^\star\hat\beta^{(c)})^2}.
$$

If $\hat R^\star \to 0$ at some $x^\star$ (the predicted sin and cos
both pass through zero simultaneously), the reconstructed angle is
arbitrary and the trajectory becomes numerically unstable. The adapter
emits a warning when $\min_{x^\star} \hat R^\star < 0.1$.

---

## 3. Uncertainty: `bootMer` parametric bootstrap

Standard Wald CIs from the two component LMEs do not directly transfer
to the reconstructed angle because of the $\text{atan2}$ nonlinearity.
The adapter uses **`bootMer`**, lme4's parametric bootstrap of joint
predictions:

1. Draw $B$ replicates of the response under each fitted model:

   $$
   \tilde s_{ij}^{(b)} \;\sim\; \mathcal{N}\!\bigl(X_{ij}\hat\beta^{(s)} + \hat\phi_i^{(s)},\ \hat\sigma^{2(s)}_\varepsilon\bigr), \qquad
   \tilde c_{ij}^{(b)} \;\sim\; \mathcal{N}\!\bigl(X_{ij}\hat\beta^{(c)} + \hat\phi_i^{(c)},\ \hat\sigma^{2(c)}_\varepsilon\bigr).
   $$

2. Refit each component LME on $(\tilde s, \tilde c)^{(b)}$ to obtain
   replicate coefficients $\hat\beta^{(s,b)}, \hat\beta^{(c,b)}$.

3. Compute the replicate trajectory at each eval point:

   $$
   \hat\mu^{\star,(b)} \;=\; \text{atan2}\!\bigl(X^\star\hat\beta^{(s,b)},\ X^\star\hat\beta^{(c,b)}\bigr).
   $$

4. The 95% trajectory band is the 2.5% / 97.5% quantiles across $b = 1,
   \ldots, B$. Wrap-around is handled by sorting the replicate angles
   within a sliding $\pi$-wide window before quantiling.

**Default $B$** is 500. Set via `opts.B`. With `opts.Band = false` the
adapter skips the bootstrap and returns `lo = hi = mean` (point estimate
only) — useful for fast layout iteration.

**Why this is the right band.** A naive sandwich CI on each component
followed by an atan2 of the bounds is wrong: the atan2 of the
endpoints is not the endpoint of the atan2. The parametric bootstrap
samples the JOINT $(\hat\beta^{(s)}, \hat\beta^{(c)})$ distribution and
applies the nonlinearity inside the bootstrap, so the resulting band
is honest.

---

## 4. Order selection: combined sin+cos LRT

For each candidate polynomial order $k$ in $0, 1, \ldots, \texttt{MaxOrder}$,
the adapter fits both component LMEs and computes the standard
likelihood-ratio test against the order-$(k-1)$ component:

$$
T_s^{(k)} \;=\; -2\bigl(\ell_s^{(k-1)} - \ell_s^{(k)}\bigr) \;\sim\; \chi^2_{\Delta_k},
\qquad
T_c^{(k)} \;=\; -2\bigl(\ell_c^{(k-1)} - \ell_c^{(k)}\bigr) \;\sim\; \chi^2_{\Delta_k}.
$$

The **combined p-value** is the Bonferroni union

$$
p_\text{combined}^{(k)} \;=\; 2 \min\!\bigl(p_s^{(k)},\ p_c^{(k)}\bigr).
$$

This is conservative: it controls the joint type-I rate at $\alpha$ when
the two component tests are positively correlated (they are, since they
share a design matrix), at the cost of statistical power. A genuine
joint test on the (sin, cos) pair would require modeling their
covariance, which lme4 does not do.

The step-up procedure accepts order $k$ if $p_\text{combined}^{(k)} < 0.05$.
This is the only available frequentist joint test in this backend.

The full per-order audit is in `OrderTable`:

```
order  n_par  LogLikelihood_s  LogLikelihood_c  R2_circ  p_lrt_s  p_lrt_c  p_combined  selected
```

---

## 5. Goodness of fit

### 5.1 $R^2_\text{circ}$

Computed on the reconstructed-angle predictions
$\hat y_{ij} = \text{atan2}(X_{ij}\hat\beta^{(s)}, X_{ij}\hat\beta^{(c)})$
via the same circular sum-of-squares formula as
[`fitcirc_lme`](fitcirc_lme.md#71-marginal-r2_textcirc-population-level).
This is the only component metric directly comparable to the von Mises
backends. **Marginal only**; the conditional analogue would require
adapting the Nakagawa-Schielzeth ICC to a two-component random-intercept
model and is not currently implemented for the lme4 backend.

### 5.2 Per-component $R^2$

The two LMEs each return their own marginal $R^2$ (via `lme4::r.squaredGLMM`,
Nakagawa-Schielzeth):

$$
R^2_{m, s} \;=\; \frac{\text{Var}_\text{fixed}(s)}{\text{Var}_\text{fixed}(s) + \sigma^{2(s)}_\phi + \sigma^{2(s)}_\varepsilon},
\qquad
R^2_{m, c} \;=\; \frac{\text{Var}_\text{fixed}(c)}{\text{Var}_\text{fixed}(c) + \sigma^{2(c)}_\phi + \sigma^{2(c)}_\varepsilon}.
$$

These are reported in the result's `Diagnostics.PerComponentR2` field
for the consumer who wants to see how much fixed-effect signal lives in
each projection. (Sometimes sin carries most of the structure; sometimes
cos; the asymmetry can be diagnostic of whether the response is closer
to $0$ or $\pi/2$ in mean direction.)

### 5.3 MAE

$$
\text{MAE}_\text{angular} \;=\; \frac{1}{n} \sum_{i, j} \bigl|y_{ij} - \hat y_{ij}\bigr|_\text{wrap}.
$$

---

## 6. Assumptions and when they bite

| Assumption | When it matters | Consequence |
|---|---|---|
| Sin and cos noises are independent | Always wrong (the angle is on the unit circle, $s$ and $c$ are negatively correlated when the angle is near $\pi/2$ etc.). | Standard errors and joint tests are approximate. The bias is small when the response is tightly concentrated (so both projections are nearly Gaussian on a narrow arc) and grows when the response spans a large angular range. |
| $\sigma^{2(s)}_\varepsilon = \sigma^{2(c)}_\varepsilon$ (free in lme4) | Equality is not assumed; each LME estimates its own residual variance. Allowing them to differ is a feature when the projection has a preferred axis. | Reasonable. |
| The reconstructed magnitude $\hat R^\star$ stays bounded away from zero | When $\hat R^\star \to 0$, atan2 is numerically unstable and the reconstructed angle is undefined. | Adapter warns when $\min_{x^\star} \hat R^\star < 0.1$. |
| Bonferroni-union for the joint LRT is conservative | Order selection may under-detect a real effect that lives in only one of the two projections at moderate magnitude. | Documented limitation. Cross-check with `fitcirc_lme` (proper joint Wald) when the response is unimodal. |

---

## 7. Numerical and computational notes

- **Per-fit cost.** Two `lme4::lmer` REML fits per order; one is $O(n p)$
  via Cholesky factorization of the random-effect block-diagonal. On
  lifespan-cohort data, one order's fit is sub-second; the bootstrap
  band dominates wall-clock at $B = 500$ (~30 seconds at $B = 500$ on a
  modern laptop).
- **REML vs ML.** Used REML for the variance components and ML for the
  LRT (REML log-likelihoods are not comparable across nested fixed-
  effect models). The adapter refits ML when computing the LRT statistic.
- **`bootMer` parallelism.** The adapter exposes `mc.cores` to lme4's
  parallel bootstrap when available; pass via `opts.Cores`.

---

## 8. References

- Bates, D., Mächler, M., Bolker, B., & Walker, S. (2015). Fitting linear mixed-effects models using lme4. *Journal of Statistical Software* **67**, 1–48.
- Bates, D. M. (2010). *lme4: Mixed-effects modeling with R*. Springer (draft book; see also `lme4`'s vignettes).
- Davison, A. C., & Hinkley, D. V. (1997). *Bootstrap Methods and Their Application*. Cambridge. (Parametric bootstrap.)
- Mardia, K. V., & Jupp, P. E. (2000). *Directional Statistics*. Wiley. Chapter 8 for the sin/cos two-stage representation of angular data.
- Nakagawa, S., & Schielzeth, H. (2013). A general and simple method for obtaining $R^2$ from generalized linear mixed-effects models. *Methods in Ecology and Evolution* **4**, 133–142.
