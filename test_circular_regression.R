# test_circular_regression.R
#
# R-side companion to test_circular_regression.m.  Reads the synthetic
# CSV produced by the MATLAB test script and fits two R-side models for
# direct comparison against the MATLAB outputs:
#
#   (a) circular::lm.circular -- proper Fisher-Lee MLE (von Mises, identity
#       link, fixed-effects only, two-sided Wald p-values).  Tests whether
#       the MATLAB-side circular_regression_fixed.m gives the same
#       coefficient estimates and SEs as a well-tested R implementation
#       of the same model.
#
#   (b) lme4::lmer on sin(phase) and cos(phase) -- mirror of the MATLAB
#       fitlme sin/cos approach.  Same model, different software.  Used
#       to confirm that the MATLAB sin/cos numbers are not an artifact of
#       fitlme idiosyncrasies.
#
# The brms (von Mises mixed-effects) route is described in
# circular_regression_pipeline.R but not run here -- it requires Stan to
# compile and would take ~30 minutes of CPU.  The frequentist outputs
# below are the practical sanity check.

suppressPackageStartupMessages({
  library(circular)
  library(lme4)
  library(readr)
})

cat("====================================================\n")
cat("  R comparison: circular::lm.circular + lme4 sin/cos \n")
cat("====================================================\n\n")

# ---------------------------------------------------------------------
# 1. Load the same synthetic data the MATLAB script just wrote
# ---------------------------------------------------------------------
d <- read_csv("test_synthetic_data.csv", show_col_types = FALSE)
d$Subj_ID   <- factor(d$Subj_ID)
d$electrode <- as.numeric(d$electrode)   # keep numeric to match MATLAB design
d$sex       <- as.numeric(d$sex)
# Center & scale Age before building polynomial columns.  Raw polynomials
# of Age over [7, 80] are >0.97 correlated, which makes lm.circular's
# IRLS choke on a near-singular design.  We model on a standardized Age
# (z-scored), recover identical model fit, and convert back for plotting.
age_mean <- mean(d$Age)
age_sd   <- sd(d$Age)
d$Agez   <- (d$Age - age_mean) / age_sd
d$Agez2  <- d$Agez^2
d$Agez3  <- d$Agez^3

# Wrap phase to (-pi, pi] explicitly.
d$phase_pref <- ((d$phase_pref + pi) %% (2*pi)) - pi
d$phase_circ <- circular(d$phase_pref, units = "radians",
                         modulo = "2pi", template = "none")

cat(sprintf("Loaded %d observations from %d subjects\n\n",
            nrow(d), nlevels(d$Subj_ID)))

# ---------------------------------------------------------------------
# 2. Frequentist von-Mises MLE via circular::lm.circular
#    type = "c-l" is identity-link Fisher-Lee with proper IRLS
# ---------------------------------------------------------------------
X <- model.matrix(~ Agez + Agez2 + Agez3 + electrode + sex +
                    Agez:electrode + Agez2:electrode + Agez3:electrode,
                  data = d)
X <- X[, -1, drop = FALSE]   # drop intercept; lm.circular adds its own

fit_circ <- lm.circular(
  y       = d$phase_circ,
  x       = X,
  init    = rep(0, ncol(X)),
  type    = "c-l",
  verbose = FALSE
)

# Build a coefficient table comparable to mdl.Coefficients in MATLAB.
coef_tbl <- data.frame(
  term     = c("(Intercept)", colnames(X)),
  estimate = c(fit_circ$mu,    as.numeric(fit_circ$coefficients)),
  se       = c(NA,             as.numeric(fit_circ$se.coef)),
  pvalue   = c(NA,             as.numeric(fit_circ$p.values))
)

cat("--- circular::lm.circular (Fisher-Lee MLE, no random effect) ---\n")
print(coef_tbl, digits = 4, row.names = FALSE)
cat(sprintf("\nKappa estimate: %.3f\n\n", fit_circ$kappa))

# Joint Wald tests on coefficient blocks.  lm.circular does not ship a
# coefTest, so we build the chi^2 statistic from the coefficient vector
# and the asymptotic covariance returned by the fit.
b   <- as.numeric(fit_circ$coefficients)
V   <- fit_circ$cov.coef
nms <- colnames(X)

wald_block <- function(rows) {
  R <- matrix(0, nrow = length(rows), ncol = length(b))
  for (i in seq_along(rows)) R[i, rows[i]] <- 1
  Rb  <- R %*% b
  RVR <- R %*% V %*% t(R)
  W   <- as.numeric(t(Rb) %*% solve(RVR, Rb))
  list(W = W, df = nrow(R), p = pchisq(W, df = nrow(R), lower.tail = FALSE))
}

age_idx  <- which(nms %in% c("Agez", "Agez2", "Agez3"))
intx_idx <- which(nms %in% c("Agez:electrode", "Agez2:electrode", "Agez3:electrode"))

age_test  <- wald_block(age_idx)
intx_test <- wald_block(intx_idx)

cat("Joint block tests (Wald chi-squared, no clustering):\n")
cat(sprintf("  Age block:           chi2(%d) = %.3f, p = %.4g\n",
            age_test$df, age_test$W, age_test$p))
cat(sprintf("  electrode:Age block: chi2(%d) = %.3f, p = %.4g\n\n",
            intx_test$df, intx_test$W, intx_test$p))

# ---------------------------------------------------------------------
# 3. lme4 sin/cos LME (random intercept per Subj_ID), mirroring MATLAB
# ---------------------------------------------------------------------
d$sin_phase <- sin(d$phase_pref)
d$cos_phase <- cos(d$phase_pref)

fit_sin <- lmer(sin_phase ~ Agez + Agez2 + Agez3 + electrode + sex +
                  Agez:electrode + Agez2:electrode + Agez3:electrode +
                  (1 | Subj_ID),
                data = d, REML = TRUE)

fit_cos <- lmer(cos_phase ~ Agez + Agez2 + Agez3 + electrode + sex +
                  Agez:electrode + Agez2:electrode + Agez3:electrode +
                  (1 | Subj_ID),
                data = d, REML = TRUE)

# Joint Wald tests in each component.  lme4 doesn't expose a single-step
# F-test like MATLAB's coefTest, so we build it by hand from the fixed-
# effects covariance.
wald_lmer_block <- function(fit, term_names) {
  vc  <- as.matrix(vcov(fit))
  fe  <- fixef(fit)
  idx <- match(term_names, names(fe))
  Rb  <- fe[idx]
  V   <- vc[idx, idx, drop = FALSE]
  W   <- as.numeric(t(Rb) %*% solve(V, Rb))
  list(W = W, df = length(idx),
       p = pchisq(W, df = length(idx), lower.tail = FALSE))
}

age_terms  <- c("Agez", "Agez2", "Agez3")
intx_terms <- c("Agez:electrode", "Agez2:electrode", "Agez3:electrode")

age_sin  <- wald_lmer_block(fit_sin,  age_terms)
age_cos  <- wald_lmer_block(fit_cos,  age_terms)
intx_sin <- wald_lmer_block(fit_sin,  intx_terms)
intx_cos <- wald_lmer_block(fit_cos,  intx_terms)

# Bonferroni across the two components for the joint phase claim.
combine <- function(p1, p2) min(1, 2 * min(p1, p2))

cat("--- lme4 sin/cos LME (random intercept per Subj_ID) ---\n")
cat(sprintf("Age block:           sin chi2=%.2f p=%.4g | cos chi2=%.2f p=%.4g | Bonf p=%.4g\n",
            age_sin$W,  age_sin$p,  age_cos$W,  age_cos$p,
            combine(age_sin$p,  age_cos$p)))
cat(sprintf("electrode:Age block: sin chi2=%.2f p=%.4g | cos chi2=%.2f p=%.4g | Bonf p=%.4g\n\n",
            intx_sin$W, intx_sin$p, intx_cos$W, intx_cos$p,
            combine(intx_sin$p, intx_cos$p)))

# ---------------------------------------------------------------------
# 4. Plot the trajectories
# ---------------------------------------------------------------------
# Same overlay style as the MATLAB plot: per-electrode subplots, truth
# vs each method.  Plotted with base graphics for portability.

age_grid  <- seq(7, 80, length.out = 200)
agez_grid <- (age_grid - age_mean) / age_sd

# Design rows in the SAME order as the fit's coefficient vector
# (intercept, Agez, Agez2, Agez3, electrode, sex, Agez:elec, Agez2:elec,
# Agez3:elec).  This matches both circular::lm.circular's coefficient
# layout (intercept handled separately in fit_circ$mu) and lme4's
# fixef() ordering for the sin/cos LMEs.
mk_X <- function(agez, elec, sex_eval = 1) {
  cbind(agez, agez^2, agez^3, elec, sex_eval,
        elec*agez, elec*agez^2, elec*agez^3)
}

# True parameters in the RAW Age coefficient space.  Kept here for
# building the truth trajectory only; not used to compare to estimates,
# since the fits live in standardized-Age space.
true_b_raw <- c(intercept   = 0.6,
                Age         = -0.06,
                Age2        = 0.0012,
                Age3        = -7e-6,
                electrode   = 0.30,
                sex         = -0.10,
                Age_x_elec  = 0.004,
                Age2_x_elec = -2e-5,
                Age3_x_elec = 0)
mk_X_raw <- function(age, elec, sex_eval = 1) {
  cbind(age, age^2, age^3, elec, sex_eval,
        elec*age, elec*age^2, elec*age^3)
}

pdf("test_circular_regression_R_output.pdf", width = 10, height = 6)
op <- par(mfrow = c(2, 1), mar = c(4, 4, 2.5, 1))

for (elec in c(1, 0)) {
  panel_title <- sprintf("Electrode = %s (sex = F)",
                         if (elec == 1) "central" else "frontal")

  # Truth: built in raw Age space (where the synthetic data was generated)
  X_truth <- cbind(1, mk_X_raw(age_grid, elec, 1))
  truth_traj <- ((X_truth %*% true_b_raw + pi) %% (2*pi)) - pi

  # circular::lm.circular fit lives in standardized Age space.
  # The package displays both the intercept and the slope coefficients
  # wrapped to [0, 2*pi) for some reason.  Unwrap to (-pi, pi] before
  # using them for prediction, otherwise out-of-range slopes produce
  # nonsensical curves.
  unwrap <- function(z) ifelse(z > pi, z - 2*pi, z)
  X_z <- cbind(1, mk_X(agez_grid, elec, 1))
  circ_b <- unwrap(c(as.numeric(fit_circ$mu),
                     as.numeric(fit_circ$coefficients)))
  circ_traj <- ((X_z %*% circ_b + pi) %% (2*pi)) - pi

  # sin/cos LME predictions (also in standardized Age space)
  ndat <- data.frame(
    Agez = agez_grid, Agez2 = agez_grid^2, Agez3 = agez_grid^3,
    electrode = elec, sex = 1,
    Subj_ID = factor(NA, levels = levels(d$Subj_ID))
  )
  sin_pred <- predict(fit_sin, ndat, re.form = NA, allow.new.levels = TRUE)
  cos_pred <- predict(fit_cos, ndat, re.form = NA, allow.new.levels = TRUE)
  sincos_traj <- atan2(sin_pred, cos_pred)

  # Mask wrap discontinuities for clean line plotting.
  mask_wrap <- function(y) {
    bad <- c(FALSE, abs(diff(y)) > pi)
    y[bad] <- NA
    y
  }

  # Data points
  is_e <- d$electrode == elec
  plot(d$Age[is_e], d$phase_pref[is_e],
       pch = 16, cex = 0.5, col = adjustcolor("gray40", 0.4),
       ylim = c(-pi, pi), xlim = c(0, 80),
       xlab = "Age (years)", ylab = "Preferred phase (rad)",
       main = panel_title, axes = FALSE)
  axis(1)
  axis(2, at = c(-pi, -pi/2, 0, pi/2, pi),
       labels = c(expression(-pi), expression(-pi/2), "0",
                  expression(pi/2), expression(pi)))
  box()

  lines(age_grid, mask_wrap(truth_traj),  col = "black",   lwd = 2.5)
  lines(age_grid, mask_wrap(circ_traj),   col = "#0072BD", lwd = 2)
  lines(age_grid, mask_wrap(sincos_traj), col = "#77AC30", lwd = 2, lty = 3)

  if (elec == 1) {
    legend("topleft", bty = "n", lwd = c(NA, 2.5, 2, 2),
           lty = c(NA, 1, 1, 3),
           pch = c(16, NA, NA, NA),
           col = c("gray40", "black", "#0072BD", "#77AC30"),
           legend = c("Data", "Truth",
                      "circular::lm.circular",
                      "lme4 sin/cos"))
  }
}
par(op); dev.off()

cat("Plot written to test_circular_regression_R_output.pdf\n")
