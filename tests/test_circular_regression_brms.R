# test_circular_regression_brms.R
#
# Full mixed-effects von-Mises fit on the synthetic data, using brms.
# This is the "publish-quality" version of the comparison: it fits the
# same model the MATLAB pipeline targets (polynomial age + electrode +
# sex + electrode:age interactions + (1|Subj_ID)) but with a proper
# circular likelihood AND a proper random effect, where the MATLAB
# pipeline can only get one of those at a time.
#
# Inference for the joint block tests is via leave-one-out cross-
# validation comparison against nested models:
#   - "any age effect"          : full vs (no Age polynomial, no intx)
#   - "electrode-by-age intx"   : full vs (no electrode:Age intx only)
#
# Outputs
#   - Coefficient summary (posterior mean, SD, 95% CrI, P(same sign))
#   - LOO comparison tables for each block test
#   - Posterior predictive trajectory plot (same panel layout as the
#     MATLAB and circular::lm.circular plots)
#
# Runtime: ~5-10 min for first fit (Stan compile + sample), then ~2 min
# each for the two nested refits.  Cached to .rds via brms file= so
# re-running this script after a cache hit is fast.

suppressPackageStartupMessages({
  library(brms)
  library(tidybayes)
  library(posterior)
  library(loo)
  library(ggplot2)
  library(readr)
})

options(mc.cores = parallel::detectCores())

OUT <- "R_outputs_brms"
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

# ---------------------------------------------------------------------
# 1. Load synthetic data (same CSV as test_circular_regression.R)
# ---------------------------------------------------------------------
d <- read_csv("test_synthetic_data.csv", show_col_types = FALSE)
d$Subj_ID   <- factor(d$Subj_ID)
d$electrode <- as.numeric(d$electrode)
d$sex       <- as.numeric(d$sex)

# Center & scale Age (same z-scoring as test_circular_regression.R) so
# the polynomial design isn't pathologically collinear.
age_mean <- mean(d$Age)
age_sd   <- sd(d$Age)
d$Agez   <- (d$Age - age_mean) / age_sd

d$phase_pref <- ((d$phase_pref + pi) %% (2*pi)) - pi

cat(sprintf("Loaded %d obs from %d subjects\n", nrow(d), nlevels(d$Subj_ID)))

# ---------------------------------------------------------------------
# 2. Priors
#    - Intercept on the tan_half link: weakly informative around 0
#    - Slopes (b): normal(0, 1) on standardized predictors
#    - SD of subject random intercept: half-Student-t (brms default)
#    - Concentration kappa: gamma(2, 0.5), mass mostly in [0.5, 10]
# ---------------------------------------------------------------------
priors <- c(
  prior(normal(0, 1),         class = "Intercept"),
  prior(normal(0, 1),         class = "b"),
  prior(student_t(3, 0, 2.5), class = "sd"),
  prior(gamma(2, 0.5),        class = "kappa")
)

# ---------------------------------------------------------------------
# 3. Three fits: full, no-age (for Age block test), no-intx (for
#    electrode:Age interaction test).  All use file= caching so
#    re-running with the same data hits the cache.
# ---------------------------------------------------------------------

# Use poly(Agez, 3, raw = TRUE) so the polynomial expansion is explicit
# in the design matrix and easy to remove via formula edits below.
fit_full <- brm(
  phase_pref ~ poly(Agez, 3, raw = TRUE) * electrode + sex + (1 | Subj_ID),
  data    = d,
  family  = von_mises(link = "tan_half", link_kappa = "log"),
  prior   = priors,
  chains  = 4, iter = 3000, warmup = 1000,
  control = list(adapt_delta = 0.95, max_treedepth = 12),
  seed    = 1,
  file    = file.path(OUT, "fit_full.rds"),
  file_refit = "on_change",
  refresh = 200
)

fit_no_age <- brm(
  phase_pref ~ electrode + sex + (1 | Subj_ID),
  data    = d,
  family  = von_mises(link = "tan_half", link_kappa = "log"),
  prior   = priors,
  chains  = 4, iter = 3000, warmup = 1000,
  control = list(adapt_delta = 0.95),
  seed    = 1,
  file    = file.path(OUT, "fit_no_age.rds"),
  file_refit = "on_change",
  refresh = 200
)

fit_no_intx <- brm(
  phase_pref ~ poly(Agez, 3, raw = TRUE) + electrode + sex + (1 | Subj_ID),
  data    = d,
  family  = von_mises(link = "tan_half", link_kappa = "log"),
  prior   = priors,
  chains  = 4, iter = 3000, warmup = 1000,
  control = list(adapt_delta = 0.95),
  seed    = 1,
  file    = file.path(OUT, "fit_no_intx.rds"),
  file_refit = "on_change",
  refresh = 200
)

# ---------------------------------------------------------------------
# 4. LOO joint-block comparisons
# ---------------------------------------------------------------------
loo_full    <- loo(fit_full)
loo_no_age  <- loo(fit_no_age)
loo_no_intx <- loo(fit_no_intx)

cat("\n========================================================\n")
cat("  brms (von-Mises mixed effects) -- LOO joint block tests\n")
cat("========================================================\n\n")

cat("--- Age block (full vs no-age) ---\n")
print(loo_compare(loo_full, loo_no_age))
cat("\n--- electrode:Age interaction block (full vs no-intx) ---\n")
print(loo_compare(loo_full, loo_no_intx))

# Convert elpd_diff into something p-value-shaped via the asymptotic
# normal approximation (z = elpd_diff / se_diff).  Not a frequentist
# p-value -- it's a calibrated effect-size summary -- but it is the
# closest single number to plug into a manuscript paragraph.
elpd_z <- function(cmp) {
  d <- cmp[2, "elpd_diff"]
  s <- cmp[2, "se_diff"]
  list(d = d, s = s, z = d / s, p_two_sided = 2*(1 - pnorm(abs(d/s))))
}
age_z  <- elpd_z(loo_compare(loo_full, loo_no_age))
intx_z <- elpd_z(loo_compare(loo_full, loo_no_intx))
cat(sprintf("\nAge block:           elpd_diff = %.2f (SE %.2f), z = %.2f, p ~ %.4g\n",
            age_z$d, age_z$s, age_z$z, age_z$p_two_sided))
cat(sprintf("electrode:Age block: elpd_diff = %.2f (SE %.2f), z = %.2f, p ~ %.4g\n",
            intx_z$d, intx_z$s, intx_z$z, intx_z$p_two_sided))

# ---------------------------------------------------------------------
# 5. Posterior coefficient summary
# ---------------------------------------------------------------------
draws <- as_draws_df(fit_full)
beta_cols <- names(draws)[startsWith(names(draws), "b_")]

summ <- do.call(rbind, lapply(beta_cols, function(nm) {
  x <- draws[[nm]]
  data.frame(
    term  = sub("^b_", "", nm),
    mean  = mean(x),
    sd    = sd(x),
    q2.5  = quantile(x, 0.025),
    q97.5 = quantile(x, 0.975),
    P_same_sign = if (mean(x) >= 0) mean(x > 0) else mean(x < 0),
    row.names = NULL
  )
}))
cat("\n--- Posterior summary (full model) ---\n")
print(summ, digits = 3, row.names = FALSE)

# Also report kappa and the random-intercept SD posteriors.
kappa_post <- draws$kappa
sd_post    <- draws$sd_Subj_ID__Intercept
cat(sprintf("\nkappa     posterior mean = %.2f (95%% CrI %.2f - %.2f)  [truth = 4]\n",
            mean(kappa_post), quantile(kappa_post, 0.025), quantile(kappa_post, 0.975)))
cat(sprintf("sd(Subj)  posterior mean = %.2f (95%% CrI %.2f - %.2f)  [truth = 0.4]\n",
            mean(sd_post),    quantile(sd_post,    0.025), quantile(sd_post,    0.975)))

write.csv(summ, file.path(OUT, "posterior_summary.csv"), row.names = FALSE)

# ---------------------------------------------------------------------
# 6. Posterior predictive trajectory plot
# ---------------------------------------------------------------------
age_grid  <- seq(7, 80, length.out = 100)
agez_grid <- (age_grid - age_mean) / age_sd

newdat <- expand.grid(
  Agez      = agez_grid,
  electrode = c(1, 0),
  sex       = 1
)
newdat$Age <- newdat$Agez * age_sd + age_mean
newdat$elec_label <- factor(
  ifelse(newdat$electrode == 1, "central", "frontal"),
  levels = c("central", "frontal")
)

# re_formula = NA marginalizes over the subject random effect.
preds <- add_epred_draws(newdat, fit_full, re_formula = NA, ndraws = 500)

# Truth in raw Age space (matches the synthetic-data generator).
true_b_raw <- c(0.6, -0.06, 0.0012, -7e-6, 0.30, -0.10, 0.004, -2e-5, 0)

truth_df <- expand.grid(Age = age_grid, electrode = c(1, 0))
truth_df$truth <- with(truth_df, {
  X <- cbind(1, Age, Age^2, Age^3, electrode, 1,
             electrode*Age, electrode*Age^2, electrode*Age^3)
  ((X %*% true_b_raw + pi) %% (2*pi)) - pi
})
truth_df$elec_label <- factor(
  ifelse(truth_df$electrode == 1, "central", "frontal"),
  levels = c("central", "frontal")
)

p <- ggplot(preds, aes(x = Age, y = .epred)) +
  stat_lineribbon(.width = c(0.5, 0.9), alpha = 0.3, fill = "#0072BD") +
  geom_line(data = truth_df, aes(x = Age, y = truth),
            colour = "black", linewidth = 1.2, inherit.aes = FALSE) +
  geom_point(data = d, aes(x = Age, y = phase_pref),
             colour = "gray40", alpha = 0.3, size = 0.8,
             inherit.aes = FALSE) +
  facet_wrap(~ elec_label, ncol = 1) +
  scale_y_continuous(
    breaks = c(-pi, -pi/2, 0, pi/2, pi),
    labels = c("-pi","-pi/2","0","pi/2","pi"),
    limits = c(-pi, pi)
  ) +
  labs(x = "Age (years)", y = "Posterior predicted phase (rad)",
       title = "brms von-Mises mixed-effects fit on synthetic data",
       subtitle = "Black = truth | Blue ribbon = 50/90% credible interval, marginalized over Subj_ID") +
  theme_minimal(base_size = 11)

ggsave(file.path("test_circular_regression_brms_output.png"),
       p, width = 9, height = 6, dpi = 150)

cat("\nPlot written to test_circular_regression_brms_output.png\n")
cat("Posterior summary written to", file.path(OUT, "posterior_summary.csv"), "\n")
