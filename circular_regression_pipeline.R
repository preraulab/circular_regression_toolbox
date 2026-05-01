# =====================================================================
# circular_regression_pipeline.R
# =====================================================================
#
# Bayesian replacement for the MATLAB circular-regression pipeline used
# to produce Supplemental Tables 5-6 (and the pref_phase / Theta columns
# of Tables 1-4) in the lifespan manuscript.
#
# The MATLAB implementation (stats/circular_regression.m, called via
# stats/get_single_order_model.m) fits an identity-link von-Mises
# regression by hand-rolled IRLS with several issues documented in
# AUDIT.md (one-sided p-values, no random effect for repeated subjects,
# unit-weight Hessian, etc.).  This script does the same job using
# brms with family = von_mises(), which:
#
#   - fits a proper mixed-effects model with (1 | Subj_ID),
#   - handles wrapping via the tan_half link,
#   - returns full posteriors (no Wald approximation issues),
#   - supports joint block tests via leave-one-out cross-validation
#     comparison against nested models.
#
# Two-stage flow:
#
#   1.  Export per-class tables from MATLAB to CSV (snippet at bottom).
#   2.  Run this script on each (class, circular-feature) pair.
#       It returns:
#         - a brmsfit object for the full polynomial model,
#         - LOO comparisons against intercept-only and against models
#           with the polynomial-age block / electrode-by-age block
#           removed (i.e. the joint "any age effect" / "any electrode-
#           by-age interaction" tests),
#         - a posterior summary table mirroring mdl.Coefficients,
#         - posterior predictive trajectories for plotting.
#
# This script does NOT depend on having the original MATLAB output --
# it is self-contained given the CSV export.
#
# REQUIREMENTS
#   R >= 4.1
#   install.packages(c("brms", "tidyverse", "loo", "tidybayes",
#                      "posterior", "patchwork"))
#   First brms fit will trigger a Stan compile (~30-60s) per unique
#   model formula; subsequent fits with the same formula reuse the
#   cached binary.
# =====================================================================

suppressPackageStartupMessages({
  library(brms)
  library(dplyr)
  library(tidyr)
  library(readr)
  library(stringr)
  library(loo)
  library(tidybayes)
  library(posterior)
  library(ggplot2)
  library(patchwork)
})

# Use all available cores by default; brms parallelizes Stan chains.
options(mc.cores = parallel::detectCores())

# Where to write outputs (fits, tables, plots).  Created if absent.
OUT_DIR <- file.path("R_outputs")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)


# =====================================================================
# 1. DATA LOADING AND PREP
# =====================================================================
#
# Expected CSV columns:
#   Subj_ID    integer/character subject identifier
#   Age        numeric age in years
#   electrode  numeric or factor: central vs frontal coding (any 0/1
#              or "central"/"frontal" works; we factorize below)
#   sex        numeric or factor (any 0/1 or "F"/"M")
#   cluster    integer 1..4 for SOPH (sigma_high / sigma_low /
#              theta-alpha / delta) or 1..2 for SOPhH (upper / lower)
#   pref_phase circular feature in radians (SOPH analyses)
#   Theta      circular feature in radians (SOPhH analyses)
#   Frequency, Amplitude, SOpower, ...   non-circular features
#                                        (passed through; not modeled
#                                        here -- those use fitlme in
#                                        MATLAB, which is fine)
#
# CSV is produced from MATLAB via the snippet at the bottom of this file.

load_table <- function(csv_path) {
  d <- read_csv(csv_path, show_col_types = FALSE)

  # Coerce factors with explicit levels so brms doesn't pick alphabetical
  # ordering by accident.  Reference levels here become the model
  # intercept's interpretation: "central" and "F" are the references.
  if (!is.factor(d$electrode)) {
    d$electrode <- factor(
      ifelse(d$electrode %in% c(1, "1", "central", "C"), "central", "frontal"),
      levels = c("central", "frontal")
    )
  }
  if (!is.factor(d$sex)) {
    d$sex <- factor(
      ifelse(d$sex %in% c("F", "female", "Female", 1, "1"), "F", "M"),
      levels = c("F", "M")
    )
  }
  d$Subj_ID <- factor(d$Subj_ID)

  # Wrap circular columns to (-pi, pi] explicitly.  MATLAB exports may
  # already be wrapped, but we don't trust it and re-wrap here.
  wrap_to_pi <- function(x) ((x + pi) %% (2 * pi)) - pi
  if ("pref_phase" %in% names(d)) d$pref_phase <- wrap_to_pi(d$pref_phase)
  if ("Theta"      %in% names(d)) d$Theta      <- wrap_to_pi(d$Theta)

  d
}


# =====================================================================
# 2. PRIORS
# =====================================================================
#
# Weakly informative.  Notes on each:
#   - Intercept on the tan_half scale: a normal(0, 1) corresponds to a
#     fairly diffuse prior on the wrapped phase, with mass within roughly
#     +/- pi/2 (the tan_half link is 2*atan(eta/2)).
#   - "b" is the prior on regression coefficients other than the
#     intercept; normal(0, 1) is standard for centered/scaled
#     predictors, and our polynomial expansion is on raw age, so we use
#     normal(0, 0.5) which is still weakly informative for typical
#     phase precessions (<pi over a 70-year span).
#   - Random-intercept SD: half-Student-t with 3 df is the brms default
#     and works well here.
#   - kappa: gamma(2, 0.5) puts most mass between 0.5 and 10, which
#     covers the regime of typical EEG phase coupling concentrations.

base_priors <- function() {
  c(
    prior(normal(0, 1),         class = "Intercept"),
    prior(normal(0, 0.5),       class = "b"),
    prior(student_t(3, 0, 2.5), class = "sd"),
    prior(gamma(2, 0.5),        class = "kappa")
  )
}


# =====================================================================
# 3. SINGLE-MODEL FITTER
# =====================================================================
#
# Fit one (class, feature) combination at a single polynomial order.
# Returns the brmsfit, plus a list of nested fits for the joint block
# tests below.  Caches fits to disk via brms file= so re-running the
# script does not refit unchanged models.

fit_circular_model <- function(data,
                               feature,                 # "pref_phase" or "Theta"
                               cluster_value,           # which cluster to subset
                               order = 3,               # polynomial order
                               include_electrode_x_age = TRUE,
                               iter = 4000, warmup = 1000, chains = 4,
                               cache_tag = "default") {

  d <- data %>%
    filter(cluster == cluster_value) %>%
    filter(!is.na(.data[[feature]]))

  # Build the formula string.  poly(Age, k, raw = TRUE) keeps the
  # coefficients on the same scale as MATLAB's Age, Age^2, Age^3 raw
  # polynomial expansion.  Use raw = FALSE if you want orthogonal
  # polynomials (joint block tests are invariant to that choice; only
  # individual coefficient interpretations change).
  age_term <- sprintf("poly(Age, %d, raw = TRUE)", order)
  if (include_electrode_x_age) {
    rhs <- sprintf("%s * electrode + sex + (1 | Subj_ID)", age_term)
  } else {
    rhs <- sprintf("%s + electrode + sex + (1 | Subj_ID)", age_term)
  }
  fml <- as.formula(sprintf("%s ~ %s", feature, rhs))

  # Cache file: same formula on the same data should hit the cache.
  cache_file <- file.path(
    OUT_DIR,
    sprintf("brmsfit_%s_%s_cluster%s_order%d_intx%s.rds",
            cache_tag, feature, cluster_value, order, include_electrode_x_age)
  )

  fit <- brm(
    formula = fml,
    data    = d,
    family  = von_mises(link = "tan_half", link_kappa = "log"),
    prior   = base_priors(),
    chains  = chains, iter = iter, warmup = warmup,
    control = list(adapt_delta = 0.95, max_treedepth = 12),
    seed    = 1,
    file    = cache_file,
    file_refit = "on_change"
  )

  fit
}


# =====================================================================
# 4. JOINT BLOCK TESTS (Bayesian analog of MATLAB coefTest)
# =====================================================================
#
# In a frequentist Wald framework, "is there an age effect" is one
# coefTest call on the polynomial-age block.  The Bayesian analog is to
# fit the nested model without the block and compare predictive
# performance via leave-one-out cross-validation (LOO-CV).  The output
# elpd_diff (with its SE) tells you how much predictive accuracy is
# lost by removing the block; if elpd_diff is large negative for the
# null model relative to the full model, the block is doing real work.
#
# This is the right object to put in the supplement.  It is calibrated
# (no Wald approximation), it does not depend on a polynomial collinearity
# pathology, and it generalizes naturally to the multilevel structure.

joint_block_loo <- function(full_fit, data, feature, cluster_value,
                            cache_tag = "default") {

  d <- data %>%
    filter(cluster == cluster_value) %>%
    filter(!is.na(.data[[feature]]))

  # --- nested 1: intercept-only (well, electrode + sex still in) ---
  # Tests "is there ANY age effect, main or interaction".
  fml_no_age <- as.formula(sprintf("%s ~ electrode + sex + (1 | Subj_ID)",
                                   feature))
  fit_no_age <- brm(
    formula = fml_no_age,
    data    = d,
    family  = von_mises(link = "tan_half", link_kappa = "log"),
    prior   = base_priors(),
    chains  = 4, iter = 4000, warmup = 1000,
    control = list(adapt_delta = 0.95),
    seed    = 1,
    file    = file.path(OUT_DIR,
                        sprintf("brmsfit_%s_%s_cluster%s_no_age.rds",
                                cache_tag, feature, cluster_value)),
    file_refit = "on_change"
  )

  # --- nested 2: keep age main effects, drop electrode * age block ---
  # Tests "is there an electrode-by-age interaction".  We use poly(Age,3)
  # (or whatever order the full model used; pulled from the full fit).
  full_terms <- attr(terms(full_fit$formula$formula), "term.labels")
  age_order  <- as.integer(stringr::str_match(
    full_terms[grepl("^poly\\(Age", full_terms)][1],
    "poly\\(Age, (\\d+)"
  )[, 2])
  if (is.na(age_order)) age_order <- 3

  fml_no_intx <- as.formula(
    sprintf("%s ~ poly(Age, %d, raw = TRUE) + electrode + sex + (1 | Subj_ID)",
            feature, age_order)
  )
  fit_no_intx <- brm(
    formula = fml_no_intx,
    data    = d,
    family  = von_mises(link = "tan_half", link_kappa = "log"),
    prior   = base_priors(),
    chains  = 4, iter = 4000, warmup = 1000,
    control = list(adapt_delta = 0.95),
    seed    = 1,
    file    = file.path(OUT_DIR,
                        sprintf("brmsfit_%s_%s_cluster%s_no_intx.rds",
                                cache_tag, feature, cluster_value)),
    file_refit = "on_change"
  )

  # Compute LOO for each model (cached on the fit object).
  loo_full    <- loo(full_fit)
  loo_no_age  <- loo(fit_no_age)
  loo_no_intx <- loo(fit_no_intx)

  list(
    age_block_test  = loo_compare(loo_full, loo_no_age),
    intx_block_test = loo_compare(loo_full, loo_no_intx),
    fits = list(no_age = fit_no_age, no_intx = fit_no_intx)
  )
}


# =====================================================================
# 5. POSTERIOR SUMMARY TABLE (mirrors MATLAB mdl.Coefficients)
# =====================================================================
#
# Produces a table comparable to what the MATLAB pipeline writes into
# the supplemental Excel files: coefficient name, posterior mean,
# posterior SD (the Bayesian analog of SE), 95% credible interval, and
# the posterior probability that the coefficient differs from zero in
# the same direction as its mean (a one-sided "P(beta same sign as
# posterior mean)" -- this replaces the frequentist p-value but is NOT
# the same thing; flag this clearly in any reported tables).

posterior_summary_table <- function(fit) {
  draws <- as_draws_df(fit)
  # Population-level (fixed-effect) coefficients -- prefixed b_ in brms.
  beta_cols <- names(draws)[startsWith(names(draws), "b_")]

  out <- lapply(beta_cols, function(nm) {
    x <- draws[[nm]]
    tibble(
      term      = sub("^b_", "", nm),
      mean      = mean(x),
      sd        = sd(x),
      q2.5      = quantile(x, 0.025),
      q97.5     = quantile(x, 0.975),
      # Posterior probability that the sign of beta matches the sign
      # of its posterior mean.  This is the cleanest single-number
      # summary of "how confident are we this effect is in this
      # direction"; do NOT label it as a p-value.
      post_prob_same_sign = if (mean(x) >= 0) mean(x > 0) else mean(x < 0)
    )
  }) %>% bind_rows()
  out
}


# =====================================================================
# 6. POSTERIOR PREDICTIVE TRAJECTORY (the figure object)
# =====================================================================
#
# Produces the population-level (random effect marginalized) predicted
# trajectory of the circular response, with 50% and 90% credible
# intervals, faceted by sex and colored by electrode.  This is the
# Bayesian replacement for plot_model_fit + the +/-2*pi-shifted
# CI ribbons drawn in the MATLAB pipeline.

trajectory_plot <- function(fit, age_grid = seq(7, 80, length.out = 100)) {
  newdat <- expand_grid(
    Age       = age_grid,
    electrode = c("central", "frontal"),
    sex       = c("F", "M")
  ) %>%
    mutate(electrode = factor(electrode, levels = c("central", "frontal")),
           sex       = factor(sex,       levels = c("F", "M")))

  # re_formula = NA marginalizes over the subject random effect, giving
  # the population-level prediction (what we want to plot for the paper).
  preds <- add_epred_draws(newdat, fit, re_formula = NA, ndraws = 500)

  ggplot(preds, aes(x = Age, y = .epred, color = electrode, fill = electrode)) +
    stat_lineribbon(.width = c(0.5, 0.9), alpha = 0.3) +
    facet_wrap(~ sex) +
    scale_y_continuous(
      breaks = c(-pi, -pi/2, 0, pi/2, pi),
      labels = c(expression(-pi), expression(-pi/2), "0",
                 expression(pi/2), expression(pi)),
      limits = c(-pi, pi)
    ) +
    labs(x = "Age (years)", y = "Posterior predicted phase (rad)") +
    theme_minimal(base_size = 11) +
    theme(legend.position = "bottom")
}


# =====================================================================
# 7. DRIVER -- loop over (class, circular feature) combinations
# =====================================================================
#
# Edit the input CSV paths and the loop below to suit.  As written this
# assumes you have exported two CSVs from MATLAB:
#
#   combined_tbl_SOPH.csv    (clusters 1..4, columns include pref_phase)
#   combined_tbl_SOPhH.csv   (clusters 1..2, columns include Theta and Phase)
#
# and that the cluster integer encoding matches the MATLAB convention
# from cluster_modes.m:
#   SOPH:  1 = sigma_high, 2 = sigma_low, 3 = theta-alpha, 4 = delta
#   SOPhH: 1 = upper,      2 = lower

run_pipeline <- function() {

  # ---- SOPH: pref_phase, one model per cluster ----
  d_soph <- load_table("combined_tbl_SOPH.csv")

  soph_results <- list()
  for (cl in 1:4) {
    cat(sprintf("\n=== SOPH cluster %d, pref_phase ===\n", cl))

    fit <- fit_circular_model(
      data            = d_soph,
      feature         = "pref_phase",
      cluster_value   = cl,
      order           = 3,
      include_electrode_x_age = TRUE,
      cache_tag       = "soph"
    )

    blocks <- joint_block_loo(
      full_fit      = fit,
      data          = d_soph,
      feature       = "pref_phase",
      cluster_value = cl,
      cache_tag     = "soph"
    )

    summ <- posterior_summary_table(fit)
    write_csv(summ, file.path(OUT_DIR,
              sprintf("posterior_summary_soph_cluster%d.csv", cl)))

    p <- trajectory_plot(fit)
    ggsave(file.path(OUT_DIR,
           sprintf("trajectory_soph_cluster%d.pdf", cl)),
           p, width = 7, height = 4)

    soph_results[[as.character(cl)]] <- list(
      fit      = fit,
      blocks   = blocks,
      summary  = summ
    )

    # Print the LOO block tests so they show up in the console log.
    cat("\nAge block (full vs no-age):\n")
    print(blocks$age_block_test)
    cat("\nElectrode x age block (full vs no-intx):\n")
    print(blocks$intx_block_test)
  }

  # ---- SOPhH: Theta, two clusters ----
  # Same structure.  Theta lives in (-pi, pi] in radians but is
  # conceptually a "tilt" parameter, not a preferred phase; nonetheless
  # it is treated identically as a circular response, which is what the
  # MATLAB pipeline does.
  d_sophh <- load_table("combined_tbl_SOPhH.csv")

  sophh_results <- list()
  for (cl in 1:2) {
    for (feat in c("Theta", "Phase")) {
      cat(sprintf("\n=== SOPhH cluster %d, %s ===\n", cl, feat))

      fit <- fit_circular_model(
        data          = d_sophh,
        feature       = feat,
        cluster_value = cl,
        order         = 3,
        include_electrode_x_age = TRUE,
        cache_tag     = "sophh"
      )

      blocks <- joint_block_loo(
        full_fit      = fit,
        data          = d_sophh,
        feature       = feat,
        cluster_value = cl,
        cache_tag     = "sophh"
      )

      summ <- posterior_summary_table(fit)
      write_csv(summ, file.path(OUT_DIR,
                sprintf("posterior_summary_sophh_cluster%d_%s.csv", cl, feat)))

      p <- trajectory_plot(fit)
      ggsave(file.path(OUT_DIR,
             sprintf("trajectory_sophh_cluster%d_%s.pdf", cl, feat)),
             p, width = 7, height = 4)

      sophh_results[[paste0(cl, "_", feat)]] <- list(
        fit     = fit,
        blocks  = blocks,
        summary = summ
      )

      cat("\nAge block (full vs no-age):\n")
      print(blocks$age_block_test)
      cat("\nElectrode x age block (full vs no-intx):\n")
      print(blocks$intx_block_test)
    }
  }

  list(soph = soph_results, sophh = sophh_results)
}


# =====================================================================
# 8. MAIN
# =====================================================================
# Comment out if sourcing for interactive use.

if (sys.nframe() == 0) {
  results <- run_pipeline()
  saveRDS(results, file.path(OUT_DIR, "all_results.rds"))
  cat("\nAll fits complete.  Outputs written to", OUT_DIR, "\n")
}


# =====================================================================
# APPENDIX -- MATLAB EXPORT SNIPPET
# =====================================================================
#
# Run this in MATLAB AFTER cluster_modes.m has built combined_tbl_SOPH
# and combined_tbl_SOPhH:
#
#   % Convert numeric electrode (1=central, 0=frontal) and sex (whatever
#   % the MATLAB column is) to strings for clarity, then write CSV.
#
#   tbl = combined_tbl_SOPH;
#   tbl.electrode = repmat("frontal", height(tbl), 1);
#   tbl.electrode(tbl.electrode_num == 1) = "central";   % rename in MATLAB
#                                                          first if needed
#   writetable(tbl, 'combined_tbl_SOPH.csv');
#
#   tbl = combined_tbl_SOPhH;
#   writetable(tbl, 'combined_tbl_SOPhH.csv');
#
# The R loader is forgiving about coding (numeric 0/1 or strings both
# work), so the simplest export is just writetable directly.
