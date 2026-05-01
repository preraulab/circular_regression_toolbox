# test_order_selection.R
#
# R-side companion to test_order_selection.m.  Runs the same iterative
# order-selection procedure for two R methods on the same N=300
# synthetic data:
#
#   1. circular::lm.circular (Fisher-Lee MLE, no random effects)
#      via likelihood-ratio chi-squared test.
#
#   2. brms von_mises mixed-effects via leave-one-out cross-validation
#      comparison (Bayesian analog: accept higher order if elpd_diff is
#      > 2 * se_diff in favor of the larger model).
#
# Both report (a) the order that the iterative procedure stops at,
# (b) the "block" comparison of every order against order 0, and
# (c) AIC / WAIC for each fit.

suppressPackageStartupMessages({
  library(circular)
  library(brms)
  library(loo)
  library(readr)
})
options(mc.cores = parallel::detectCores())

OUT <- "R_outputs_order"
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------
d <- read_csv("test_synthetic_data_N300.csv", show_col_types = FALSE)
d$Subj_ID <- factor(d$Subj_ID)
d$electrode <- as.numeric(d$electrode)
d$sex       <- as.numeric(d$sex)

age_mean <- mean(d$Age)
age_sd   <- sd(d$Age)
d$Agez   <- (d$Age - age_mean) / age_sd
d$phase_pref <- ((d$phase_pref + pi) %% (2*pi)) - pi
d$phase_circ <- circular(d$phase_pref, units = "radians",
                         modulo = "2pi", template = "none")

cat(sprintf("Loaded %d obs from %d subjects\n", nrow(d), nlevels(d$Subj_ID)))
cat("Truth: cubic age main effect; selected order should be 3\n\n")

orders <- 0:4

# ---------------------------------------------------------------------
# (1) circular::lm.circular at each order: log-likelihood + npar
# ---------------------------------------------------------------------
build_X_at_order <- function(k, dat) {
  if (k == 0) {
    X <- model.matrix(~ electrode + sex, data = dat)
  } else {
    poly_terms <- sprintf("I(Agez^%d)", 1:k)
    intx_terms <- sprintf("I(Agez^%d):electrode", 1:k)
    rhs <- paste(c(poly_terms, "electrode", "sex", intx_terms), collapse = " + ")
    X <- model.matrix(as.formula(paste("~", rhs)), data = dat)
  }
  X[, -1, drop = FALSE]   # drop intercept; lm.circular has its own
}

LL_lc <- numeric(length(orders))
np_lc <- numeric(length(orders))
for (i in seq_along(orders)) {
  k <- orders[i]
  X <- build_X_at_order(k, d)
  fit <- tryCatch(lm.circular(y = d$phase_circ, x = X, type = "c-l",
                              init = rep(0, ncol(X)), verbose = FALSE),
                  error = function(e) NULL)
  if (is.null(fit)) {
    LL_lc[i] <- NA; np_lc[i] <- NA
    next
  }
  # Compute log-likelihood directly: n * (-log(2*pi*I0(kappa))) +
  # kappa * sum(cos(residuals))
  eta <- as.numeric(cbind(1, X) %*% c(fit$mu, fit$coefficients))
  r   <- ((d$phase_pref - eta + pi) %% (2*pi)) - pi
  LL_lc[i] <- -length(r) * log(2*pi*besselI(fit$kappa, 0)) +
               fit$kappa * sum(cos(r))
  np_lc[i] <- ncol(X) + 1   # +1 for intercept (mu)
}

cat("--- circular::lm.circular log-likelihood by order ---\n")
print(data.frame(order = orders, LL = LL_lc, npar = np_lc), row.names = FALSE)

iter_lrt <- function(LL, npar) {
  sel <- 0
  for (k in 2:length(LL)) {
    chi2 <- 2 * (LL[k] - LL[k-1])
    df   <- npar[k] - npar[k-1]
    p    <- pchisq(max(chi2, 0), df, lower.tail = FALSE)
    accepted <- !is.na(p) && p < 0.05
    cat(sprintf("  order %d -> %d:  chi2(%d) = %.2f  p = %.4g  %s\n",
                k-2, k-1, df, chi2, p,
                if (accepted) "[accept]" else "[stop]"))
    if (accepted) sel <- k - 1 else break
  }
  sel
}

cat("\n--- Iterative LRT (lm.circular) ---\n")
sel_lc <- iter_lrt(LL_lc, np_lc)

cat("\n--- Block joint LRT vs order 0 (lm.circular) ---\n")
for (k in 2:length(orders)) {
  chi2 <- 2 * (LL_lc[k] - LL_lc[1])
  df   <- np_lc[k] - np_lc[1]
  p    <- pchisq(max(chi2, 0), df, lower.tail = FALSE)
  cat(sprintf("  order 0 -> %d:  chi2(%d) = %.2f  p = %.4g\n",
              orders[k], df, chi2, p))
}

# ---------------------------------------------------------------------
# (2) brms von_mises mixed-effects at each order: LOO comparison
# ---------------------------------------------------------------------
priors <- c(
  prior(normal(0, 1),         class = "Intercept"),
  prior(normal(0, 1),         class = "b"),
  prior(student_t(3, 0, 2.5), class = "sd"),
  prior(gamma(2, 0.5),        class = "kappa")
)

build_brms_formula <- function(k) {
  if (k == 0) {
    as.formula("phase_pref ~ electrode + sex + (1 | Subj_ID)")
  } else {
    as.formula(sprintf("phase_pref ~ poly(Agez, %d, raw = TRUE) * electrode + sex + (1 | Subj_ID)", k))
  }
}

cat("\n--- Fitting brms at each order (cached if previously fit) ---\n")
fits <- list()
loos <- list()
for (i in seq_along(orders)) {
  k <- orders[i]
  cat(sprintf("  order %d... ", k))
  fits[[i]] <- brm(
    formula = build_brms_formula(k),
    data    = d,
    family  = von_mises(link = "tan_half", link_kappa = "log"),
    prior   = priors,
    chains  = 4, iter = 3000, warmup = 1000,
    control = list(adapt_delta = 0.95),
    seed    = 1, refresh = 0,
    file    = file.path(OUT, sprintf("brms_order%d.rds", k)),
    file_refit = "on_change",
    silent  = 2
  )
  loos[[i]] <- loo(fits[[i]])
  cat(sprintf("elpd_loo = %.2f\n", loos[[i]]$estimates["elpd_loo", "Estimate"]))
}

# Iterative LOO selection: accept higher order if elpd_diff > 2 * se_diff
# in favor of the higher-order model.  This is the conventional "2-sigma
# detectable improvement" cutoff used in the brms ecosystem.
cat("\n--- Iterative LOO selection (brms) ---\n")
sel_brms <- 0
for (k in 2:length(orders)) {
  cmp <- loo_compare(loos[[k-1]], loos[[k]])
  # cmp: row 1 is best model, row 2 is second-best (with negative elpd_diff)
  # Find which row corresponds to the higher-order model
  best_name <- rownames(cmp)[1]
  d_diff <- cmp[2, "elpd_diff"]   # always negative or zero
  d_se   <- cmp[2, "se_diff"]

  # If the higher-order model is "best" (i.e., other model has negative elpd_diff)
  # AND the gap is more than 2 SEs, accept higher order.
  higher_better <- best_name == "..1" || best_name == "fits[[k]]" ||
                   best_name == "model2"
  # loo_compare names rows by the input order; we passed (k-1, k), so
  # row 1 = whichever is best, identified by deparse.
  # Robust check: compare elpd_loo directly.
  e1 <- loos[[k-1]]$estimates["elpd_loo","Estimate"]
  e2 <- loos[[k]]$estimates["elpd_loo","Estimate"]
  e_diff <- e2 - e1
  e_se   <- sqrt(loos[[k-1]]$estimates["elpd_loo","SE"]^2 +
                 loos[[k]]$estimates["elpd_loo","SE"]^2)
  # Better: use loo_compare se directly
  e_se_compare <- abs(d_se)

  accepted <- (e_diff > 2 * e_se_compare)
  cat(sprintf("  order %d -> %d:  elpd_diff = %.2f, se = %.2f, z = %.2f  %s\n",
              orders[k-1], orders[k], e_diff, e_se_compare,
              e_diff / max(e_se_compare, 1e-9),
              if (accepted) "[accept]" else "[stop]"))
  if (accepted) sel_brms <- orders[k] else break
}

cat("\n--- Block joint LOO comparison vs order 0 (brms) ---\n")
for (k in 2:length(orders)) {
  cmp <- loo_compare(loos[[1]], loos[[k]])
  e1 <- loos[[1]]$estimates["elpd_loo","Estimate"]
  e2 <- loos[[k]]$estimates["elpd_loo","Estimate"]
  e_diff <- e2 - e1
  e_se   <- abs(cmp[2, "se_diff"])
  cat(sprintf("  order 0 -> %d:  elpd_diff = %.2f, se = %.2f, z = %.2f\n",
              orders[k], e_diff, e_se, e_diff / max(e_se, 1e-9)))
}

# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------
cat("\n========================================================\n")
cat("  SUMMARY: selected order by method (truth = 3)\n")
cat("========================================================\n")
cat(sprintf("  circular::lm.circular (iterative LRT):  order = %d\n", sel_lc))
cat(sprintf("  brms von_mises mixed (iterative LOO):   order = %d\n", sel_brms))
