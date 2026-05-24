# circ_fit_brms_impl.R — brms (Stan) von-Mises GLMM with tan_half link.
# Internal LOO order selection (step up while elpd_diff > 2*se_diff); a
# chi-square LRT p is reported alongside. Returns the standard result list.
# Sourced by circ_fit.R (wrap_pi, circ_gof, age_block_idx, remap_poly_names
# come from circ_fit_common.R).

fit_brms <- function(d, grid, meta) {
  suppressPackageStartupMessages({ library(brms); library(loo) })
  # Run chains sequentially (cores = 1). Parallel chain workers fail to
  # spawn when this script is launched from MATLAB's system() (the worker
  # processes can't initialize rstan), so force sequential sampling for
  # robustness; it is slower but works in every launch environment.

  has_elec <- as.logical(meta$has_electrode)
  has_sex  <- as.logical(meta$has_sex)
  x_col    <- if (!is.null(meta$x_col)) meta$x_col else "Age"

  # Standardize Age (tan_half link + raw Age would wrap many times).
  age_mu <- mean(d$Age); age_sd <- sd(d$Age)
  d$Age_z    <- (d$Age - age_mu) / age_sd
  grid$Age_z <- (grid$Age - age_mu) / age_sd

  build_fml <- function(k) {
    if (k == 0) {
      rhs <- "1"; if (has_elec) rhs <- paste(rhs, "+ electrode")
    } else {
      rhs <- sprintf("poly(Age_z, %d, raw = TRUE)", k)
      if (has_elec) rhs <- paste0(rhs, " * electrode")
    }
    if (has_sex) rhs <- paste(rhs, "+ sex")
    as.formula(sprintf("y ~ %s + (1|Subj_ID)", rhs))
  }

  pri <- c(prior(normal(0, 2),         class = "Intercept"),
           prior(normal(0, 1),         class = "b"),
           prior(student_t(3, 0, 2.5), class = "sd"),
           prior(gamma(2, 0.5),        class = "kappa"))

  chains <- meta$chains %||% 4; iter <- meta$iter %||% 2000
  warmup <- meta$warmup %||% 1000; seed <- meta$seed %||% 1
  adapt  <- meta$adapt_delta %||% 0.95
  if (warmup >= iter) warmup <- floor(iter / 2)   # keep warmup < iter
  orders <- if (isTRUE(as.logical(meta$select))) 0:meta$max_order else meta$order

  fits <- list(); LLs <- c(); npars <- c(); R2s <- c(); loos <- list()
  for (k in orders) {
    has_b <- (k >= 1) || has_elec || has_sex
    pk <- if (has_b) pri else pri[pri$class != "b", ]
    fit <- brm(build_fml(k), data = d,
               family = von_mises(link = "tan_half", link_kappa = "log"),
               prior = pk, chains = chains, cores = 1, iter = iter, warmup = warmup,
               seed = seed, control = list(adapt_delta = adapt),
               refresh = 0, silent = 2,
               file = sprintf("brms_%s_o%d", meta$feature, k),
               file_refit = "on_change")
    ll  <- log_lik(fit)
    LL  <- sum(log(colMeans(exp(ll - max(ll)))) + max(ll))
    yht <- posterior_epred(fit, re_formula = NA)
    yhm <- atan2(colMeans(sin(yht)), colMeans(cos(yht)))
    R2  <- circ_gof(d$y, yhm)$R2
    ki  <- as.character(k)
    fits[[ki]] <- fit; LLs[ki] <- LL
    npars[ki]  <- nrow(fixef(fit)); R2s[ki] <- R2
    loos[[ki]] <- loo(fit)
  }

  ok <- as.character(orders)
  crit_elpd <- vapply(ok, function(k) loos[[k]]$estimates["elpd_loo","Estimate"], numeric(1))
  elpd_se   <- vapply(ok, function(k) loos[[k]]$estimates["elpd_loo","SE"], numeric(1))

  # LRT p per consecutive step (reported alongside LOO).
  lrt_p <- rep(NA_real_, length(ok))
  for (i in seq_along(ok)[-1]) {
    chi <- max(2 * (LLs[i] - LLs[i-1]), 0)
    df  <- max(npars[i] - npars[i-1], 1)
    lrt_p[i] <- pchisq(chi, df, lower.tail = FALSE)
  }

  # LOO selection: take the order with the best elpd_loo, then prefer the
  # SIMPLEST order whose elpd is within 2*se_diff of that best (parsimony).
  # (A consecutive step-up that breaks at the first non-significant step can
  # stop early and miss a higher order that is decisively better.)
  best_i   <- which.max(crit_elpd)
  chosen_i <- best_i
  if (best_i > 1) {
    for (j in 1:(best_i - 1)) {
      lc <- loo_compare(loos[[ok[j]]], loos[[ok[best_i]]])
      se_diff <- lc[2, "se_diff"]
      if (!is.na(se_diff) && (crit_elpd[best_i] - crit_elpd[j]) <= 2 * se_diff) {
        chosen_i <- j; break
      }
    }
  }
  if (length(ok) == 1) chosen_i <- 1
  selected <- rep(FALSE, length(ok)); selected[chosen_i] <- TRUE
  chosen_order <- orders[chosen_i]
  fit <- fits[[ok[chosen_i]]]

  order_table <- data.frame(order = orders, n_par = as.integer(npars),
                            LogLikelihood = as.numeric(LLs), R2_circ = as.numeric(R2s),
                            criterion_value = as.numeric(crit_elpd),
                            elpd_loo = as.numeric(crit_elpd), elpd_se = as.numeric(elpd_se),
                            lrt_p = lrt_p, selected = selected, row.names = NULL)

  # --- chosen-order predictions on the grid (shifted frame + offsets) ---
  pred <- posterior_epred(fit, newdata = grid, re_formula = NA)
  mean_shifted <- atan2(colMeans(sin(pred)), colMeans(cos(pred)))
  res_draw <- wrap_pi(sweep(pred, 2, mean_shifted, "-"))
  lo_off <- apply(res_draw, 2, quantile, 0.025)
  hi_off <- apply(res_draw, 2, quantile, 0.975)
  predictions <- data.frame(
    Age = grid$Age,
    electrode = if ("electrode" %in% names(grid)) grid$electrode else 0,
    sex       = if ("sex"       %in% names(grid)) grid$sex       else 0,
    mean_shifted = mean_shifted, lo_off = as.numeric(lo_off), hi_off = as.numeric(hi_off))

  # --- coefficients (remapped) + posterior covariance ---
  fe   <- fixef(fit)
  nms  <- remap_poly_names(rownames(fe), x_col)
  est  <- fe[, "Estimate"]; se <- fe[, "Est.Error"]
  pval <- 2 * (1 - pnorm(abs(est / se)))
  coefs <- data.frame(name = nms, estimate = as.numeric(est),
                      se = as.numeric(se), pvalue = as.numeric(pval), row.names = NULL)
  V <- as.matrix(vcov(fit))
  # align vcov to fixef order
  V <- V[rownames(fe), rownames(fe), drop = FALSE]

  # --- AgeEffect: posterior Wald on the Age block ---
  ai <- age_block_idx(nms, x_col)
  if (length(ai) > 0) {
    R   <- matrix(0, length(ai), length(est)); for (j in seq_along(ai)) R[j, ai[j]] <- 1
    Rb  <- R %*% est; Vr <- R %*% V %*% t(R)
    chi <- as.numeric(t(Rb) %*% solve(Vr, Rb)); df <- length(ai)
    age_effect <- list(pValue = pchisq(chi, df, lower.tail = FALSE),
                       stat = chi, df = df, method = "Wald")
  } else {
    age_effect <- list(pValue = NA_real_, stat = NA_real_, df = NA_real_, method = "Wald")
  }

  # --- GOF + diagnostics at chosen order ---
  yht <- posterior_epred(fit, re_formula = NA)
  yhm <- atan2(colMeans(sin(yht)), colMeans(cos(yht)))
  g   <- circ_gof(d$y, yhm)
  np  <- brms::nuts_params(fit)
  diagnostics <- list(rhat_max = max(rhat(fit), na.rm = TRUE),
                      divergent = sum(np$Value[np$Parameter == "divergent__"]),
                      age_z_mu = age_mu, age_z_sd = age_sd)

  stats <- list(LL = LLs[ok[chosen_i]], R2_circ = g$R2, mae_angular = g$MAE,
                AIC = NA_real_, BIC = NA_real_, chosen_order = chosen_order,
                select_criterion = "LOO", age_effect = age_effect,
                diagnostics = diagnostics)

  list(predictions = predictions, order_table = order_table, stats = stats,
       coefs = coefs, cov_b = V)
}

`%||%` <- function(a, b) if (is.null(a)) b else a
