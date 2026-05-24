# circ_fit_bpnreg_impl.R — bpnreg::bpnme Bayesian projected-normal mixed model.
# Order selection + AgeEffect by WAIC (Bayesian IC; no frequentist p-value).
# Two posterior coefficient sets (beta1/beta2) -> no single-beta table, so
# coefs/cov_b are not produced. Returns the standard result list.

fit_bpnreg <- function(d, grid, meta) {
  suppressPackageStartupMessages({ library(bpnreg) })

  has_elec <- as.logical(meta$has_electrode)
  has_sex  <- as.logical(meta$has_sex)
  d$Subj_ID <- as.numeric(factor(d$Subj_ID))
  d$y <- d$y %% (2*pi)

  max_ord <- if (isTRUE(as.logical(meta$select))) meta$max_order else meta$order
  for (k in seq_len(max_ord)) {
    nm <- if (k == 1) "Age" else paste0("Age_p", k)
    d[[nm]]    <- d$Age^k
    grid[[nm]] <- grid$Age^k
  }
  if (has_elec) { d$electrode <- factor(d$electrode, levels = c(0,1))
                  grid$electrode <- factor(grid$electrode, levels = c(0,1)) }
  if (has_sex)  { lv <- sort(unique(c(d$sex, grid$sex)))
                  d$sex <- factor(d$sex, levels = lv); grid$sex <- factor(grid$sex, levels = lv) }

  build_parts <- function(k) {
    if (k == 0) {
      parts <- character(0)
      if (has_elec) parts <- c(parts, "electrode")
    } else {
      poly_terms <- "Age"; if (k >= 2) poly_terms <- c(poly_terms, sprintf("Age_p%d", 2:k))
      if (has_elec) parts <- vapply(poly_terms, function(pt) sprintf("%s*electrode", pt), character(1))
      else          parts <- poly_terms
    }
    if (has_sex) parts <- c(parts, "sex")
    if (length(parts) == 0) parts <- "1"
    as.character(parts)
  }

  fit_one <- function(k) {
    parts <- build_parts(k)
    fml   <- as.formula(sprintf("y ~ %s + (1|Subj_ID)", paste(parts, collapse = " + ")))
    set.seed(meta$seed %||% 1)
    fit <- bpnme(pred.I = fml, data = as.data.frame(d), its = 2000, burn = 500, n.lag = 3)
    mf  <- as.list(as.data.frame(fit$model.fit))
    list(fit = fit, parts = parts, npar = ncol(fit$beta1),
         WAIC = num_or_na(mf$WAIC), lppd = num_or_na(mf$lppd))
  }

  # Training angle for R2 from a fit's posterior.
  train_angle <- function(o) {
    fix_form <- as.formula(sprintf("~ %s", paste(o$parts, collapse = " + ")))
    mm <- model.matrix(fix_form, data = d)
    fc <- intersect(colnames(o$fit$beta1), colnames(mm))
    mm <- mm[, fc, drop = FALSE]
    I  <- mm %*% t(o$fit$beta1[, fc, drop = FALSE])
    II <- mm %*% t(o$fit$beta2[, fc, drop = FALSE])
    ang <- atan2(II, I)
    atan2(rowMeans(sin(ang)), rowMeans(cos(ang)))
  }

  orders <- if (isTRUE(as.logical(meta$select))) 0:max_ord else meta$order
  O <- list(); WAICs <- c(); LLs <- c(); npars <- c(); R2s <- c()
  for (k in orders) {
    o <- fit_one(k); ki <- as.character(k)
    R2 <- circ_gof(wrap_pi(d$y), train_angle(o))$R2
    O[[ki]] <- o; WAICs[ki] <- o$WAIC; LLs[ki] <- o$lppd; npars[ki] <- o$npar; R2s[ki] <- R2
  }
  ok <- as.character(orders)

  # WAIC selection: lowest WAIC wins.
  chosen_i <- which.min(WAICs); if (length(chosen_i) == 0) chosen_i <- 1
  selected <- rep(FALSE, length(ok)); selected[chosen_i] <- TRUE
  chosen_order <- orders[chosen_i]; o <- O[[ok[chosen_i]]]

  order_table <- data.frame(order = orders, n_par = as.integer(npars),
                            LogLikelihood = as.numeric(LLs), R2_circ = as.numeric(R2s),
                            criterion_value = as.numeric(WAICs), WAIC = as.numeric(WAICs),
                            selected = selected, row.names = NULL)

  # --- predictions on the grid ---
  fix_form <- as.formula(sprintf("~ %s", paste(o$parts, collapse = " + ")))
  mm_grid  <- model.matrix(fix_form, data = grid)
  fc <- intersect(colnames(o$fit$beta1), colnames(mm_grid))
  mm_grid <- mm_grid[, fc, drop = FALSE]
  Ig  <- mm_grid %*% t(o$fit$beta1[, fc, drop = FALSE])
  IIg <- mm_grid %*% t(o$fit$beta2[, fc, drop = FALSE])
  ang <- atan2(IIg, Ig)
  mean_shifted <- atan2(rowMeans(sin(ang)), rowMeans(cos(ang)))
  res <- wrap_pi(ang - mean_shifted)
  lo_off <- apply(res, 1, quantile, 0.025); hi_off <- apply(res, 1, quantile, 0.975)
  predictions <- data.frame(
    Age = grid$Age,
    electrode = if ("electrode" %in% names(grid)) as.numeric(as.character(grid$electrode)) else 0,
    sex       = if ("sex"       %in% names(grid)) as.numeric(as.character(grid$sex))       else 0,
    mean_shifted = mean_shifted, lo_off = as.numeric(lo_off), hi_off = as.numeric(hi_off))

  # --- AgeEffect: WAIC, full vs no-Age (Bayesian IC; no p-value) ---
  o0 <- if ("0" %in% ok) O[["0"]] else fit_one(0)
  if (chosen_order >= 1) {
    age_effect <- list(pValue = NA_real_, stat = o0$WAIC - o$WAIC, df = NA_real_, method = "WAIC")
  } else {
    age_effect <- list(pValue = NA_real_, stat = NA_real_, df = NA_real_, method = "WAIC")
  }

  g <- circ_gof(wrap_pi(d$y), train_angle(o))
  stats <- list(LL = o$lppd, R2_circ = g$R2, mae_angular = g$MAE,
                AIC = NA_real_, BIC = NA_real_, chosen_order = chosen_order,
                select_criterion = "WAIC", age_effect = age_effect,
                diagnostics = list(WAIC = o$WAIC))

  list(predictions = predictions, order_table = order_table, stats = stats,
       coefs = NULL, cov_b = NULL)
}

num_or_na <- function(x) tryCatch(as.numeric(x), error = function(e) NA_real_)
`%||%` <- function(a, b) if (is.null(a)) b else a
