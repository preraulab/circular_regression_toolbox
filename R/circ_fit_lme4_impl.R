# circ_fit_lme4_impl.R — lme4 sin/cos parallel LMEs (frequentist
# projected-Gaussian). Order selection + AgeEffect via a combined sin+cos
# likelihood-ratio test. No single-beta coefficient table (two components
# behind atan2), so coefs/cov_b are not produced. Returns the standard list.

fit_lme4 <- function(d, grid, meta) {
  suppressPackageStartupMessages({ library(lme4) })

  has_elec <- as.logical(meta$has_electrode)
  has_sex  <- as.logical(meta$has_sex)
  d$sin_y <- sin(d$y); d$cos_y <- cos(d$y)

  build_rhs <- function(k) {
    if (k == 0) { rhs <- "1"; if (has_elec) rhs <- paste(rhs, "+ electrode") }
    else { rhs <- sprintf("poly(Age, %d, raw = TRUE)", k); if (has_elec) rhs <- paste0(rhs, " * electrode") }
    if (has_sex) rhs <- paste(rhs, "+ sex")
    rhs
  }
  fit_k <- function(k) {
    rhs <- build_rhs(k)
    fs <- lmer(as.formula(sprintf("sin_y ~ %s + (1|Subj_ID)", rhs)), data = d, REML = FALSE)
    fc <- lmer(as.formula(sprintf("cos_y ~ %s + (1|Subj_ID)", rhs)), data = d, REML = FALSE)
    list(sin = fs, cos = fc, npar = length(lme4::fixef(fs)),
         LL = as.numeric(logLik(fs)) + as.numeric(logLik(fc)))
  }

  orders <- if (isTRUE(as.logical(meta$select))) 0:meta$max_order else meta$order
  M <- list(); LLs <- c(); npars <- c(); R2s <- c()
  for (k in orders) {
    m <- fit_k(k); ki <- as.character(k)
    sh <- predict(m$sin, re.form = NA); ch <- predict(m$cos, re.form = NA)
    R2 <- circ_gof(d$y, atan2(sh, ch))$R2
    M[[ki]] <- m; LLs[ki] <- m$LL; npars[ki] <- m$npar; R2s[ki] <- R2
  }
  ok <- as.character(orders)

  # Combined sin+cos LRT per consecutive step; step up while p < 0.05.
  lrt_p <- rep(NA_real_, length(ok)); chosen_i <- 1
  for (i in 2:length(ok)) {
    chi <- max(2 * (LLs[i] - LLs[i-1]), 0)
    df  <- max(2 * (npars[i] - npars[i-1]), 1)
    lrt_p[i] <- pchisq(chi, df, lower.tail = FALSE)
    if (!is.na(lrt_p[i]) && lrt_p[i] < 0.05) chosen_i <- i else break
  }
  if (length(ok) == 1) chosen_i <- 1
  selected <- rep(FALSE, length(ok)); selected[chosen_i] <- TRUE
  chosen_order <- orders[chosen_i]; m <- M[[ok[chosen_i]]]

  order_table <- data.frame(order = orders, n_par = as.integer(npars),
                            LogLikelihood = as.numeric(LLs), R2_circ = as.numeric(R2s),
                            criterion_value = lrt_p, lrt_p = lrt_p,
                            selected = selected, row.names = NULL)

  # --- predictions on grid ---
  sh <- predict(m$sin, newdata = grid, re.form = NA, allow.new.levels = TRUE)
  ch <- predict(m$cos, newdata = grid, re.form = NA, allow.new.levels = TRUE)
  mean_shifted <- atan2(sh, ch)
  lo_off <- rep(0, length(mean_shifted)); hi_off <- rep(0, length(mean_shifted))
  if (isTRUE(as.logical(meta$band))) {
    bb <- tryCatch({
      ps <- bootMer(m$sin, function(x) predict(x, newdata = grid, re.form = NA, allow.new.levels = TRUE),
                    nsim = 200, type = "parametric", use.u = FALSE)$t
      pc <- bootMer(m$cos, function(x) predict(x, newdata = grid, re.form = NA, allow.new.levels = TRUE),
                    nsim = 200, type = "parametric", use.u = FALSE)$t
      ang <- atan2(ps, pc)                                  # nsim x Ngrid
      res <- wrap_pi(sweep(ang, 2, mean_shifted, "-"))
      list(lo = apply(res, 2, quantile, 0.025), hi = apply(res, 2, quantile, 0.975))
    }, error = function(e) { cat("lme4 bootMer band failed:", conditionMessage(e), "\n"); NULL })
    if (!is.null(bb)) { lo_off <- as.numeric(bb$lo); hi_off <- as.numeric(bb$hi) }
  }
  predictions <- data.frame(
    Age = grid$Age,
    electrode = if ("electrode" %in% names(grid)) grid$electrode else 0,
    sex       = if ("sex"       %in% names(grid)) grid$sex       else 0,
    mean_shifted = mean_shifted, lo_off = lo_off, hi_off = hi_off)

  # --- AgeEffect: combined LRT, chosen vs order-0 ---
  m0 <- if ("0" %in% ok) M[["0"]] else fit_k(0)
  chi <- max(2 * (m$LL - m0$LL), 0); df <- max(2 * (m$npar - m0$npar), 1)
  if (chosen_order >= 1) {
    age_effect <- list(pValue = pchisq(chi, df, lower.tail = FALSE),
                       stat = chi, df = df, method = "LRT-sincos")
  } else {
    age_effect <- list(pValue = NA_real_, stat = NA_real_, df = NA_real_, method = "LRT-sincos")
  }

  # --- GOF + diagnostics ---
  sh <- predict(m$sin, re.form = NA); ch <- predict(m$cos, re.form = NA)
  g  <- circ_gof(d$y, atan2(sh, ch))
  conv <- is.null(m$sin@optinfo$conv$lme4$messages) && is.null(m$cos@optinfo$conv$lme4$messages)
  stats <- list(LL = m$LL, R2_circ = g$R2, mae_angular = g$MAE,
                AIC = AIC(m$sin) + AIC(m$cos), BIC = BIC(m$sin) + BIC(m$cos),
                chosen_order = chosen_order, select_criterion = "LRT-sincos",
                age_effect = age_effect, diagnostics = list(converged = conv))

  list(predictions = predictions, order_table = order_table, stats = stats,
       coefs = NULL, cov_b = NULL)
}
