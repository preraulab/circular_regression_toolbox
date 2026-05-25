# circ_fit_common.R — shared helpers for the unified circular-regression
# R workers (sourced by circ_fit.R). Each backend impl returns a standard
# result list; finalize_and_write() does the unshift + per-electrode unwrap
# and writes the unified output contract.

`%||%` <- function(a, b) if (is.null(a)) b else a

wrap_pi <- function(x) ((x + pi) %% (2*pi)) - pi

# Remove 2*pi jumps to make an angle sequence continuous along its index.
unwrap_angles <- function(a) {
  if (length(a) < 2) return(a)
  da     <- diff(a)
  da_adj <- da - 2*pi*round(da/(2*pi))
  c(a[1], a[1] + cumsum(da_adj))
}

# Circular goodness of fit on the angle scale (matches across backends).
circ_gof <- function(y, yhat) {
  r   <- wrap_pi(y - yhat)
  mu  <- atan2(mean(sin(y)), mean(cos(y)))
  sse <- sum(1 - cos(r))
  sst <- sum(1 - cos(wrap_pi(y - mu)))
  list(R2 = 1 - sse / max(sst, 1e-12), MAE = mean(abs(r)))
}

# Identify the OMNIBUS age block: every coefficient that involves the x_col
# predictor, including interactions ('Age', 'Age^k', 'Age:cat', 'Age^k:cat').
# Used for the single "any age effect" Wald test.
age_block_idx <- function(names, x_col = "Age") {
  is_poly <- function(f) f == x_col || grepl(paste0("^", x_col, "\\^?[0-9]+$"), f)
  involves_age <- function(nm)
    any(vapply(strsplit(nm, ":", fixed = TRUE)[[1]], function(f) is_poly(trimws(f)), logical(1)))
  which(vapply(names, involves_age, logical(1)))
}

# Remap brms poly() fixef rownames -> Wilkinson 'Age^k' grammar.
# brms 2.23 mangles 'poly(Age_z, K, raw = TRUE)<col>' to e.g.
# 'polyAge_z1rawEQTRUE' (degree-1, no col suffix) or 'polyAge_z2rawEQTRUE1'
# (degree-2, col 1). The actual power is the LAST integer run in the token
# (which equals the degree when degree==1). Also handles ':electrode' and
# the bare-'Intercept' (no parens) that brms uses.
remap_poly_names <- function(names, x_col = "Age") {
  out <- names
  for (i in seq_along(names)) {
    nm <- names[i]
    if (nm == "(Intercept)" || nm == "Intercept") { out[i] <- "(Intercept)"; next }
    parts <- strsplit(nm, ":", fixed = TRUE)[[1]]
    parts2 <- vapply(parts, function(p) {
      if (grepl("poly", p)) {
        digs <- regmatches(p, gregexpr("[0-9]+", p))[[1]]
        if (length(digs)) {
          k <- as.integer(tail(digs, 1))
          if (k == 1) x_col else paste0(x_col, "^", k)
        } else x_col
      } else if (grepl(paste0("^", x_col, "_?z?$"), p)) {
        x_col
      } else {
        p
      }
    }, character(1))
    out[i] <- paste(parts2, collapse = ":")
  }
  out
}

# Take a backend result list and write the unified output contract.
#   res$predictions: data.frame(Age, electrode, sex, mean_shifted, lo_off, hi_off)
#   res$order_table: data.frame(order, n_par, LL, R2_circ, criterion_value, selected)
#   res$stats:       list(LL, R2_circ, mae_angular, AIC, BIC, chosen_order,
#                         select_criterion, age_effect=list(...), diagnostics=list(...))
#   res$coefs:       data.frame(name, estimate, se, pvalue) or NULL  (brms only)
#   res$cov_b:       matrix or NULL                                   (brms only)
finalize_and_write <- function(work_dir, backend, res, meta, d) {
  theta_shift <- if (!is.null(meta$theta_shift)) as.numeric(meta$theta_shift) else 0

  P <- res$predictions
  # Unshift to the original frame, then unwrap per electrode so the curve is
  # continuous; carry the same correction to the band via the stored offsets.
  P$electrode <- if ("electrode" %in% names(P)) P$electrode else 0
  P$sex       <- if ("sex"       %in% names(P)) P$sex       else 0
  P$mean <- NA_real_; P$lo <- NA_real_; P$hi <- NA_real_
  for (e in unique(P$electrode)) {
    idx  <- which(P$electrode == e)
    ord  <- idx[order(P$Age[idx])]
    m_orig <- wrap_pi(P$mean_shifted[ord] + theta_shift)
    m_cont <- unwrap_angles(m_orig)
    P$mean[ord] <- m_cont
    P$lo[ord]   <- m_cont + P$lo_off[ord]
    P$hi[ord]   <- m_cont + P$hi_off[ord]
  }
  preds_out <- P[, c("Age","electrode","sex","mean","lo","hi")]
  preds_out <- preds_out[order(preds_out$electrode, preds_out$Age), ]
  readr::write_csv(preds_out, file.path(work_dir, paste0(backend, "_predictions.csv")))

  readr::write_csv(res$order_table,
                   file.path(work_dir, paste0(backend, "_order_table.csv")))

  stats <- res$stats
  stats$n_obs  <- nrow(d)
  stats$n_subj <- length(unique(d$Subj_ID))
  stats$backend <- backend
  write(jsonlite::toJSON(stats, auto_unbox = TRUE, pretty = TRUE, na = "null"),
        file.path(work_dir, paste0(backend, "_stats.json")))

  if (!is.null(res$coefs)) {
    readr::write_csv(res$coefs, file.path(work_dir, paste0(backend, "_coefs.csv")))
  }
  if (!is.null(res$cov_b)) {
    utils::write.table(res$cov_b, file.path(work_dir, paste0(backend, "_cov_b.csv")),
                       sep = ",", row.names = FALSE, col.names = FALSE)
  }
  cat(sprintf("[%s] done: order=%s R2_circ=%.3f mae=%.3f\n",
              backend, stats$chosen_order, stats$R2_circ, stats$mae_angular))
}
