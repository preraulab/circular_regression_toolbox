# R brms (Stan-backed) vM-GLMM with identity link.
# Reads <results_dir>/data.csv + eval_grid.csv + meta.json.
# Writes <results_dir>/brms_predictions.csv + brms_stats.json.
suppressPackageStartupMessages({
  library(brms); library(readr); library(jsonlite)
})
options(mc.cores = parallel::detectCores())

args <- commandArgs(trailingOnly = TRUE)
if (length(args) >= 1) {
  results_dir <- args[1]
} else {
  results_dir <- "/Users/Mike/Desktop/phase_fit_dumps/results"
}
setwd(results_dir)

d    <- read_csv("data.csv",      show_col_types = FALSE)
grid <- read_csv("eval_grid.csv", show_col_types = FALSE)
meta <- fromJSON("meta.json")
d$Subj_ID    <- factor(d$Subj_ID)
grid$Subj_ID <- factor(grid$Subj_ID, levels = levels(d$Subj_ID))
d$y <- ((d$y + pi) %% (2*pi)) - pi

# y in data.csv has been pre-shifted by theta_shift (variance-min) so the
# seam falls in a data gap and the data looks Cartesian to the model.
# Predictions on the eval grid are produced in the shifted frame and
# unshifted at the end. Default 0 preserves backward compatibility.
theta_shift <- if (!is.null(meta$theta_shift)) as.numeric(meta$theta_shift) else 0
wrap <- function(x) ((x + pi) %% (2*pi)) - pi

# Standardize Age so the b ~ N(0, 0.5) prior is reasonable. Without this,
# raw Age (range ~7..80) combined with an identity-link vM model lets the
# linear predictor span many wrap-arounds, and the prior pulls coefs to 0.
age_mu <- mean(d$Age); age_sd <- sd(d$Age)
d$Age_z    <- (d$Age - age_mu) / age_sd
grid$Age_z <- (grid$Age - age_mu) / age_sd

cat("brms: n=", nrow(d), " subj=", nlevels(d$Subj_ID),
    " has_elec=", meta$has_electrode, " has_sex=", meta$has_sex,
    "  Age z-scored: mean=", round(age_mu,2), " sd=", round(age_sd,2), "\n")

# Translate "1 + Age^k * electrode + sex" -> brms-friendly, swapping in Age_z.
rhs <- sub("^y\\s*~\\s*", "", meta$formula)
rhs <- sub("\\+\\s*\\(1\\s*\\|\\s*Subj_ID\\)\\s*$", "", rhs)
rhs <- gsub("Age\\^(\\d+)", "poly(Age_z, \\1, raw = TRUE)", rhs)
rhs <- gsub("\\bAge\\b", "Age_z", rhs)
fml_str <- sprintf("y ~ %s + (1|Subj_ID)", trimws(rhs))
cat("brms formula:", fml_str, "\n")

cache_file <- sprintf("brms_%s_o%d_full", meta$feature, meta$order)

fit <- brm(
  as.formula(fml_str), data = d,
  # tan_half is the natural link for circular outcomes: it maps the
  # unbounded linear predictor onto (-pi, pi) via 2*atan(eta). With
  # identity link and data at the +-pi seam (e.g. Theta phase), the
  # posterior had multiple modes (eta ~ +pi vs eta ~ -pi parameterize
  # the same point on the circle), so chains failed to mix (Rhat ~ 2).
  family = von_mises(link = "tan_half", link_kappa = "log"),
  prior  = c(prior(normal(0, 2),         class = "Intercept"),
             prior(normal(0, 1),         class = "b"),
             prior(student_t(3, 0, 2.5), class = "sd"),
             prior(gamma(2, 0.5),        class = "kappa")),
  chains = 4, iter = 2000, warmup = 1000, seed = 1,
  control = list(adapt_delta = 0.95),
  file = cache_file, file_refit = "on_change"
)

# Population-marginal predictions (random effect = 0). For circular data,
# take the CIRCULAR mean across posterior draws (linear colMeans averages
# +pi and -pi to 0, which is wrong on the circle). Predictions are in
# the SHIFTED frame; unshift by adding theta_shift before writing so the
# CSV is on the original (-pi, pi] scale that the plot expects.
pred <- posterior_epred(fit, newdata = grid, re_formula = NA)
circ_mean <- function(M) atan2(colMeans(sin(M)), colMeans(cos(M)))
mean_pred_shifted <- circ_mean(pred)
res_draw  <- wrap(sweep(pred, 2, mean_pred_shifted, "-"))  # offsets are rotation-invariant
lo_off    <- apply(res_draw, 2, quantile, 0.025)
hi_off    <- apply(res_draw, 2, quantile, 0.975)
mean_pred <- wrap(mean_pred_shifted + theta_shift)
out  <- data.frame(
  Age       = grid$Age,
  electrode = if ("electrode" %in% names(grid)) grid$electrode else 0,
  sex       = if ("sex"       %in% names(grid)) grid$sex       else 0,
  mean      = mean_pred,
  lo        = wrap(mean_pred + lo_off),
  hi        = wrap(mean_pred + hi_off)
)
write_csv(out, "brms_predictions.csv")

# Goodness-of-fit on the training set (held-in by design):
#   - posterior log predictive density at observed y
#   - mean angular residual |y - yhat| (wrapped)
#   - circular R^2 = 1 - SSE_circ / SST_circ
yhat_train <- posterior_epred(fit, re_formula = NA)  # iter x N
# Circular posterior mean per column (linear colMeans averages +pi/-pi to 0
# and would break on the wrap-around).
yhat_mean  <- circ_mean(yhat_train)
ang_resid  <- wrap(d$y - yhat_mean)
mu_y      <- atan2(mean(sin(d$y)), mean(cos(d$y)))
sse       <- sum(1 - cos(ang_resid))
sst       <- sum(1 - cos(wrap(d$y - mu_y)))
R2_circ   <- 1 - sse / max(sst, 1e-12)
mae_ang   <- mean(abs(ang_resid))

# brms log-likelihood (per observation, posterior mean across draws)
llmat <- log_lik(fit)   # iter x N
ll_obs <- log(colMeans(exp(llmat - max(llmat))) ) + max(llmat) # log-mean-exp per obs
LL     <- sum(ll_obs)

stats <- list(
  method      = "R brms vM-GLMM (Stan, identity link)",
  formula     = fml_str,
  n_obs       = nrow(d),
  n_subj      = nlevels(d$Subj_ID),
  LL          = LL,
  R2_circ     = R2_circ,
  mae_angular = mae_ang,
  rhat_max    = max(rhat(fit), na.rm = TRUE),
  divergent   = sum(brms::nuts_params(fit)$Value[brms::nuts_params(fit)$Parameter == "divergent__"])
)
write(toJSON(stats, auto_unbox = TRUE, pretty = TRUE), "brms_stats.json")
cat("done. LL=", LL, " R2_circ=", R2_circ, " mae=", mae_ang, "\n")
