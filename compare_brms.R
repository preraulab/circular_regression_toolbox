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

cat("brms: n=", nrow(d), " subj=", nlevels(d$Subj_ID),
    " has_elec=", meta$has_electrode, " has_sex=", meta$has_sex, "\n")

# Translate "1 + Age^k * electrode + sex" -> brms-friendly
rhs <- sub("^y\\s*~\\s*", "", meta$formula)
rhs <- sub("\\+\\s*\\(1\\s*\\|\\s*Subj_ID\\)\\s*$", "", rhs)
rhs <- gsub("Age\\^(\\d+)", "poly(Age, \\1, raw = TRUE)", rhs)
fml_str <- sprintf("y ~ %s + (1|Subj_ID)", trimws(rhs))
cat("brms formula:", fml_str, "\n")

cache_file <- sprintf("brms_%s_o%d_full", meta$feature, meta$order)

fit <- brm(
  as.formula(fml_str), data = d,
  family = von_mises(link = "identity", link_kappa = "log"),
  prior  = c(prior(normal(0, 1),         class = "Intercept"),
             prior(normal(0, 0.5),       class = "b"),
             prior(student_t(3, 0, 2.5), class = "sd"),
             prior(gamma(2, 0.5),        class = "kappa")),
  chains = 4, iter = 2000, warmup = 1000, seed = 1,
  control = list(adapt_delta = 0.95),
  file = cache_file, file_refit = "on_change"
)

# Population-marginal predictions (random effect = 0). For circular data,
# take the CIRCULAR mean across posterior draws (linear colMeans averages
# +pi and -pi to 0, which is wrong on the circle).
pred <- posterior_epred(fit, newdata = grid, re_formula = NA)
wrap <- function(x) ((x + pi) %% (2*pi)) - pi
circ_mean <- function(M) atan2(colMeans(sin(M)), colMeans(cos(M)))
# CI: rotate each draw by -circular-mean, take quantiles, rotate back, wrap
mean_pred <- circ_mean(pred)
res_draw  <- wrap(sweep(pred, 2, mean_pred, "-"))   # iter x N centered
lo_off    <- apply(res_draw, 2, quantile, 0.025)
hi_off    <- apply(res_draw, 2, quantile, 0.975)
out  <- data.frame(
  Age       = grid$Age,
  electrode = if ("electrode" %in% names(grid)) grid$electrode else 0,
  sex       = if ("sex"       %in% names(grid)) grid$sex       else 0,
  mean      = wrap(mean_pred),
  lo        = wrap(mean_pred + lo_off),
  hi        = wrap(mean_pred + hi_off)
)
write_csv(out, "brms_predictions.csv")

# Goodness-of-fit on the training set (held-in by design):
#   - posterior log predictive density at observed y
#   - mean angular residual |y - yhat| (wrapped)
#   - circular R^2 = 1 - SSE_circ / SST_circ
yhat_train <- posterior_epred(fit, re_formula = NA)  # iter x N
yhat_mean  <- colMeans(yhat_train)
wrap <- function(x) ((x + pi) %% (2*pi)) - pi
ang_resid <- wrap(d$y - yhat_mean)
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
