# R bpnreg::bpnme - Bayesian projected-normal mixed-effects (Stan-backed).
# True projected-normal model: jointly fits the bivariate Gaussian on the
# (cos y, sin y) pair, projects to the circle. Bayesian counterpart of
# the frequentist lme4-sin/cos approach.
suppressPackageStartupMessages({
  library(bpnreg); library(readr); library(jsonlite)
})

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
d$Subj_ID <- as.numeric(factor(d$Subj_ID))  # bpnme expects numeric subject ID
d$y <- ((d$y + pi) %% (2*pi)) - pi

# Build a bpnme formula. bpnme syntax follows lme4 closely. Polynomial
# terms expanded explicitly because bpnme parses formulas more strictly.
ord <- meta$order
poly_terms <- if (ord >= 1) paste0("Age", c("", paste0("_p", 2:ord)[seq_len(ord-1)])) else character(0)
for (k in seq_len(ord)) {
  d[[ if (k==1) "Age" else paste0("Age_p",k) ]] <- d$Age^k
}
# Build the same polynomial terms in the eval grid:
for (k in seq_len(ord)) {
  grid[[ if (k==1) "Age" else paste0("Age_p",k) ]] <- grid$Age^k
}

rhs_terms <- character(0)
if (ord >= 1) rhs_terms <- c(rhs_terms, "Age")
if (ord >= 2) rhs_terms <- c(rhs_terms, sprintf("Age_p%d", 2:ord))
if (meta$has_electrode) {
  # electrode main + electrode * each polynomial term
  rhs_terms <- c(rhs_terms, "electrode",
                 paste0("electrode:", c("Age", paste0("Age_p", seq_len(ord)[-1]))))
}
if (meta$has_sex) rhs_terms <- c(rhs_terms, "sex")

if (length(rhs_terms) == 0) rhs_terms <- "1"
fml_str <- sprintf("y ~ %s + (1|Subj_ID)", paste(rhs_terms, collapse = " + "))
cat("bpnme formula:", fml_str, "\n")

set.seed(1)
fit <- bpnme(
  pred.I = as.formula(fml_str),
  data   = as.data.frame(d),
  its    = 2000,
  burn   = 500,
  n.lag  = 3
)

# Posterior predictive on the grid. bpnme stores beta as columns of a
# matrix (post draws x coef). We project (cos, sin) means.
# The bpnme output structure has:
#   fit$Beta.I  - posterior of "I" (intercept-coded) component coefficients
#   fit$Beta.II - posterior of "II" component
# These correspond to the two real parts of the bivariate normal.
beta_I  <- fit$Beta.I    # iter x P
beta_II <- fit$Beta.II   # iter x P

# Build design matrix for grid using the same column structure bpnme used.
# bpnme adds an intercept column automatically.
mm_grid <- model.matrix(as.formula(sub("^y\\s*~\\s*", "~ ", sub("\\s*\\+\\s*\\(1\\|Subj_ID\\)\\s*$", "", fml_str))),
                        data = grid)

# Posterior draws of (cos y, sin y) on the grid
I_draws  <- mm_grid %*% t(beta_I)   # N x iter
II_draws <- mm_grid %*% t(beta_II)  # N x iter

# Project each draw to an angle, then circular-mean across draws
ang_draws <- atan2(II_draws, I_draws)   # N x iter
yhat_mean <- atan2(rowMeans(sin(ang_draws)), rowMeans(cos(ang_draws)))
# Compute 95% CI by sorting the angle draws after un-wrapping relative to mean
wrap <- function(x) ((x + pi) %% (2*pi)) - pi
res_ang   <- wrap(ang_draws - yhat_mean)         # centered residuals
yhat_lo   <- yhat_mean + apply(res_ang, 1, quantile, 0.025)
yhat_hi   <- yhat_mean + apply(res_ang, 1, quantile, 0.975)

out <- data.frame(
  Age       = grid$Age,
  electrode = if ("electrode" %in% names(grid)) grid$electrode else 0,
  sex       = if ("sex"       %in% names(grid)) grid$sex       else 0,
  mean      = wrap(yhat_mean),
  lo        = wrap(yhat_lo),
  hi        = wrap(yhat_hi)
)
write_csv(out, "bpnreg_predictions.csv")

# Goodness of fit on training set
mm_train <- model.matrix(as.formula(sub("^y\\s*~\\s*", "~ ", sub("\\s*\\+\\s*\\(1\\|Subj_ID\\)\\s*$", "", fml_str))),
                         data = d)
I_train  <- mm_train %*% t(beta_I)
II_train <- mm_train %*% t(beta_II)
ang_tr   <- atan2(II_train, I_train)
yhat_tr  <- atan2(rowMeans(sin(ang_tr)), rowMeans(cos(ang_tr)))
ang_resid <- wrap(d$y - yhat_tr)
mu_y <- atan2(mean(sin(d$y)), mean(cos(d$y)))
sse  <- sum(1 - cos(ang_resid))
sst  <- sum(1 - cos(wrap(d$y - mu_y)))
R2_circ <- 1 - sse / max(sst, 1e-12)
mae_ang <- mean(abs(ang_resid))

# bpnme reports a posterior summary with DIC/WAIC depending on version
LL <- tryCatch(as.numeric(fit$lppd), error = function(e) NA_real_)

stats <- list(
  method      = "R bpnreg::bpnme (Bayesian projected-normal, Stan)",
  formula     = fml_str,
  n_obs       = nrow(d),
  n_subj      = length(unique(d$Subj_ID)),
  lppd        = LL,
  R2_circ     = R2_circ,
  mae_angular = mae_ang
)
write(toJSON(stats, auto_unbox = TRUE, pretty = TRUE), "bpnreg_stats.json")
cat("done. lppd=", LL, " R2_circ=", R2_circ, " mae=", mae_ang, "\n")
