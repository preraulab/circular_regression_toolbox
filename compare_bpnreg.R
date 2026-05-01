# R bpnreg::bpnme - Bayesian projected-normal mixed-effects model.
# Reads <results_dir>/data.csv + eval_grid.csv + meta.json.
# Writes <results_dir>/bpnreg_predictions.csv + bpnreg_stats.json.
#
# Notes on bpnme (v2.0.3) output we rely on:
#   fit$beta1, fit$beta2   - posterior fixed effects, dim = iter x P,
#                            colnames match the design matrix columns
#   fit$mm$XI, fit$mm$XII  - design matrices the fit used
#   fit$model.fit          - matrix with lppd, WAIC, DIC columns
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

# y is pre-shifted in data.csv; unshift predictions before writing.
theta_shift <- if (!is.null(meta$theta_shift)) as.numeric(meta$theta_shift) else 0
wrap <- function(x) ((x + pi) %% (2*pi)) - pi

# bpnme needs a numeric grouping column and y on [0, 2*pi)
d$Subj_ID <- as.numeric(factor(d$Subj_ID))
d$y       <- d$y %% (2*pi)

# Build the bpnme formula. bpnme requires CATEGORICAL predictors to be
# factors (it dispatches differently for factors vs numerics in its
# summary code). We treat electrode and sex as factors and let
# bpnme handle the * interaction expansion via R's standard formula
# operators.
ord <- meta$order
add_col <- function(df, name, vals) { df[[name]] <- vals; df }
for (k in seq_len(ord)) {
  nm <- if (k == 1) "Age" else paste0("Age_p", k)
  d   <- add_col(d,   nm, d$Age^k)
  grid<- add_col(grid,nm, grid$Age^k)
}
if (as.logical(meta$has_electrode)) {
  d$electrode    <- factor(d$electrode,    levels = c(0,1))
  grid$electrode <- factor(grid$electrode, levels = c(0,1))
}
if (as.logical(meta$has_sex)) {
  d$sex    <- factor(d$sex,    levels = sort(unique(c(d$sex, grid$sex))))
  grid$sex <- factor(grid$sex, levels = levels(d$sex))
}

# Build a Wilkinson formula. With electrode as a factor, "Age*electrode"
# expands to Age + electrode + Age:electrode and bpnme handles the
# expansion itself.
poly_terms <- "Age"
if (ord >= 2) poly_terms <- c(poly_terms, sprintf("Age_p%d", 2:ord))
rhs_parts <- character(0)
if (as.logical(meta$has_electrode)) {
  for (pt in poly_terms) rhs_parts <- c(rhs_parts, sprintf("%s*electrode", pt))
} else {
  rhs_parts <- poly_terms
}
if (as.logical(meta$has_sex)) rhs_parts <- c(rhs_parts, "sex")
if (length(rhs_parts) == 0) rhs_parts <- "1"

fml_str <- sprintf("y ~ %s + (1|Subj_ID)", paste(rhs_parts, collapse = " + "))
cat("bpnme formula:", fml_str, "\n  n=", nrow(d), " subj=", length(unique(d$Subj_ID)), "\n")

set.seed(1)
fit <- bpnme(
  pred.I = as.formula(fml_str),
  data   = as.data.frame(d),
  its    = 2000,
  burn   = 500,
  n.lag  = 3
)

# Posterior fixed-effect draws (iter x P)
beta1 <- fit$beta1
beta2 <- fit$beta2
P     <- ncol(beta1)
cat("bpnme fit done. iter=", nrow(beta1), " P=", P, "  cols:", paste(colnames(beta1), collapse=","), "\n")

# Build design matrix on the eval grid using the SAME column structure
# bpnme used internally (fit$mm$XI). The fit removed the random-effect
# block from the formula already, so its colnames are the fixed-effect cols.
fix_cols <- colnames(beta1)
# Match by re-building the model matrix from the fixed-effects formula and
# then realigning columns; safer than relying on R's term ordering.
fix_form <- as.formula(sprintf("~ %s", paste(rhs_parts, collapse = " + ")))
mm_grid  <- model.matrix(fix_form, data = grid)
mm_train <- model.matrix(fix_form, data = d)

# Realign columns to match beta1 column order. If a column is missing
# (e.g. the data.frame has it as factor and bpnme expanded differently),
# fall back to the intersection.
missing_cols <- setdiff(fix_cols, colnames(mm_grid))
if (length(missing_cols) > 0) {
  cat("WARNING: design matrix missing columns vs beta1:", paste(missing_cols, collapse=","), "\n")
  fix_cols <- intersect(fix_cols, colnames(mm_grid))
  beta1 <- beta1[, fix_cols, drop = FALSE]
  beta2 <- beta2[, fix_cols, drop = FALSE]
}
mm_grid  <- mm_grid[,  fix_cols, drop = FALSE]
mm_train <- mm_train[, fix_cols, drop = FALSE]

# Posterior draws of the bivariate latent (component I, II) -> angle.
# Components I,II are treated as cos/sin halves of an unnormalized vector;
# bpnme uses I = component 1 = "cos-like", II = "sin-like".
I_grid  <- mm_grid  %*% t(beta1)   # N_grid  x iter
II_grid <- mm_grid  %*% t(beta2)
I_train <- mm_train %*% t(beta1)   # N_train x iter
II_train<- mm_train %*% t(beta2)

ang_grid  <- atan2(II_grid,  I_grid)        # N x iter, range (-pi, pi]
ang_train <- atan2(II_train, I_train)

# Circular posterior mean per row
circ_row_mean <- function(M) atan2(rowMeans(sin(M)), rowMeans(cos(M)))
yhat_grid_mu  <- circ_row_mean(ang_grid)
yhat_train_mu <- circ_row_mean(ang_train)

# 95% CI via centered residual quantiles, then re-shift
res_grid <- wrap(ang_grid - yhat_grid_mu)            # N x iter
lo_off   <- apply(res_grid, 1, quantile, 0.025)
hi_off   <- apply(res_grid, 1, quantile, 0.975)

mean_unshift <- wrap(yhat_grid_mu + theta_shift)
out <- data.frame(
  Age       = grid$Age,
  electrode = if ("electrode" %in% names(grid)) grid$electrode else 0,
  sex       = if ("sex"       %in% names(grid)) grid$sex       else 0,
  mean      = mean_unshift,
  lo        = wrap(mean_unshift + lo_off),
  hi        = wrap(mean_unshift + hi_off)
)
write_csv(out, "bpnreg_predictions.csv")

# Goodness of fit on the training set. Compare predictions in (-pi, pi];
# d$y was put on [0, 2pi) for bpnme, so wrap both back to (-pi, pi] for
# residuals consistent with the other comparisons.
dy_wrapped <- wrap(d$y)
ang_resid  <- wrap(dy_wrapped - yhat_train_mu)
mu_y       <- atan2(mean(sin(dy_wrapped)), mean(cos(dy_wrapped)))
sse        <- sum(1 - cos(ang_resid))
sst        <- sum(1 - cos(wrap(dy_wrapped - mu_y)))
R2_circ    <- 1 - sse / max(sst, 1e-12)
mae_ang    <- mean(abs(ang_resid))

# Pull bpnme-reported model fit (lppd, WAIC, DIC)
mf <- as.list(as.data.frame(fit$model.fit))
LL <- tryCatch(as.numeric(mf$lppd), error = function(e) NA_real_)

stats <- list(
  method      = "R bpnreg::bpnme (Bayesian projected-normal, Stan)",
  formula     = fml_str,
  n_obs       = nrow(d),
  n_subj      = length(unique(d$Subj_ID)),
  LL          = LL,
  WAIC        = tryCatch(as.numeric(mf$WAIC),  error=function(e) NA_real_),
  DIC         = tryCatch(as.numeric(mf$DIC),   error=function(e) NA_real_),
  R2_circ     = R2_circ,
  mae_angular = mae_ang
)
write(toJSON(stats, auto_unbox = TRUE, pretty = TRUE), "bpnreg_stats.json")
cat("done. lppd=", LL, " R2_circ=", R2_circ, " mae=", mae_ang, "\n")
