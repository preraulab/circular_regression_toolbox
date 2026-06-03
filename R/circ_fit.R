# circ_fit.R — unified entry point for the R circular-regression backends.
#
#   Rscript circ_fit.R <work_dir> <backend>
#
# backend in {brms, bpnreg}. Reads data.csv / eval_grid.csv / meta.json
# from work_dir, dispatches to the per-backend impl (which does its own
# polynomial-order selection), and writes the unified output contract:
#   <backend>_predictions.csv  (Age, electrode, sex, mean, lo, hi; unwrapped)
#   <backend>_stats.json       (LL, R2_circ, mae_angular, AIC, BIC, n_obs,
#                               n_subj, chosen_order, select_criterion,
#                               age_effect, diagnostics)
#   <backend>_order_table.csv  (order, n_par, LL, R2_circ, criterion_value, selected)
#   <backend>_coefs.csv + <backend>_cov_b.csv   (brms only)

suppressPackageStartupMessages({ library(readr); library(jsonlite) })

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop("usage: Rscript circ_fit.R <work_dir> <backend>")
work_dir <- args[1]
backend  <- tolower(args[2])

# Resolve this script's directory BEFORE changing cwd (so a relative
# --file= path still resolves correctly), then setwd into the work dir.
this_dir <- tryCatch({
  a <- commandArgs(FALSE)
  f <- sub("^--file=", "", a[grep("^--file=", a)])
  if (length(f)) dirname(normalizePath(f)) else getwd()
}, error = function(e) getwd())

work_dir <- normalizePath(work_dir)
setwd(work_dir)

source(file.path(this_dir, "circ_fit_common.R"))

d    <- read_csv("data.csv",      show_col_types = FALSE)
grid <- read_csv("eval_grid.csv", show_col_types = FALSE)
meta <- fromJSON("meta.json")
d$Subj_ID <- factor(d$Subj_ID)
d$y <- wrap_pi(d$y)

impl_file <- file.path(this_dir, sprintf("circ_fit_%s_impl.R", backend))
if (!file.exists(impl_file)) stop(sprintf("no impl for backend '%s'", backend))
source(impl_file)
fit_fun <- get(sprintf("fit_%s", backend))

res <- fit_fun(d, grid, meta)
finalize_and_write(work_dir, backend, res, meta, d)
