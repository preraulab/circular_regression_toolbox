# export_brms_summary.R
# From the cached brms fit at kappa=20, order=3, write:
#   1. A posterior parameter summary (for kappa, sigma_phi, sex, electrode)
#   2. Predicted population trajectories at a common age grid for each
#      electrode level, so MATLAB can overlay them on the comparison
#      plot without having to undo the internal Z-scoring of Age.

suppressPackageStartupMessages({
  library(brms)
  library(posterior)
  library(tidybayes)
})

fit <- readRDS("R_outputs_kappa/brms_K20_order3.rds")

# ---- (1) Posterior parameter summary ----
post <- as_draws_df(fit)
nm   <- names(post)
keep <- grepl("^b_|^kappa$|^sd_", nm)
post <- post[, keep, drop = FALSE]

out <- do.call(rbind, lapply(names(post), function(p) {
  x <- post[[p]]
  data.frame(
    term  = sub("^b_", "", p),
    raw   = p,
    mean  = mean(x),
    sd    = sd(x),
    q2_5  = unname(quantile(x, 0.025)),
    q97_5 = unname(quantile(x, 0.975)),
    stringsAsFactors = FALSE
  )
}))
write.csv(out, "R_outputs_kappa/brms_K20_order3_summary.csv", row.names = FALSE)
cat("Wrote brms_K20_order3_summary.csv\n")

# ---- (2) Predicted population trajectory ----
# Need the original training data to re-derive the Z-score that brms used.
csv_path <- "test_kappa_sweep_K20.csv"
d <- read.csv(csv_path)
age_mean <- mean(d$Age)
age_sd   <- sd(d$Age)

age_grid <- seq(7, 80, length.out = 200)
agez_grid <- (age_grid - age_mean) / age_sd

newdat <- expand.grid(
  Agez      = agez_grid,
  electrode = c(0, 1),
  sex       = 1
)
newdat$Subj_ID <- d$Subj_ID[1]

# Population-level prediction: re_formula = NA marginalizes the (1|Subj_ID).
preds <- add_epred_draws(newdat, fit, re_formula = NA, ndraws = 500)

# Summarize median and 95% CrI
preds_summary <- preds |>
  dplyr::group_by(Agez, electrode) |>
  dplyr::summarize(
    median = median(.epred),
    q2_5   = quantile(.epred, 0.025),
    q97_5  = quantile(.epred, 0.975),
    .groups = "drop"
  ) |>
  dplyr::arrange(electrode, Agez)
preds_summary$Age <- preds_summary$Agez * age_sd + age_mean

write.csv(preds_summary, "R_outputs_kappa/brms_K20_order3_trajectory.csv",
          row.names = FALSE)
cat("Wrote brms_K20_order3_trajectory.csv\n")
