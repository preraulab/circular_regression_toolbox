# R lme4 sin/cos parallel LMEs (frequentist projected-Gaussian).
suppressPackageStartupMessages({
  library(lme4); library(readr); library(jsonlite)
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
d$Subj_ID <- factor(d$Subj_ID)

d$y <- ((d$y + pi) %% (2*pi)) - pi
d$sin_y <- sin(d$y); d$cos_y <- cos(d$y)

rhs <- sub("^y\\s*~\\s*", "", meta$formula)
rhs <- sub("\\+\\s*\\(1\\s*\\|\\s*Subj_ID\\)\\s*$", "", rhs)
rhs <- gsub("Age\\^(\\d+)", "poly(Age, \\1, raw = TRUE)", rhs)
fml_sin <- as.formula(sprintf("sin_y ~ %s + (1|Subj_ID)", trimws(rhs)))
fml_cos <- as.formula(sprintf("cos_y ~ %s + (1|Subj_ID)", trimws(rhs)))

fit_sin <- lmer(fml_sin, data = d, REML = FALSE)
fit_cos <- lmer(fml_cos, data = d, REML = FALSE)

s_hat_grid <- predict(fit_sin, newdata = grid, re.form = NA)
c_hat_grid <- predict(fit_cos, newdata = grid, re.form = NA)
yhat_grid  <- atan2(s_hat_grid, c_hat_grid)

out <- data.frame(
  Age       = grid$Age,
  electrode = if ("electrode" %in% names(grid)) grid$electrode else 0,
  sex       = if ("sex"       %in% names(grid)) grid$sex       else 0,
  yhat      = yhat_grid
)
write_csv(out, "lme4_predictions.csv")

# Goodness of fit on training data
s_hat_train <- predict(fit_sin, re.form = NA)
c_hat_train <- predict(fit_cos, re.form = NA)
yhat_train  <- atan2(s_hat_train, c_hat_train)
wrap <- function(x) ((x + pi) %% (2*pi)) - pi
ang_resid <- wrap(d$y - yhat_train)
mu_y <- atan2(mean(sin(d$y)), mean(cos(d$y)))
sse  <- sum(1 - cos(ang_resid))
sst  <- sum(1 - cos(wrap(d$y - mu_y)))
R2_circ <- 1 - sse / max(sst, 1e-12)
mae_ang <- mean(abs(ang_resid))

# Approximate joint LL (sin + cos), matching what fitlme_circ reports
LL <- as.numeric(logLik(fit_sin)) + as.numeric(logLik(fit_cos))

stats <- list(
  method      = "R lme4 sin/cos (frequentist projected-Gaussian)",
  formula     = format(fml_sin),
  n_obs       = nrow(d),
  n_subj      = nlevels(d$Subj_ID),
  LL_sin_plus_cos = LL,
  R2_circ     = R2_circ,
  mae_angular = mae_ang
)
write(toJSON(stats, auto_unbox = TRUE, pretty = TRUE), "lme4_stats.json")
cat("done. LL=", LL, " R2_circ=", R2_circ, " mae=", mae_ang, "\n")
