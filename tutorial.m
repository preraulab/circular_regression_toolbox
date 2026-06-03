%TUTORIAL  Walkthrough: simulate vM-GLMM data and recover the truth.
%
% This script is the canonical first thing to run after installing the
% toolbox. It generates one synthetic dataset where we know every
% parameter (population trend, subject offsets, noise concentration),
% fits the von Mises GLMM with circ_fit, and shows how to read every
% piece of the result struct against the truth that produced it.
%
% Total runtime: a few seconds. Deterministic via rng(0).
%
% Topics covered, in order:
%   1.  How to simulate from a vM-GLMM by hand
%   2.  How to call circ_fit
%   3.  Reading SelectedOrder, AgeEffect, Coefficients
%   4.  Marginal vs conditional R^2 (R2_circ_marginal, R2_circ_conditional)
%   5.  Plotting the fitted population curve with circ_fit's CI band
%
% Run from MATLAB with the toolbox on the path:
%   addpath(genpath('/path/to/circular_regression_toolbox'));
%   run(fullfile('/path/to/circular_regression_toolbox', 'tutorial.m'));

clear; close all; rng(0);

%% ====================================================================
%% 1. Ground truth: what we will try to recover
%% ====================================================================
%
% Cohort: 100 subjects, 8 observations each (e.g. two electrodes x 4
% reps). The covariate Age is between-subject (each subject has one
% age), spanning 8 to 80 years.

n_subj   = 100;
n_per    = 8;
ages_subj = linspace(8, 80, n_subj)';   % one age per subject
Subj_ID  = repelem((1:n_subj)', n_per);
Age      = repelem(ages_subj, n_per);   % broadcast to all rows
n_obs    = numel(Age);

% --- True fixed effects ---------------------------------------------
% The population angular trend with age is a SHIFTED PARABOLA: starts
% high at age 8, dips around mid-life, comes back up at 80. Working in
% radians, we use a modest slope so the trajectory stays within ~one
% revolution and the von Mises likelihood is appropriate.
beta_intercept = 0.30;                   % radians (intercept at mean age)
beta_age       = -0.025;                 % rad/year (linear slope)
beta_age2      =  0.00050;               % rad/year^2 (curvature)

age_centered = Age - mean(Age);
mu_fixed = beta_intercept ...
         + beta_age  * age_centered ...
         + beta_age2 * age_centered.^2;

% --- True random and residual concentration -------------------------
%   kappa_phi  -> how alike subjects are (large = subjects share a
%                  baseline; small = each subject has a distinct
%                  preferred phase)
%   kappa_eps  -> within-subject residual noise (large = tight clusters
%                  around the fitted curve)
kappa_phi = 6;     % moderate between-subject heterogeneity
kappa_eps = 10;    % tight within-subject noise

% --- Draw the per-subject offsets and the per-row noise -------------
phi_subj = circ_vmrnd(0, kappa_phi, [n_subj, 1]);   % one per subject
eps_row  = circ_vmrnd(0, kappa_eps, [n_obs, 1]);    % one per row

% Generate the angles, wrapped to (-pi, pi].
y_unwrapped = mu_fixed + phi_subj(Subj_ID) + eps_row;
wrap = @(x) ((x + pi) - 2*pi*floor((x + pi)/(2*pi))) - pi;
Phase = wrap(y_unwrapped);

T = table(Subj_ID, Age, Phase);

fprintf('\nSimulated %d subjects, %d obs each (%d total rows).\n', ...
        n_subj, n_per, n_obs);
fprintf('Truth:  beta = [%.3f, %.4f, %.5g],  kappa = %.0f,  kappa_phi = %.0f\n', ...
        beta_intercept, beta_age, beta_age2, kappa_eps, kappa_phi);


%% ====================================================================
%% 2. Fit with circ_fit
%% ====================================================================
%
% The formula side resembles fitlme: response ~ fixed terms + random
% term. The toolbox automatically chooses the polynomial Age order by
% step-up likelihood-ratio test up to MaxOrder; with Select=false it
% fits the single order written into the formula.

result = circ_fit(T, ...
    'Phase ~ 1 + Age + Age^2 + (1|Subj_ID)', ...
    'fitcirc_lme', ...
    'Select', true, 'MaxOrder', 3);


%% ====================================================================
%% 3. Read the result against truth
%% ====================================================================

fprintf('\n--- Top-line answers ---\n');
fprintf('Selected polynomial order: %d   (truth = 2)\n', ...
        result.SelectedOrder);
fprintf('Omnibus Age test p-value:  %.3g (any-Age-effect joint Wald)\n', ...
        result.AgeEffect.pValue);
fprintf('R2_circ marginal:          %.3f (fixed effects only)\n', ...
        result.GOF.R2_circ_marginal);
fprintf('R2_circ conditional:       %.3f (fixed + subject baseline)\n', ...
        result.GOF.R2_circ_conditional);
fprintf('MAE_angular:               %.3f rad (= %.1f deg)\n', ...
        result.GOF.MAE_angular, rad2deg(result.GOF.MAE_angular));

% The coefficient table. Because circ_fit_fitcirc fits in an
% orthonormal-polynomial reparameterization for numerical conditioning
% (Age_op1, Age_op2 columns), the per-coefficient Estimates do not match
% beta_age and beta_age2 directly -- but the joint Wald and the fitted
% trajectory are invariant under that reparameterization.
fprintf('\n--- Coefficient table ---\n');
disp(result.Coefficients);

fprintf(['Note: Age coefficients are reported in an orthogonal-polynomial\n', ...
         'reparameterization (Age_op1, Age_op2) used for numerical\n', ...
         'conditioning. The fitted curve and the joint Age test are\n', ...
         'identical to what the raw [Age, Age^2] basis would give.\n']);


%% ====================================================================
%% 4. Marginal vs conditional R2 in plain words
%% ====================================================================
%
% R2_marginal answers "how well can we predict a brand-new subject from
% their Age alone?" It uses fixed effects only -- the subject random
% intercept is set to zero. If subjects all sat on the population curve
% perfectly, R2_marginal would be near 1. If subjects vary widely from
% the curve in ways Age does not capture, R2_marginal stays modest no
% matter how good the fit is.
%
% R2_conditional answers "how well can we predict if we have already
% measured this subject and know their personal baseline?" It uses
% fixed effects PLUS the subject random intercept. In a cohort with
% real per-subject heterogeneity, R2_conditional will be noticeably
% higher than R2_marginal.
%
% The gap between the two is a quantitative report on how much subject-
% to-subject variation there is on top of the population trend.

icc_estimated = result.GOF.R2_circ_conditional - result.GOF.R2_circ_marginal;
fprintf('\n--- Subject heterogeneity ---\n');
fprintf('R2_circ_conditional - R2_circ_marginal = %.3f\n', icc_estimated);
fprintf('This gap is the share of the response variation explained by\n');
fprintf('the subject random intercept on top of what Age explains.\n');


%% ====================================================================
%% 5. Plot the fit
%% ====================================================================
%
% plot_circ_fit draws the trajectory mean and CI band. The angular axis
% is repeated above and below at +-2*pi so the curve never "jumps" at
% the seam (a visualization trick documented in plot_circ_fit.m).

plot_circ_fit(result, T);
title(sprintf(['Tutorial fit: vM-GLMM (kappa = %.0f, kappa_phi = %.0f, ', ...
               'n = %d)'], kappa_eps, kappa_phi, n_obs));

fprintf('\nTutorial complete. Try:\n');
fprintf('  - Changing kappa_phi (line 67) to 1 (large subject variation)\n');
fprintf('    or 30 (very alike). Watch how R2_marginal and R2_conditional\n');
fprintf('    diverge or converge.\n');
fprintf('  - Setting beta_age2 = 0 and re-running. SelectedOrder should\n');
fprintf('    drop to 1 (the LRT no longer accepts a curvature term).\n');
fprintf('  - Adding a sex covariate: simulate it, add ''+ sex'' to the\n');
fprintf('    formula, refit, and inspect result.Coefficients.\n');
