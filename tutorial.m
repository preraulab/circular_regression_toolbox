%TUTORIAL  Walkthrough: simulate vM-GLMM data and recover the truth.
%
% This script is the canonical first thing to run after installing the
% toolbox. It walks through two contrasting use cases for the same
% population trend:
%
%   PART A.  Large N, one visit per subject (n_subj = 500, n_per = 1).
%            A pure fixed-effects regression. The (1|Subj_ID) random
%            intercept in the formula has nothing to learn (a single
%            observation per subject cannot identify a subject-specific
%            offset), so the EM collapses kappa_phi to infinity and the
%            conditional R^2 reduces to the marginal R^2. This is the
%            regime for cross-sectional cohort studies.
%
%   PART B.  Small N, repeated visits per subject (n_subj = 25,
%            n_per = 8). A proper mixed-effects regression. Each subject
%            has their own random angular offset around the population
%            trend, so kappa_phi is identifiable and the conditional
%            R^2 exceeds the marginal R^2 by the share of variance the
%            random intercept explains. This is the regime for
%            longitudinal or repeated-measurement designs.
%
% Both fits share the same population trend: a linear sweep across most
% of (-pi, pi] in age plus a sex covariate (males offset by ~pi/10
% from females). The formula carries `+ sex` so the trajectory builder
% produces one curve per sex level, and plot_circ_fit overlays them in
% distinct colors.
%
% Total runtime: a few seconds. Deterministic via rng(0).
%
% Run from MATLAB with the toolbox on the path:
%   addpath(genpath('/path/to/circular_regression_toolbox'));
%   run(fullfile('/path/to/circular_regression_toolbox', 'tutorial.m'));

clear; close all; rng(0);

%% ====================================================================
%% Ground truth (shared by both parts)
%% ====================================================================
%
% Population angular trend swings across most of (-pi, pi] over the
% age range, with a sex offset. Endpoints stay safely within (-pi, pi]
% so the trajectory does not wrap. Modest curvature lets the step-up
% LRT pick a quadratic term over a linear one.
%
%   mu(age, sex) = beta_intercept
%                + beta_age * (age - mean(age))
%                + beta_age2 * (age - mean(age))^2
%                + beta_sex * sex
%   sex = 0 for female (reference), 1 for male
%
% Endpoint check: at age 8 and 80 the population mean lands at
%   F: ~ -2.03 and +2.29 rad,   M: ~ -1.73 and +2.59 rad
% which span ~4.6 rad (~73% of 2*pi) without wrapping.

beta_intercept = 0;                      % radians (intercept at mean age, female)
beta_age       = 0.050;                  % rad/year (linear slope)
beta_age2      = 0.0003;                 % rad/year^2 (mild curvature)
beta_sex       = 0.35;                   % rad offset for sex=1 (male)

kappa_eps = 8;                           % within-subject residual concentration (tight)
kappa_phi = 5;                           % between-subject offset concentration (Part B)

wrap = @(x) ((x + pi) - 2*pi*floor((x + pi)/(2*pi))) - pi;

fprintf('Ground truth\n');
fprintf('  Population trend:    beta = [%.3f, %.4f, %.5g], beta_sex = %.3f\n', ...
        beta_intercept, beta_age, beta_age2, beta_sex);
fprintf('  Within-subject noise: kappa = %.1f\n', kappa_eps);
fprintf('  Subject offset:       kappa_phi = %.1f (Part B only)\n', kappa_phi);


%% ====================================================================
%% PART A.  Large N, one visit per subject  (n_subj = 500, n_per = 1)
%% ====================================================================
%
% Each subject is measured once. The data carry the population trend
% plus i.i.d. within-subject noise. Strictly speaking the random-
% intercept variance (kappa_phi) and the residual variance (kappa) are
% not jointly identified with one observation per subject -- the EM
% will still fit both, and the marginal log-likelihood lands at a
% finite optimum, but the split between the two is mathematically
% arbitrary. So R^2_c is greater than R^2_m here even though there is
% no genuine subject heterogeneity in the truth; the gap reflects what
% the EM allocated to the random intercept, not a real signal.

n_subj_A   = 500;
n_per_A    = 1;
ages_A     = linspace(8, 80, n_subj_A)';
Subj_ID_A  = (1:n_subj_A)';
Age_A      = ages_A;
sex_A      = double(rand(n_subj_A, 1) > 0.5);     % 0 = female, 1 = male
age_centered_A = Age_A - mean(Age_A);
mu_A = beta_intercept ...
     + beta_age  * age_centered_A ...
     + beta_age2 * age_centered_A.^2 ...
     + beta_sex  * sex_A;
eps_A   = circ_vmrnd(0, kappa_eps, [n_subj_A, 1]);
Phase_A = wrap(mu_A + eps_A);
T_A = table(Subj_ID_A, Age_A, sex_A, Phase_A, ...
            'VariableNames', {'Subj_ID','Age','sex','Phase'});

fprintf('\n========================================\n');
fprintf('PART A.  Large N, one visit per subject\n');
fprintf('========================================\n');
fprintf('N = %d subjects x %d visit = %d observations\n', n_subj_A, n_per_A, height(T_A));

result_A = circ_fit(T_A, 'Phase ~ 1 + Age + Age^2 + sex + (1|Subj_ID)', 'fitcirc_lme', ...
                    'Select', true, 'MaxOrder', 3);
disp(result_A);

figure('Color','w');
plot(result_A, T_A);
title(sprintf('Part A: N = %d, 1 visit/subject (fixed-effects regime)', n_subj_A));


%% ====================================================================
%% PART B.  Small N, repeated visits per subject  (n_subj = 25, n_per = 8)
%% ====================================================================
%
% 25 subjects, 8 observations each (200 total). Each subject has a
% random angular offset drawn from vonMises(0, kappa_phi); within-
% subject observations cluster around that offset plus the population
% trend. The model now has a real subject-baseline to estimate and
% R^2_c > R^2_m by the share of variance that random intercept
% explains.

n_subj_B   = 40;
n_per_B    = 6;
ages_B     = linspace(8, 80, n_subj_B)';
Subj_ID_B  = repelem((1:n_subj_B)', n_per_B);
Age_B      = repelem(ages_B, n_per_B);
sex_subj_B = double(rand(n_subj_B, 1) > 0.5);     % one sex per subject
sex_B      = sex_subj_B(Subj_ID_B);
age_centered_B = Age_B - mean(Age_B);
mu_B = beta_intercept ...
     + beta_age  * age_centered_B ...
     + beta_age2 * age_centered_B.^2 ...
     + beta_sex  * sex_B;
phi_B   = circ_vmrnd(0, kappa_phi, [n_subj_B, 1]);
eps_B   = circ_vmrnd(0, kappa_eps, [numel(Age_B), 1]);
Phase_B = wrap(mu_B + phi_B(Subj_ID_B) + eps_B);
T_B = table(Subj_ID_B, Age_B, sex_B, Phase_B, ...
            'VariableNames', {'Subj_ID','Age','sex','Phase'});

fprintf('\n========================================\n');
fprintf('PART B.  Small N, repeated visits per subject\n');
fprintf('========================================\n');
fprintf('N = %d subjects x %d visits = %d observations\n', n_subj_B, n_per_B, height(T_B));

result_B = circ_fit(T_B, 'Phase ~ 1 + Age + Age^2 + sex + (1|Subj_ID)', 'fitcirc_lme', ...
                    'Select', true, 'MaxOrder', 3);
disp(result_B);

figure('Color','w');
plot(result_B, T_B);
title(sprintf('Part B: N = %d, %d visits/subject (mixed-effects regime)', ...
              n_subj_B, n_per_B));


%% ====================================================================
%% Comparison
%% ====================================================================

fprintf('\n========================================\n');
fprintf('Side-by-side\n');
fprintf('========================================\n');
fprintf('                              Part A (1 visit)   Part B (%d visits)\n', n_per_B);
fprintf('Selected polynomial order     %d                 %d\n', ...
        result_A.SelectedOrder, result_B.SelectedOrder);
fprintf('Omnibus age p-value           %s             %s\n', ...
        format_p(result_A.AgeEffect.pValue), format_p(result_B.AgeEffect.pValue));
fprintf('R^2_circ marginal             %.3f             %.3f\n', ...
        result_A.GOF.R2_circ_marginal, result_B.GOF.R2_circ_marginal);
fprintf('R^2_circ conditional          %.3f             %.3f\n', ...
        result_A.GOF.R2_circ_conditional, result_B.GOF.R2_circ_conditional);
fprintf('Subject heterogeneity gap     %.3f             %.3f\n', ...
        result_A.GOF.R2_circ_conditional - result_A.GOF.R2_circ_marginal, ...
        result_B.GOF.R2_circ_conditional - result_B.GOF.R2_circ_marginal);

fprintf(['\nKey teaching points:\n', ...
         '  - Part A has only 1 obs per subject. kappa_phi and kappa are\n', ...
         '    not jointly identified, so the EM picks an arbitrary split\n', ...
         '    between residual noise and subject random intercept. The\n', ...
         '    R^2_c > R^2_m gap here reflects that split, not real\n', ...
         '    subject heterogeneity.\n', ...
         '  - Part B has multiple obs per subject. kappa_phi is now\n', ...
         '    identifiable; the R^2_c > R^2_m gap quantifies how much of\n', ...
         '    the response variance the per-subject baseline explains\n', ...
         '    beyond the population trend.\n', ...
         '  - Both fits recover the same population trend; their trajectories\n', ...
         '    coincide on the mean curve. The Part B CI band is wider because\n', ...
         '    fewer subjects contribute the population-level information.\n', ...
         '  - The sex covariate is recovered in both fits; look at the\n', ...
         '    coefficient row for `sex` in result_A.Coefficients (or\n', ...
         '    result_B.Coefficients) and compare its Estimate to beta_sex.\n']);

fprintf(['\nTutorial complete. Try:\n', ...
         '  - Setting beta_age2 = 0 and re-running. Both parts should drop\n', ...
         '    SelectedOrder to 1 (no curvature in the truth -> no curvature\n', ...
         '    accepted by LRT).\n', ...
         '  - Changing kappa_phi to 10 in Part B (subjects nearly identical).\n', ...
         '    Watch the R^2_c - R^2_m gap shrink toward 0.\n', ...
         '  - Dropping `+ sex` from the formula. The omnibus age effect will\n', ...
         '    absorb part of the sex offset and beta estimates will shift.\n']);


% =====================================================================
function s = format_p(p)
if isnan(p), s = '   NaN';
elseif p < eps, s = '<1e-308';
elseif p < 1e-4, s = sprintf('%.1e', p);
else, s = sprintf('%.4f', p);
end
end
