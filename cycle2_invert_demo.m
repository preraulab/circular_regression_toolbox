% CYCLE2_INVERT_DEMO  Reproducible single-replicate demonstration of the
% inversion formula eq.~(\ref{eq:invert}) requested by cycle-2 reviewer.
%
% Re-runs the sin/cos LME on a single kappa=20 replicate of the Monte
% Carlo harness (ki=3, r=1: rng(7 + 1000*3 + 1) == rng(3008)), extracts
% the raw (gamma_sin, gamma_cos) pair on the Age coefficient via
% fitlme_circ, and applies the inversion
%
%   |beta_k| = sqrt(gamma_s^2 + gamma_c^2) / (A_1(kappa) * exp(-sigma^2/2))
%
% The cycle-2 reviewer's algebra notes that this recovers
%   |beta_k| * sqrt(mean(cos eta)^2 + mean(sin eta)^2),
% not |beta_k| exactly; we report both the inversion output AND the
% residual factor sqrt(mean_cos^2 + mean_sin^2) explicitly.
%
% Saves /Users/Mike/code/projects/trends_v_individual/mc_cycle2_invert_demo.csv

clear; close all;
addpath(fileparts(mfilename('fullpath')));

repo_root = fullfile(fileparts(mfilename('fullpath')), '..');

% ---------------------------------------------------------------------
% DGP, identical to test_mc_simulation.m at ki=3 (kappa=20), r=1.
% ---------------------------------------------------------------------
kappa_true = 20;
n_subjects = 300;
n_per      = 2;
sigma_phi  = 0.4;
true_b = [ 0.6;  -0.12;  0.0024;  -1.4e-5;  0.30;  -0.10;  0.008;  -4e-5;  0 ];

ki = 3;     % index of kappa=20 in the MC harness
r  = 1;     % first replicate
rng(7 + 1000*ki + r);

subj_age   = 7 + 73 * rand(n_subjects, 1);
subj_sex   = double(rand(n_subjects, 1) > 0.5);
subj_re    = sigma_phi * randn(n_subjects, 1);
subj_id    = repelem((1:n_subjects)', n_per);
Age        = repelem(subj_age, n_per);
sex        = repelem(subj_sex, n_per);
electrode  = repmat([1; 0], n_subjects, 1);
re         = repelem(subj_re, n_per);
n          = length(Age);

X_true = [ones(n,1), Age, Age.^2, Age.^3, electrode, sex, ...
          electrode.*Age, electrode.*Age.^2, electrode.*Age.^3];
eta_true   = X_true * true_b + re;
phase_pref = wrapToPi(eta_true + randn(n,1)/sqrt(kappa_true));

tbl = table(subj_id, Age, sex, electrode, phase_pref, ...
    'VariableNames', {'Subj_ID','Age','sex','electrode','phase_pref'});
tbl.Subj_ID = categorical(tbl.Subj_ID);
tbl.Age2 = tbl.Age.^2;
tbl.Age3 = tbl.Age.^3;

fml = ['phase_pref ~ 1 + Age + Age2 + Age3 + electrode + sex + ' ...
       'Age:electrode + Age2:electrode + Age3:electrode + (1|Subj_ID)'];

% ---------------------------------------------------------------------
% Fit fitlme_circ; extract raw (EstSin, EstCos) for the Age term
% ---------------------------------------------------------------------
mdl = fitlme_circ(tbl, fml);
idx_age = find(strcmp(mdl.CoefficientNames, 'Age'), 1);
assert(~isempty(idx_age), 'Age coefficient not found');

gamma_sin = mdl.Coefficients.EstSin(idx_age);
gamma_cos = mdl.Coefficients.EstCos(idx_age);

% ---------------------------------------------------------------------
% Sample means of cos(eta), sin(eta) on this realized design (no RE)
% so we can compute the residual factor sqrt(cos^2 + sin^2).
% ---------------------------------------------------------------------
eta_no_re   = X_true * true_b;
mean_cos_eta = mean(cos(eta_no_re));
mean_sin_eta = mean(sin(eta_no_re));
residual_factor = sqrt(mean_cos_eta^2 + mean_sin_eta^2);

% ---------------------------------------------------------------------
% Apply the inversion at the true kappa AND at the MC-mean kappa-hat.
% Per the variance-component table, vMEM mean kappa_hat at kappa=20 is
% 21.19; we use that as the practitioner-realistic plug-in.
% ---------------------------------------------------------------------
kappa_used   = 21.19;            % MC-mean of kappa_hat from vMEM at truth=20
A1           = besseli(1, kappa_used) / besseli(0, kappa_used);
exp_factor   = exp(-sigma_phi^2 / 2);
beta_inverted = sqrt(gamma_sin^2 + gamma_cos^2) / (A1 * exp_factor);

% Also compute at kappa = 20 (truth) for comparison
A1_truth        = besseli(1, kappa_true) / besseli(0, kappa_true);
beta_inv_truth  = sqrt(gamma_sin^2 + gamma_cos^2) / (A1_truth * exp_factor);

% ---------------------------------------------------------------------
% Print
% ---------------------------------------------------------------------
fprintf('\n=== Cycle 2 inversion demo (single replicate, ki=3, r=1) ===\n');
fprintf('  rng seed             : 7 + 1000*%d + %d = %d\n', ki, r, 7+1000*ki+r);
fprintf('  kappa_true           : %g\n', kappa_true);
fprintf('  sigma_phi            : %.3f\n', sigma_phi);
fprintf('  truth |beta_Age|     : %.4f\n', abs(true_b(2)));
fprintf('\n  Raw sin/cos LME coefficients on Age:\n');
fprintf('    gamma_sin (EstSin) : %+.5f\n', gamma_sin);
fprintf('    gamma_cos (EstCos) : %+.5f\n', gamma_cos);
fprintf('\n  Sample means of (cos eta, sin eta) on the design:\n');
fprintf('    mean(cos eta)      : %+.4f\n', mean_cos_eta);
fprintf('    mean(sin eta)      : %+.4f\n', mean_sin_eta);
fprintf('    residual factor    : sqrt(%.3f^2 + %.3f^2) = %.4f\n', ...
    mean_cos_eta, mean_sin_eta, residual_factor);
fprintf('\n  Inversion at kappa = %.2f (MC-mean kappa_hat from vMEM):\n', kappa_used);
fprintf('    A1(kappa)            : %.4f\n', A1);
fprintf('    exp(-sigma^2/2)      : %.4f\n', exp_factor);
fprintf('    |beta_inverted|      : %.4f\n', beta_inverted);
fprintf('    expected (truth*rf)  : %.4f\n', abs(true_b(2)) * residual_factor);
fprintf('\n  Inversion at kappa = %g (truth, sanity check):\n', kappa_true);
fprintf('    A1(20)               : %.4f\n', A1_truth);
fprintf('    |beta_inverted|      : %.4f\n', beta_inv_truth);

% ---------------------------------------------------------------------
% Save CSV
% ---------------------------------------------------------------------
csv_path = fullfile(repo_root, 'mc_cycle2_invert_demo.csv');
fid = fopen(csv_path, 'w');
fprintf(fid, 'kappa_used,A1,exp_factor,gamma_sin,gamma_cos,beta_inverted,residual_factor\n');
fprintf(fid, '%.4f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n', ...
    kappa_used, A1, exp_factor, gamma_sin, gamma_cos, ...
    beta_inverted, residual_factor);
% Second row at truth kappa for reference
fprintf(fid, '%.4f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n', ...
    kappa_true, A1_truth, exp_factor, gamma_sin, gamma_cos, ...
    beta_inv_truth, residual_factor);
fclose(fid);
fprintf('\nSaved %s\n', csv_path);
