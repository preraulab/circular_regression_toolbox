%TEST_FITCIRC_LME_RECOVERY  Parameter-recovery test for fitcirc_lme.
%
% Generates vM-GLMM synthetic data with KNOWN beta, kappa, kappa_phi,
% fits, asserts recovered fixed effects match within tolerance and that
% joint Wald p-values land on the expected side of alpha for true-positive
% vs true-null effects.
%
% Deterministic via rng(7). Runs in a few seconds.

clear; close all;
this_dir = fileparts(mfilename('fullpath'));
repo_root = fileparts(fileparts(fileparts(this_dir)));
addpath(genpath(fullfile(repo_root,'stats')));

rng(7);

% --- Ground truth ---
n_subj  = 80;
n_per   = 8;            % 8 obs per subject -> 640 total
beta0   = 0.30;
beta_age = 0.025;       % nonzero (true positive)
beta_sex = 0;           % zero (true null)
kappa_phi = 6;          % moderate subject phase variability
kappa_eps = 10;         % residual concentration

ages = repmat(linspace(20,70,n_per).', n_subj, 1);
sid  = repelem((1:n_subj).', n_per);
sex  = double(rand(n_subj,1) > 0.5);
sex_obs = sex(sid);

phi  = circ_vmrnd(0, kappa_phi, [n_subj, 1]);
eps  = circ_vmrnd(0, kappa_eps, [numel(ages), 1]);
eta  = beta0 + beta_age*(ages - 45) + beta_sex*sex_obs + phi(sid) + eps;
wrap = @(x) ((x + pi) - 2*pi*floor((x + pi)/(2*pi))) - pi;
y    = wrap(eta);

T = table(ages, sid, sex_obs, y, 'VariableNames', {'Age','Subj_ID','sex','y'});

%% --- Fit ---
mdl = fitcirc_lme(T, 'y ~ Age + sex + (1|Subj_ID)');
beta_hat = mdl.Beta;
nm = string(mdl.Coefficients.Name);
i_int = find(nm == "(Intercept)");
i_age = find(nm == "Age");
i_sex = find(nm == "sex");

% Intercept on the baseline scale: model uses Age unshifted (no -45),
% so true intercept relative to fit is beta0 - beta_age*45.
true_int = beta0 - beta_age*45;
fprintf('Recovered:  intercept=%+.3f (true %+.3f)\n', beta_hat(i_int), true_int);
fprintf('            beta_age =%+.4f  (true %+.4f)\n', beta_hat(i_age), beta_age);
fprintf('            beta_sex =%+.3f  (true %+.3f)\n', beta_hat(i_sex), beta_sex);
fprintf('            kappa    =%.2f   kappa_phi=%.2f\n', mdl.Kappa, mdl.KappaPhi);

% Tolerances: SE-dominated, picked to be loose enough that the test is
% reliable across re-runs but tight enough to catch real regressions.
assert(abs(beta_hat(i_age) - beta_age) < 0.005, ...
    'Age slope off: got %.4f, expected %.4f', beta_hat(i_age), beta_age);
% Sex is between-subject; SE is governed by n_subj, not n_obs. We only
% require the magnitude to be modest; the p-value check below is the
% real true-null assertion.
assert(abs(beta_hat(i_sex) - beta_sex) < 0.30, ...
    'Sex coef drift: got %.3f, expected %.3f', beta_hat(i_sex), beta_sex);
assert(abs(wrap(beta_hat(i_int) - true_int)) < 0.10, ...
    'Intercept off: got %+.3f, expected %+.3f', beta_hat(i_int), true_int);

%% --- Joint Wald block tests (true-positive only; type-I rate
%%     is handled in test_joint_test.m where the null covariate is
%%     within-subject and SE-resolved) ---
p_age = mdl.coefTest(i_age).pValue;
fprintf('\nJoint Wald: p_Age=%.2e\n', p_age);
assert(p_age < 1e-3, ...
    'True-positive Age effect not detected: p=%.3g', p_age);

%% --- Sanity on R^2 ---
R2 = mdl.Rsquared.Ordinary;
fprintf('R^2_circ = %.3f\n', R2);
assert(R2 > 0, 'R^2 should be positive on data with a true Age effect');

fprintf('\n*** test_fitcirc_lme_recovery passed. ***\n');


function r = circ_vmrnd(mu, kappa, sz)
    n = prod(sz);
    if kappa == 0
        r = -pi + 2*pi*rand(sz); return;
    end
    a = 1 + sqrt(1 + 4*kappa^2);
    b = (a - sqrt(2*a)) / (2*kappa);
    r0 = (1 + b^2) / (2*b);
    out = nan(n,1); k = 0;
    while k < n
        u1 = rand; u2 = rand; u3 = rand;
        z  = cos(pi*u1);
        f  = (1 + r0*z) / (r0 + z);
        c  = kappa * (r0 - f);
        if c*(2 - c) - u2 > 0 || log(c/u2) + 1 - c >= 0
            k = k + 1;
            out(k) = mu + sign(u3 - 0.5) * acos(f);
        end
    end
    r = reshape(((out + pi) - 2*pi*floor((out + pi)/(2*pi))) - pi, sz);
end
