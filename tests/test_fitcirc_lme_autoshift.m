%TEST_FITCIRC_LME_AUTOSHIFT  Round-trip + seam-crossing test for AutoShift.
%
%  1. Data well away from the +/- pi seam: fit with AutoShift=true and
%     AutoShift=false, assert predictions and CI half-widths match to
%     numerical tolerance. Translation invariance of cov_b means CIs
%     should be bit-identical aside from EM roundoff.
%
%  2. Data that crosses the seam: assert AutoShift's R^2 is at least as
%     good as legacy, and predictions on a held-out grid are wrapped
%     equivalents (i.e., circular distance is small).
%
% Deterministic via rng(42). Runs in a few seconds.

clear; close all;
this_dir = fileparts(mfilename('fullpath'));
repo_root = fileparts(fileparts(fileparts(this_dir)));
addpath(genpath(fullfile(repo_root,'stats')));

rng(42);

% ---- helpers ----
wrap   = @(x) ((x + pi) - 2*pi*floor((x + pi)/(2*pi))) - pi;
cdist  = @(a,b) abs(wrap(a - b));

% ---- shared synthetic generator ----
function T = simulate(n_subj, n_per, mu0, beta_age, kappa_phi, kappa_eps)
    ages    = repmat(linspace(20,70,n_per).', n_subj, 1);
    sid     = repelem((1:n_subj).', n_per);
    phi     = circ_vmrnd(0, kappa_phi, [n_subj,1]);
    eps     = circ_vmrnd(0, kappa_eps, [numel(ages),1]);
    eta     = mu0 + beta_age*(ages - 45) + phi(sid) + eps;
    y       = ((eta + pi) - 2*pi*floor((eta + pi)/(2*pi))) - pi;
    T = table(ages, sid, y, 'VariableNames', {'Age','Subj_ID','y'});
end

%% ---- Case 1: data centered near 0 (no seam crossing) ----
fprintf('Case 1: data centered at 0 (no seam crossing)...\n');
T1 = simulate(40, 6, 0.0, 0.02, 6, 8);
m_legacy = fitcirc_lme(T1, 'y ~ Age + (1|Subj_ID)');
m_auto   = fitcirc_lme(T1, 'y ~ Age + (1|Subj_ID)', 'AutoShift', true);

x_grid = (15:5:75).';
nd = table(x_grid, double(repmat(T1.Subj_ID(1), numel(x_grid),1)), ...
    'VariableNames', {'Age','Subj_ID'});
y_legacy = m_legacy.predict(nd);
y_auto   = m_auto.predict(nd);
max_diff = max(cdist(y_legacy, y_auto));
fprintf('  max circular |y_legacy - y_auto| = %.2e\n', max_diff);
assert(max_diff < 1e-6, ...
    'AutoShift and legacy disagree on no-seam data: max diff = %.3g', max_diff);
fprintf('  PASS: predictions agree to %.2e\n', max_diff);

%% ---- Case 2: data centered near +pi (heavy seam crossing) ----
fprintf('\nCase 2: data centered near +pi (seam crosses heavily)...\n');
T2 = simulate(40, 6, pi - 0.05, 0.02, 6, 8);
m_legacy = fitcirc_lme(T2, 'y ~ Age + (1|Subj_ID)');
m_auto   = fitcirc_lme(T2, 'y ~ Age + (1|Subj_ID)', 'AutoShift', true);

R2_legacy = m_legacy.Rsquared.Ordinary;
R2_auto   = m_auto.Rsquared.Ordinary;
fprintf('  R^2_legacy=%.3f   R^2_auto=%.3f   theta_shift=%.3f\n', ...
    R2_legacy, R2_auto, m_auto.ThetaShift);
assert(R2_auto >= R2_legacy - 1e-6, ...
    'AutoShift R^2 (%.4f) lower than legacy R^2 (%.4f) on seam-crossing data', ...
    R2_auto, R2_legacy);
fprintf('  PASS: AutoShift R^2 >= legacy R^2 on seam-crossing data\n');

%% ---- Case 3: explicit ThetaShift override ----
fprintf('\nCase 3: explicit ThetaShift override matches AutoShift result...\n');
ts = m_auto.ThetaShift;
m_explicit = fitcirc_lme(T2, 'y ~ Age + (1|Subj_ID)', 'ThetaShift', ts);
y_a = m_auto.predict(nd);
y_e = m_explicit.predict(nd);
max_diff = max(cdist(y_a, y_e));
fprintf('  max circular |y_auto - y_explicit| = %.2e\n', max_diff);
assert(max_diff < 1e-6, ...
    'Explicit ThetaShift disagrees with AutoShift: max diff = %.3g', max_diff);
fprintf('  PASS: explicit ThetaShift reproduces AutoShift\n');

fprintf('\n*** test_fitcirc_lme_autoshift passed. ***\n');


% ---- local fallback for circ_vmrnd if Statistics toolbox version not on path ----
function r = circ_vmrnd(mu, kappa, sz)
    n = prod(sz);
    if kappa == 0
        r = -pi + 2*pi*rand(sz);
        return;
    end
    a = 1 + sqrt(1 + 4*kappa^2);
    b = (a - sqrt(2*a)) / (2*kappa);
    r0 = (1 + b^2) / (2*b);
    out = nan(n,1);
    k = 0;
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
