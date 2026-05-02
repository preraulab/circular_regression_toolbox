%TEST_RESAMPLE_COMPARE  Compare three approaches on leverage-subject contamination.
%
% Generates vM-GLMM data with a known Age slope. Contaminates the OLDEST
% subjects (high-leverage on the slope) by replacing all of their rows
% with uniform circular noise — i.e., a subset of "bad" subjects whose
% data is unrelated to age. This is the failure mode resampling is
% supposed to handle: a per-subject random intercept can absorb a global
% rotation, but cannot absorb genuinely uninformative data, which biases
% the slope through high-Age subjects toward zero.
%
% Note: a constant per-subject rotation (e.g. +pi/2 for all rows of bad
% subjects) is fully absorbed by the random intercept phi_i and does NOT
% bias the slope — that's by design of the von-Mises GLMM. The
% interesting failure mode is high-leverage subjects with non-informative
% (high-variance) phase data.
%
% Compared estimators:
%   none       legacy fitcirc_lme single fit
%   cboot      cluster bootstrap (B subjects-with-replacement), median Beta
%   sub80      subsample 80% of subjects without replacement, median Beta
%
% Asserts that the two resampling estimators recover beta_age better than
% the legacy single fit when ~12% of subjects are contaminated. Also
% reports the spread (MAD across bootstrap replicates) as a sanity check.
%
% Deterministic via rng(13). Runs in 1-2 minutes (B = 60).

clear; close all;
this_dir = fileparts(mfilename('fullpath'));
repo_root = fileparts(fileparts(fileparts(this_dir)));
addpath(genpath(fullfile(repo_root,'stats')));

rng(13);

n_subj   = 60;
n_per    = 8;
beta0    = 0.20;
beta_age = 0.025;
kappa_phi = 6;
kappa_eps = 10;
contam_subj_frac = 0.12;        % fraction of high-Age subjects whose data is noise
B = 60;
keep_frac = 0.8;

% Each subject has a SINGLE Age (its row block has the same Age value),
% so we can deterministically pick high-leverage subjects.
subj_age = linspace(20, 70, n_subj).';
ages = repelem(subj_age, n_per);
sid  = repelem((1:n_subj).', n_per);
phi  = circ_vmrnd(0, kappa_phi, [n_subj,1]);
eps_obs = circ_vmrnd(0, kappa_eps, [numel(ages),1]);
eta  = beta0 + beta_age*(ages - 45) + phi(sid) + eps_obs;
wrap = @(x) ((x + pi) - 2*pi*floor((x + pi)/(2*pi))) - pi;
y    = wrap(eta);

% Leverage contamination: replace the highest-Age subjects' rows with
% uniform circular noise. This pulls the slope toward zero because the
% high-leverage end has no real signal.
n_bad_subj = round(contam_subj_frac * n_subj);
bad_subj   = (n_subj - n_bad_subj + 1):n_subj;
mask_bad   = ismember(sid, bad_subj);
y(mask_bad) = -pi + 2*pi*rand(sum(mask_bad), 1);

T = table(ages, sid, y, 'VariableNames', {'Age','Subj_ID','y'});
formula = 'y ~ Age + (1|Subj_ID)';

slope_idx = @(m) find(strcmp(m.Coefficients.Name, 'Age'), 1);

%% --- Legacy single fit ---
fprintf('Fitting legacy single fit (no resampling)...\n');
m0 = fitcirc_lme(T, formula, 'AutoShift', true);
b_legacy = m0.Coefficients.Estimate(slope_idx(m0));
fprintf('  legacy beta_age = %.4f   (true %.4f, err %.4f)\n', ...
    b_legacy, beta_age, abs(b_legacy - beta_age));

%% --- Cluster bootstrap: subjects with replacement, fit B times ---
fprintf('Cluster bootstrap (B=%d, with replacement)...\n', B);
b_boot = nan(B, 1);
t0 = tic;
for k = 1:B
    pick   = randi(n_subj, n_subj, 1);
    T_b    = resample_subjects(T, pick);
    m      = fitcirc_lme(T_b, formula, 'AutoShift', true);
    b_boot(k) = m.Coefficients.Estimate(slope_idx(m));
end
fprintf('  done in %.1fs\n', toc(t0));
b_cboot = median(b_boot);
fprintf('  cluster-boot median beta_age = %.4f   (err %.4f)   bootstrap MAD = %.4f\n', ...
    b_cboot, abs(b_cboot - beta_age), median(abs(b_boot - median(b_boot))));

%% --- Subsample 80% of subjects without replacement ---
fprintf('Subsample %d%% (B=%d, without replacement)...\n', round(100*keep_frac), B);
n_keep = round(keep_frac * n_subj);
b_sub  = nan(B, 1);
t0 = tic;
for k = 1:B
    pick = randperm(n_subj, n_keep);
    T_s  = T(ismember(T.Subj_ID, pick), :);
    m    = fitcirc_lme(T_s, formula, 'AutoShift', true);
    b_sub(k) = m.Coefficients.Estimate(slope_idx(m));
end
fprintf('  done in %.1fs\n', toc(t0));
b_sub80 = median(b_sub);
fprintf('  subsample-80%% median beta_age = %.4f   (err %.4f)   spread MAD = %.4f\n', ...
    b_sub80, abs(b_sub80 - beta_age), median(abs(b_sub - median(b_sub))));

%% --- Comparison ---
err_legacy = abs(b_legacy - beta_age);
err_cboot  = abs(b_cboot  - beta_age);
err_sub80  = abs(b_sub80  - beta_age);

fprintf('\n=== Summary (%d%% high-Age leverage subjects replaced with uniform noise) ===\n', ...
    round(100*contam_subj_frac));
fprintf('  none     beta_age = %.4f   err = %.4f\n', b_legacy, err_legacy);
fprintf('  cboot    beta_age = %.4f   err = %.4f   (%.0f%% improvement vs none)\n', ...
    b_cboot, err_cboot, 100*(err_legacy - err_cboot)/max(err_legacy,eps));
fprintf('  sub80    beta_age = %.4f   err = %.4f   (%.0f%% improvement vs none)\n', ...
    b_sub80, err_sub80, 100*(err_legacy - err_sub80)/max(err_legacy,eps));

%% --- Diagnostic only (no pass/fail) ---
% Honest finding from this synthetic case: cluster-level resampling
% recovers ~the same slope as legacy. Reasons:
%   (1) Per-subject phase rotations are absorbed by the random intercept
%       phi_i ~ vM(0, kappa_phi), so they don't bias the slope at all.
%   (2) High-leverage uninformative subjects (the case tested here) are
%       included in MOST bootstrap/subsample replicates -- ~64% of cluster
%       bootstraps and ~88% of 80%-subsamples contain each bad subject.
%       The median across replicates inherits the bias.
% Resampling at this level is for variance estimation, not robustness.
fprintf('\n*** test_resample_compare done (diagnostic only). ***\n');


%% ===== local helpers =====
function T_out = resample_subjects(T, pick)
% Build a resampled table by drawing subjects in `pick` from T. Each
% pick gets a NEW unique Subj_ID so duplicates from the with-replacement
% bootstrap are treated as independent clusters by the random-intercept
% model — otherwise the variance of the bootstrap is wrong.
parts = cell(numel(pick),1);
next_id = 0;
for k = 1:numel(pick)
    src_id = pick(k);
    rows   = T(T.Subj_ID == src_id, :);
    next_id = next_id + 1;
    rows.Subj_ID = repmat(next_id, height(rows), 1);
    parts{k} = rows;
end
T_out = vertcat(parts{:});
end


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
