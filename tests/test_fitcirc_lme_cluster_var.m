%TEST_FITCIRC_LME_CLUSTER_VAR  Robust standard errors clustered on a nesting variable.
%
% Subjects are nested in families, each family shares a phase offset, and
% the predictor `f` is constant within a family. Checks that:
%   1. ClusterVar naming the subject column reproduces the default fit.
%   2. Clustering on family counts families as clusters (NumClusters, DFE)
%      and widens the SE of the family-level predictor, since subjects of a
%      family are correlated beyond their own random offsets.
%   3. A cluster variable that does not nest the subjects is rejected.
%
% Deterministic via rng(11). Runs in a few seconds.

clear; close all;
this_dir = fileparts(mfilename('fullpath'));
addpath(fileparts(this_dir));

rng(11);
n_fam     = 60;
subj_fam  = repelem((1:n_fam)', randi([2 4], n_fam, 1));   % 2-4 subjects per family
n_subj    = numel(subj_fam);
n_per     = 4;
sid       = repelem((1:n_subj)', n_per);
fid       = subj_fam(sid);
f_fam     = randn(n_fam, 1);                                 % family-level predictor
f         = f_fam(fid);
fam_off   = circ_vmrnd(0, 3, [n_fam, 1]);                    % shared family offset
subj_off  = circ_vmrnd(0, 8, [n_subj, 1]);
eps       = circ_vmrnd(0, 6, [numel(sid), 1]);
wrap      = @(x) ((x + pi) - 2*pi*floor((x + pi)/(2*pi))) - pi;
y         = wrap(0.2 + 0.3*f + fam_off(fid) + subj_off(sid) + eps);
T = table(y, f, sid, fid, 'VariableNames', {'y', 'f', 'Subj_ID', 'Fam_ID'});

fml   = 'y ~ f + (1|Subj_ID)';
m_def = fitcirc_lme(T, fml);
m_sub = fitcirc_lme(T, fml, 'ClusterVar', 'Subj_ID');
m_fam = fitcirc_lme(T, fml, 'ClusterVar', 'Fam_ID');

%% 1. Subject clustering is the default
assert(max(abs(m_def.cov_b(:) - m_sub.cov_b(:))) < 1e-12, 'ClusterVar = Subj_ID differs from the default');
assert(m_def.NumClusters == n_subj && m_def.DFE == n_subj - 1, 'default clusters should be subjects');

%% 2. Family clustering
assert(m_fam.NumClusters == n_fam, 'NumClusters %d, expected %d families', m_fam.NumClusters, n_fam);
assert(m_fam.DFE == n_fam - 1, 'DFE %d, expected %d', m_fam.DFE, n_fam - 1);
assert(max(abs(m_fam.Beta - m_def.Beta)) < 1e-12, 'clustering must not change the estimates');
i_f    = strcmp(m_def.Coefficients.Name, 'f');
se_sub = m_def.Coefficients.SE(i_f);
se_fam = m_fam.Coefficients.SE(i_f);
fprintf('SE(f): subject clusters %.4f, family clusters %.4f (ratio %.2f)\n', se_sub, se_fam, se_fam / se_sub);
assert(se_fam > 1.1 * se_sub, 'family clustering should widen the SE of a family-level predictor');

%% 3. Non-nested cluster variable
T.Bad = mod(T.Subj_ID + (1:height(T))', 7);    % varies within subject
try
    fitcirc_lme(T, fml, 'ClusterVar', 'Bad');
    error('test:noError', 'a non-nesting cluster variable was accepted');
catch err
    assert(strcmp(err.identifier, 'fitcirc_lme:clusterNotNested'), 'unexpected error: %s', err.message);
end

fprintf('test_fitcirc_lme_cluster_var: PASS\n');
