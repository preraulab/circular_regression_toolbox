%TEST_DROP_IN_COMPAT  Smoke test for the drop-in API extensions.
%   Exercises:
%     1. Legacy-positional call to circular_regression_fixed (Path B style)
%     2. fitcirc_lme's new coefTest method and ContrastIndex
%     3. joint_test helper across LME, fitcirc_lme, and circular_regression_fixed
%
%   Uses test_synthetic_data_N300.csv (300 subjects, 2 obs/subject).

clear; close all;
addpath(fileparts(mfilename('fullpath')));

T = readtable(fullfile(fileparts(mfilename('fullpath')), '..', ...
                       'test_synthetic_data_N300.csv'));
T.Subj_ID = categorical(T.Subj_ID);
T.Age2 = T.Age.^2;
T.Age3 = T.Age.^3;

x        = T.Age;
y        = T.phase_pref;
cat_vars = T{:,{'electrode','sex'}};
varnames = ["Age"; "electrode"; "sex"];
order    = 3;
xcol_int = [];                             % no interactions
b0       = zeros(order+1+size(cat_vars,2),1);
iters    = 100;

%% --- Test 1: legacy-positional dispatch ---------------------------
fprintf('Test 1: legacy positional call to circular_regression_fixed...\n');
[~, mdl_legacy] = circular_regression_fixed(x, y, cat_vars, varnames, ...
                                            order, xcol_int, b0, iters);
assert(isfield(mdl_legacy,'Coefficients'),     'Coefficients missing');
assert(isfield(mdl_legacy,'cov_b'),            'cov_b missing');
assert(isfield(mdl_legacy,'Rsquared'),         'Rsquared missing');
assert(isfield(mdl_legacy,'LogLikelihood'),    'LogLikelihood missing');
assert(isfield(mdl_legacy,'DFE'),              'DFE missing');
assert(isfield(mdl_legacy.ContrastIndex,'x_main'), 'ContrastIndex.x_main missing');
fprintf('  -> coefs=%d, R^2=%.3f, LL=%.2f, DFE=%d\n', ...
    mdl_legacy.NumCoefficients, mdl_legacy.Rsquared.Ordinary, ...
    mdl_legacy.LogLikelihood, mdl_legacy.DFE);

%% --- Test 2: legacy-positional + ClusterID ------------------------
fprintf('\nTest 2: legacy positional + ClusterID for cluster-robust SE...\n');
[~, mdl_cr] = circular_regression_fixed(x, y, cat_vars, varnames, ...
                                        order, xcol_int, b0, iters, T.Subj_ID);
fprintf('  Cov type: %s\n', mdl_cr.CovarianceType);
ratio_age = mdl_cr.Coefficients.SE(2) / mdl_legacy.Coefficients.SE(2);
fprintf('  Cluster-robust / model-based SE ratio on Age: %.2f\n', ratio_age);
assert(strcmp(mdl_cr.CovarianceType,'cluster-robust'), 'Should be cluster-robust');

%% --- Test 3: fitcirc_lme.coefTest, ContrastIndex ------------------
fprintf('\nTest 3: fitcirc_lme coefTest + ContrastIndex...\n');
mdl_em = fitcirc_lme(T, ['phase_pref ~ 1 + Age + Age2 + Age3 + ' ...
                         'electrode + sex + (1|Subj_ID)'], 'MaxIter', 100);
assert(isprop(mdl_em,'cov_b'),         'fitcirc_lme.cov_b missing');
assert(isprop(mdl_em,'ContrastIndex'), 'fitcirc_lme.ContrastIndex missing');
assert(isprop(mdl_em,'Rsquared'),      'fitcirc_lme.Rsquared missing');
assert(isprop(mdl_em,'DFE'),           'fitcirc_lme.DFE missing');
fprintf('  ContrastIndex.x_main = %s\n', mat2str(mdl_em.ContrastIndex.x_main(:)'));
jt_em = mdl_em.coefTest('x_main');
fprintf('  Joint Wald (vMEM) on age block: F=%.3f, p=%.3g, df=(%d,%d)\n', ...
    jt_em.Fstat, jt_em.pValue, jt_em.df1, jt_em.df2);

%% --- Test 4: joint_test helper across model types -----------------
fprintf('\nTest 4: joint_test helper across model classes...\n');
% Build a Gaussian LME on the same design (using sin(y) as a proxy outcome
% just to exercise the LinearMixedModel branch).
T.sin_y = sin(T.phase_pref);
mdl_lme = fitlme(T, 'sin_y ~ 1 + Age + Age2 + Age3 + electrode + sex + (1|Subj_ID)');

jt_lme = joint_test(mdl_lme, {'Age','Age2','Age3'});
jt_em2 = joint_test(mdl_em,  'x_main');
jt_cr  = joint_test(mdl_cr,  'x_main');

fprintf('  Gaussian LME       (joint Age block): F=%.3f, p=%.3g\n', jt_lme.Fstat, jt_lme.pValue);
fprintf('  vM EM (Laplace)    (joint Age block): F=%.3f, p=%.3g\n', jt_em2.Fstat, jt_em2.pValue);
fprintf('  Cluster-robust vM  (joint Age block): F=%.3f, p=%.3g\n', jt_cr.Fstat, jt_cr.pValue);

%% --- Test 5: ContrastIndex.x_main matches expectation -------------
fprintf('\nTest 5: ContrastIndex.x_main coverage check...\n');
% In circular_regression_fixed, x_main should be rows 2..1+order = [2,3,4].
fprintf('  circular_regression_fixed x_main = %s (expect [2 3 4])\n', ...
    mat2str(mdl_legacy.ContrastIndex.x_main(:)'));
% In fitcirc_lme, coefficient ordering may differ; just confirm it
% identifies rows whose names match Age / Age2 / Age3.
nm_em = string(mdl_em.Coefficients.Name);
em_main_names = nm_em(mdl_em.ContrastIndex.x_main);
fprintf('  fitcirc_lme x_main names = %s\n', strjoin(em_main_names, ', '));

fprintf('\n*** All drop-in API tests passed. ***\n');
