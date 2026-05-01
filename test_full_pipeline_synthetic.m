%TEST_FULL_PIPELINE_SYNTHETIC  End-to-end smoke test of the lifespan-paper
% stats pipeline on synthetic data, after the drop-in upgrade to:
%   - circular_regression_fixed (cluster-robust SEs)
%   - joint_test (Age-block joint Wald p-value in get_stats_tbl)
%
% Exercises:
%   1. get_feature_group_models('best', ...) -> iterative-LRT order selection
%      using circular_regression_fixed under the hood
%   2. get_feature_group_stats_tbls -> get_stats_tbl writes xlsx with the
%      new "Age block" joint p-value columns
%   3. get_model_fit_manual -> trajectory prediction works on the new mdl
%      structure (cov_b, Coefficients.Row both present)

clear; close all;
addpath(fileparts(mfilename('fullpath')));

% --- Load synthetic data and add the mode_cluster column the pipeline expects ---
T = readtable(fullfile(fileparts(mfilename('fullpath')), '..', ...
                       'test_synthetic_data_N300.csv'));
T.Subj_ID = categorical(T.Subj_ID);
% Lifespan pipeline iterates over modes; assign two synthetic mode_cluster
% groups (split by sex) so we exercise group iteration.
T.mode_cluster = T.sex + 1;       % mode_cluster groups {1, 2}

% --- Pipeline configuration matching SOPH_trajectories.m style ---
features            = {'phase_pref'};
mdl_type            = {'circ'};
is_circ             = logical([1]);   %#ok<NBRAK>
categorical_varnames= {'electrode'};  % keep 'sex' out (it's the grouper)
xcol_int            = false(1,length(categorical_varnames));
x_col               = 'Age';
group_by            = 'mode_cluster';
groups              = [1, 2];

%% --- 1. Build models with iterative-LRT order selection ----------
fprintf('Building models with order = best (iterative LRT)...\n');
mdls = get_feature_group_models('best', T, x_col, group_by, groups, ...
                                features, mdl_type, categorical_varnames, ...
                                xcol_int);
for ii = 1:length(groups)
    for jj = 1:length(features)
        m = mdls{ii,jj};
        fprintf('  group=%d feature=%s -> selected order=%d, LRT p=%.3g\n', ...
            groups(ii), features{jj}, m{2}, m{3});
        mdl = m{1};
        fprintf('    NumCoefficients=%d, R^2=%.3f, kappa=%.3f, DFE=%d\n', ...
            mdl.NumCoefficients, mdl.Rsquared.Ordinary, mdl.Kappa, mdl.DFE);
        fprintf('    CovarianceType=%s\n', mdl.CovarianceType);
    end
end

%% --- 2. Write xlsx stats tables and inspect ----------------------
out_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'synthetic_run_out');
if ~exist(out_dir,'dir'), mkdir(out_dir); end
% Clean any leftover xlsx
delete(fullfile(out_dir,'group_*.xlsx'));

fprintf('\nWriting stats tables to %s...\n', out_dir);
get_feature_group_stats_tbls(out_dir, mdls, features, groups, ...
                             features, groups, mdl_type);

% Read back the per-group sheets to verify the new joint-test columns
for ii = 1:length(groups)
    g     = groups(ii);
    fname = fullfile(out_dir, sprintf('group_%d.xlsx', g));
    fprintf('\n=== %s, sheet=%s ===\n', fname, features{1});
    Ttop  = readtable(fname,'Sheet',features{1},'Range','A1:H2');
    disp(Ttop);
    Tcoef = readtable(fname,'Sheet',features{1},'Range','A4');
    disp(Tcoef);
end

%% --- 3. Trajectory prediction via get_model_fit_manual -----------
fprintf('\nTesting trajectory prediction (get_model_fit_manual)...\n');
x_eval = linspace(min(T.Age), max(T.Age), 50)';
mdl1   = mdls{1,1}{1};
[phat, phat_CI] = get_model_fit_manual(x_eval, x_col, mdl1, 'circ', ...
                                       categorical_varnames, [0]);
fprintf('  group=1, electrode=0 trajectory: %d points, range [%.2f, %.2f] rad\n', ...
    numel(phat), min(phat), max(phat));
fprintf('  CI columns: %s\n', mat2str(size(phat_CI)));

%% --- 4. Cross-check joint test against direct computation --------
fprintf('\nCross-check: joint Age-block test (group=1, phase_pref):\n');
jt = joint_test(mdl1, 'x_main');
fprintf('  Direct joint_test:    F=%.3f, p=%.4g, df=(%d,%d)\n', ...
    jt.Fstat, jt.pValue, jt.df1, jt.df2);
% Read from the xlsx
Tj = readtable(fullfile(out_dir,'group_1.xlsx'),'Sheet',features{1},'Range','A1:H2');
fprintf('  From xlsx (Age block): F=%.3f, p=%.4g, df=(%d,%d)\n', ...
    Tj.AgeBlockFstat, Tj.AgeBlockPValue, Tj.AgeBlockDf1, Tj.AgeBlockDf2);

fprintf('\n*** Full-pipeline synthetic test complete. ***\n');
