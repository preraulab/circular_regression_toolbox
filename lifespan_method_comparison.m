%LIFESPAN_METHOD_COMPARISON  Side-by-side comparison of three circular
% regression methods on a representative pref_phase-style dataset, to
% inform whether to refactor get_feature_fits.m for the lifespan paper.
%
% Methods compared:
%   1. Current pipeline: circular_regression (no RE, non-cluster SE)
%   2. vM EM with Laplace correction: fitcirc_lme (RE, EM sandwich)
%   3. Cluster-robust vM MLE: circular_regression_fixed + ClusterID
%
% Reports per-coefficient: Estimate, SE, p-value side-by-side.

clear; close all;
addpath(fileparts(mfilename('fullpath')));

%% Load data
T = readtable(fullfile(fileparts(mfilename('fullpath')), '..', ...
                       'test_synthetic_data_N300.csv'));
T.Subj_ID = categorical(T.Subj_ID);
fprintf('Loaded %d obs, %d subjects, n_j=%g\n', height(T), ...
    numel(unique(T.Subj_ID)), height(T)/numel(unique(T.Subj_ID)));

order = 3;
x_col = 'Age';
y_col = 'phase_pref';
cat_names = {'electrode','sex'};

%% --- Method 1: current pipeline -------------------------------------
fprintf('\n=== Method 1: circular_regression (current pipeline) ===\n');
x  = T.(x_col);
y  = T.(y_col);
cat_vars = T{:,cat_names};
% xcol_categorical_interactions: empty = no interactions (matches the
% methodology paper's "no interactions" baseline). The varnames must
% include the predictor name first, then categorical names.
varnames_full = [string(x_col); string(cat_names(:))];
[~, mdl1] = circular_regression(x, y, cat_vars, varnames_full, order, ...
                                [], ...
                                zeros(order+1+length(cat_names),1), 1000);
% Pull the coefficient table from mdl1
if istable(mdl1)
    M1 = mdl1;
elseif isstruct(mdl1) && isfield(mdl1,'Coefficients') && istable(mdl1.Coefficients)
    M1 = mdl1.Coefficients;
else
    % fallback: build from named fields
    M1 = table(string(mdl1.coeffnames(:)), mdl1.parameters(:), ...
               nan(numel(mdl1.parameters),1), mdl1.t_stats(:), mdl1.p_values(:), ...
               'VariableNames', {'Name','Estimate','SE','tStat','pValue'});
end
disp(M1);

%% --- Method 2: fitcirc_lme (vM EM with Laplace) ---------------------
fprintf('\n=== Method 2: fitcirc_lme (vM EM, Laplace, sandwich SE) ===\n');
T.Age2 = T.Age.^2;
T.Age3 = T.Age.^3;
fml2 = ['phase_pref ~ 1 + Age + Age2 + Age3 + electrode + sex + ' ...
        '(1|Subj_ID)'];
mdl2 = fitcirc_lme(T, fml2, 'MaxIter', 100);
T2 = mdl2.Coefficients(:, {'Name','Estimate','SE','tStat','pValue'});
T2.Properties.VariableNames{1} = 'Coeff';
disp(T2);
fprintf('  kappa_hat = %.3f, sigma_phi_hat = %.3f\n', mdl2.Kappa, mdl2.SigmaPhi);

%% --- Method 3: cluster-robust vM MLE --------------------------------
fprintf('\n=== Method 3: circular_regression_fixed (cluster-robust) ===\n');
[~, mdl3] = circular_regression_fixed(x, y, ...
    'Order', order, ...
    'Categorical', cat_vars, ...
    'CategoricalNames', cat_names, ...
    'PredictorName', 'Age', ...
    'ClusterID', T.Subj_ID);
T3 = mdl3.Coefficients;
if ismember('Name', T3.Properties.VariableNames)
    T3 = T3(:, {'Name','Estimate','SE','tStat','pValue'});
else
    T3 = addvars(T3(:, {'Estimate','SE','tStat','pValue'}), ...
                 string(T3.Properties.RowNames), 'Before', 1, ...
                 'NewVariableNames', {'Name'});
end
disp(T3);
fprintf('  kappa_hat (no-RE; conflates within/between) = %.3f\n', mdl3.Kappa);

%% --- Side-by-side merged comparison ---------------------------------
fprintf('\n\n=== SIDE-BY-SIDE: Estimate / SE / p-value across methods ===\n');

% Normalize names: method 1 uses 'Age^2','Age^3', methods 2+3 use 'Age2','Age3'
norm_name = @(s) regexprep(string(s), {'\^2','\^3'}, {'2','3'});

% Build aligned rows for the polynomial coefs only
target_terms = {'Intercept','Age','Age2','Age3','electrode','sex'};

% Helper: get coefficient names from each model
n1_list = string(M1.Properties.RowNames);
if isempty(n1_list), n1_list = string(M1.Name); end
n2_list = string(mdl2.Coefficients.Name);
if ismember('Name', mdl3.Coefficients.Properties.VariableNames)
    n3_list = string(mdl3.Coefficients.Name);
else
    n3_list = string(mdl3.Coefficients.Properties.RowNames);
end

rows = cell(0,10);
for k = 1:length(target_terms)
    t = target_terms{k};
    n1 = norm_name(n1_list);
    n2 = norm_name(n2_list);
    n3 = norm_name(n3_list);
    i1 = find(strcmp(n1, t) | (strcmp(t,'Intercept') & contains(n1_list,"Inter")), 1);
    i2 = find(strcmp(n2, t) | (strcmp(t,'Intercept') & contains(n2_list,"Inter")), 1);
    i3 = find(strcmp(n3, t) | (strcmp(t,'Intercept') & contains(n3_list,"Inter")), 1);

    e1=nan; s1=nan; p1_=nan; e2=nan; s2=nan; p2_=nan; e3=nan; s3=nan; p3_=nan;
    if ~isempty(i1), e1=M1.Estimate(i1); s1=M1.SE(i1); p1_=M1.pValue(i1); end
    if ~isempty(i2), e2=mdl2.Coefficients.Estimate(i2); s2=mdl2.Coefficients.SE(i2); p2_=mdl2.Coefficients.pValue(i2); end
    if ~isempty(i3), e3=mdl3.Coefficients.Estimate(i3); s3=mdl3.Coefficients.SE(i3); p3_=mdl3.Coefficients.pValue(i3); end

    rows(end+1,:) = {t, e1, s1, p1_, e2, s2, p2_, e3, s3, p3_}; %#ok<SAGROW>
end
Tcmp = cell2table(rows, 'VariableNames', ...
    {'Term','Est_M1','SE_M1','p_M1','Est_M2','SE_M2','p_M2','Est_M3','SE_M3','p_M3'});
disp(Tcmp);

% Summary: which terms change significance status across methods?
sig = @(p) p < 0.05;
flips = false(height(Tcmp),1);
for k = 1:height(Tcmp)
    sigs = [sig(Tcmp.p_M1(k)), sig(Tcmp.p_M2(k)), sig(Tcmp.p_M3(k))];
    flips(k) = any(sigs ~= sigs(1)) && all(~isnan(sigs));
end
if any(flips)
    fprintf('\nTerms with significance flips across methods:\n');
    disp(Tcmp(flips, {'Term','p_M1','p_M2','p_M3'}));
else
    fprintf('\nNo significance flips across methods at alpha=0.05.\n');
end

% p-value ratio: M3/M1 (cluster-robust vs current)
fprintf('\np-value ratio (cluster-robust / current) per coef:\n');
ratio = Tcmp.p_M3 ./ Tcmp.p_M1;
for k = 1:height(Tcmp)
    fprintf('  %-10s  p_current=%.4g  p_clRob=%.4g  ratio=%.2f\n', ...
        char(Tcmp.Term(k)), Tcmp.p_M1(k), Tcmp.p_M3(k), ratio(k));
end

% Save
out = fullfile(fileparts(mfilename('fullpath')), '..', ...
               'lifespan_method_comparison.csv');
writetable(Tcmp, out);
fprintf('\nSaved %s\n', out);
