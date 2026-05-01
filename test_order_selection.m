%TEST_ORDER_SELECTION  Compare polynomial-order selection across methods.
%
% Generates synthetic circular data with known cubic age trajectory
% (truth: nonzero linear, quadratic, and cubic age main effects + nonzero
% linear and quadratic electrode-by-age interactions; cubic interaction
% is zero), then runs the iterative LRT order-selection procedure used
% by the paper (get_best_iterative_order.m) for four MATLAB-side methods:
%
%   1. circular_regression.m      (original)
%   2. circular_regression_fixed.m (cluster-robust)
%   3. fitlme on sin(phase_pref)
%   4. fitlme on cos(phase_pref)
%
% For each method, fits at orders 0..4 and runs the iterative LRT:
%   accept order k+1 over k iff p(LRT) < 0.05, else stop.
%
% Truth -> correct selected order is 3.
%
% Also exports the synthetic data to test_synthetic_data_N300.csv so the
% R counterpart (test_order_selection.R) can use the same observations.

clear; close all;
addpath(fileparts(mfilename('fullpath')));
rng(42);

% =====================================================================
% Synthetic data, N = 300 subjects (600 observations)
% =====================================================================
n_subjects = 300;
subj_age   = 7 + 73 * rand(n_subjects, 1);
subj_sex   = double(rand(n_subjects, 1) > 0.5);
subj_re_sd = 0.4;
subj_re    = subj_re_sd * randn(n_subjects, 1);

subj_id   = repelem((1:n_subjects)', 2);
Age       = repelem(subj_age, 2);
sex       = repelem(subj_sex, 2);
electrode = repmat([1; 0], n_subjects, 1);
re        = repelem(subj_re, 2);
n         = length(Age);

true_b = [ 0.6;  -0.06;  0.0012;  -7e-6;  0.30;  -0.10;  0.004;  -2e-5;  0 ];
X_true = [ones(n,1), Age, Age.^2, Age.^3, electrode, sex, ...
          electrode.*Age, electrode.*Age.^2, electrode.*Age.^3];
true_phase = X_true * true_b + re;

kappa_true = 4;
phase_pref = wrapToPi(true_phase + randn(n,1)/sqrt(kappa_true));

tbl = table(subj_id, Age, sex, electrode, phase_pref, ...
            'VariableNames', {'Subj_ID','Age','sex','electrode','phase_pref'});
tbl.sin_phase = sin(tbl.phase_pref);
tbl.cos_phase = cos(tbl.phase_pref);
tbl.Age2 = tbl.Age.^2;
tbl.Age3 = tbl.Age.^3;
tbl.Age4 = tbl.Age.^4;

writetable(tbl, fullfile(fileparts(mfilename('fullpath')), '..', ...
                         'test_synthetic_data_N300.csv'));

fprintf('Synthetic data: %d subjects x 2 electrodes = %d observations\n', ...
        n_subjects, n);
fprintf('Truth: cubic age main effect + quadratic electrode:age interaction\n');
fprintf('       => correct selected order = 3\n\n');

% =====================================================================
% Fit each method at each order; collect log-likelihoods + npar
% =====================================================================
orders        = 0:4;
nO            = length(orders);
LL_orig       = nan(1, nO);
LL_fixed      = nan(1, nO);
LL_sin        = nan(1, nO);
LL_cos        = nan(1, nO);
np_circ       = nan(1, nO);   % npar of circular models (incl. intercept)
np_sincos     = nan(1, nO);   % npar of each sin/cos LME (single component)

cat_vars   = [tbl.electrode, tbl.sex];
varnames   = {'phase_pref','Age','electrode','sex'};
intx_full  = [true, false];

fprintf('=== Fitting all methods at each polynomial order ===\n\n');
for ii = 1:nO
    k = orders(ii);
    if k == 0
        intx_k = [false, false];
    else
        intx_k = intx_full;
    end

    % --- Original circular_regression ---
    np = 1 + k + size(cat_vars,2) + sum(intx_k)*k;
    b0 = zeros(np, 1);
    b0(1) = atan2(mean(sin(tbl.phase_pref)), mean(cos(tbl.phase_pref)));
    [~, mdl_o] = circular_regression(tbl.Age, tbl.phase_pref, cat_vars, ...
        varnames(2:end), k, intx_k, b0, 1000);
    LL_orig(ii) = mdl_o.LogLikelihood;
    np_circ(ii) = mdl_o.NumCoefficients;

    % --- Fixed circular_regression_fixed ---
    [~, mdl_f] = circular_regression_fixed(tbl.Age, tbl.phase_pref, ...
        'Order', k, ...
        'Categorical', cat_vars, ...
        'CategoricalNames', {'electrode','sex'}, ...
        'Interactions', intx_k, ...
        'PredictorName', 'Age', ...
        'ClusterID', tbl.Subj_ID);
    LL_fixed(ii) = mdl_f.LogLikelihood;

    % --- sin/cos LME: build formula for this order ---
    fml = build_lme_formula(k);
    mdl_s = fitlme(tbl, sprintf(fml, 'sin_phase'));
    mdl_c = fitlme(tbl, sprintf(fml, 'cos_phase'));
    LL_sin(ii)  = mdl_s.LogLikelihood;
    LL_cos(ii)  = mdl_c.LogLikelihood;
    np_sincos(ii) = mdl_s.NumCoefficients;

    fprintf(['Order %d  npar(circ)=%d npar(sin/cos)=%d  ', ...
             'LL: orig=%.2f fixed=%.2f sin=%.2f cos=%.2f\n'], ...
        k, np_circ(ii), np_sincos(ii), ...
        LL_orig(ii), LL_fixed(ii), LL_sin(ii), LL_cos(ii));
end

% =====================================================================
% Iterative LRT order selection
% =====================================================================
% Procedure (matches get_best_iterative_order.m): start at order 0,
% step up by 1 each iteration.  Compare adjacent orders; if the
% improvement is significant at p < 0.05, accept the higher order and
% continue; otherwise stop and return the lower order.

fprintf('\n=== ITERATIVE LRT ORDER SELECTION ===\n');
fprintf('Procedure: keep stepping up while LRT p < 0.05; stop otherwise.\n');
fprintf('Truth: order 3 (cubic age main effect is real)\n\n');

[sel_orig,  trace_orig]  = iter_lrt(LL_orig,  np_circ);
[sel_fixed, trace_fixed] = iter_lrt(LL_fixed, np_circ);

% sin/cos joint: at each step compute LRT in each component, combine
% via Bonferroni (min p * 2).  Conservative but valid for two correlated
% tests, and matches what the paper would do for a sin/cos joint claim.
[sel_sincos, trace_sincos] = iter_lrt_sincos(LL_sin, LL_cos, np_sincos);

fprintf('--- Method-by-method results ---\n');
fprintf('Original circular_regression:  selected order = %d\n', sel_orig);
print_trace(trace_orig);
fprintf('\nFixed circular_regression_fixed:  selected order = %d\n', sel_fixed);
print_trace(trace_fixed);
fprintf('\nsin/cos LME (Bonferroni joint):   selected order = %d\n', sel_sincos);
print_trace(trace_sincos);

fprintf('\n--- Summary table (iterative LRT) ---\n');
fprintf('%-40s %-10s %-10s\n', 'Method', 'Selected', 'Correct?');
fprintf('%s\n', repmat('-', 1, 60));
fprintf('%-40s %-10d %-10s\n', 'Original circular_regression',  sel_orig,   tf(sel_orig==3));
fprintf('%-40s %-10d %-10s\n', 'Fixed circular_regression',     sel_fixed,  tf(sel_fixed==3));
fprintf('%-40s %-10d %-10s\n', 'sin/cos LME (Bonferroni joint)',sel_sincos, tf(sel_sincos==3));


% =====================================================================
% Diagnostic: block joint LRT (order 0 vs each higher order) + AIC
% =====================================================================
% The iterative LRT can fail to detect higher-order structure when the
% linear approximation alone is insufficient (e.g., a true U-shape that
% a flat line cannot capture).  Show what the methods CAN detect when
% allowed to compare directly against the order-0 baseline, plus the
% AIC/BIC trace across orders for an information-criterion view.

fprintf('\n=== DIAGNOSTIC: block joint LRT (order 0 vs each higher) ===\n');
fprintf('Tests "is the entire polynomial block significant?" non-iteratively.\n\n');
fprintf('%-8s %-30s %-30s %-30s\n', 'Order', 'Original LRT', 'Fixed LRT', 'sin/cos joint (Bonferroni)');
fprintf('%s\n', repmat('-', 1, 110));
for ii = 2:nO   % orders 1..4 vs order 0
    k = orders(ii);
    df = np_circ(ii) - np_circ(1);

    % Original / fixed
    chi2_o = 2*(LL_orig(ii)  - LL_orig(1));
    chi2_f = 2*(LL_fixed(ii) - LL_fixed(1));
    p_o = 1 - chi2cdf(max(chi2_o,0), df);
    p_f = 1 - chi2cdf(max(chi2_f,0), df);

    % sin/cos joint
    df_sc = np_sincos(ii) - np_sincos(1);
    chi2_s = 2*(LL_sin(ii) - LL_sin(1));
    chi2_c = 2*(LL_cos(ii) - LL_cos(1));
    p_s = 1 - chi2cdf(max(chi2_s,0), df_sc);
    p_c = 1 - chi2cdf(max(chi2_c,0), df_sc);
    p_joint = min(1, 2*min(p_s, p_c));

    fprintf('%-8d chi2(%d)=%.2f p=%.4g    chi2(%d)=%.2f p=%.4g    p_sin=%.4g p_cos=%.4g joint=%.4g\n', ...
        k, df, chi2_o, p_o, df, chi2_f, p_f, p_s, p_c, p_joint);
end

% AIC/BIC for the circular and sin/cos models.  AIC = -2*LL + 2*npar;
% BIC = -2*LL + log(n)*npar.  Lower is better.
n_obs = height(tbl);
fprintf('\n=== AIC / BIC across orders ===\n');
fprintf('%-8s %-25s %-25s %-25s %-25s\n', 'Order', 'Orig (AIC, BIC)', 'Fixed (AIC, BIC)', ...
        'sin (AIC, BIC)', 'cos (AIC, BIC)');
fprintf('%s\n', repmat('-', 1, 120));
for ii = 1:nO
    k = orders(ii);
    aic_o = -2*LL_orig(ii)  + 2*np_circ(ii);
    bic_o = -2*LL_orig(ii)  + log(n_obs)*np_circ(ii);
    aic_f = -2*LL_fixed(ii) + 2*np_circ(ii);
    bic_f = -2*LL_fixed(ii) + log(n_obs)*np_circ(ii);
    aic_s = -2*LL_sin(ii)   + 2*np_sincos(ii);
    bic_s = -2*LL_sin(ii)   + log(n_obs)*np_sincos(ii);
    aic_c = -2*LL_cos(ii)   + 2*np_sincos(ii);
    bic_c = -2*LL_cos(ii)   + log(n_obs)*np_sincos(ii);
    fprintf('%-8d %.1f, %.1f          %.1f, %.1f          %.1f, %.1f          %.1f, %.1f\n', ...
        k, aic_o, bic_o, aic_f, bic_f, aic_s, bic_s, aic_c, bic_c);
end
[~, best_aic_orig]   = min(-2*LL_orig  + 2*np_circ);
[~, best_aic_fixed]  = min(-2*LL_fixed + 2*np_circ);
[~, best_aic_sin]    = min(-2*LL_sin   + 2*np_sincos);
[~, best_aic_cos]    = min(-2*LL_cos   + 2*np_sincos);
fprintf('\nBest AIC order: orig=%d  fixed=%d  sin=%d  cos=%d\n', ...
    orders(best_aic_orig), orders(best_aic_fixed), ...
    orders(best_aic_sin),  orders(best_aic_cos));
[~, best_bic_orig]   = min(-2*LL_orig  + log(n_obs)*np_circ);
[~, best_bic_fixed]  = min(-2*LL_fixed + log(n_obs)*np_circ);
[~, best_bic_sin]    = min(-2*LL_sin   + log(n_obs)*np_sincos);
[~, best_bic_cos]    = min(-2*LL_cos   + log(n_obs)*np_sincos);
fprintf('Best BIC order: orig=%d  fixed=%d  sin=%d  cos=%d\n', ...
    orders(best_bic_orig), orders(best_bic_fixed), ...
    orders(best_bic_sin),  orders(best_bic_cos));

fprintf('\nbrms order-selection via LOO is in test_order_selection.R\n');


% =====================================================================
% Helpers
% =====================================================================
function fml = build_lme_formula(k)
% Build a fitlme formula for sin_phase / cos_phase at polynomial order k,
% with electrode + sex main effects, electrode * polynomial interactions,
% and a per-subject random intercept.
base = '%s ~ 1 + electrode + sex + (1|Subj_ID)';
if k == 0
    fml = base;
    return
end
% Add polynomial main effects + interactions.
poly_terms = {};
intx_terms = {};
for kk = 1:k
    if kk == 1
        v = 'Age';
    else
        v = sprintf('Age%d', kk);
    end
    poly_terms{end+1} = v; %#ok<AGROW>
    intx_terms{end+1} = [v ':electrode']; %#ok<AGROW>
end
extra = strjoin([poly_terms intx_terms], ' + ');
fml = ['%s ~ 1 + ' extra ' + electrode + sex + (1|Subj_ID)'];
end


function [sel_order, trace] = iter_lrt(LL, npar)
% Standard iterative LRT on log-likelihoods.
% LL(i) is the log-likelihood at order(i) = i-1 (0-indexed).
% npar(i) is the parameter count at that order.
%
% Step from order 0 upward; accept higher order if p(chi^2 LRT) < .05,
% else stop.  Returns the largest accepted order and the per-step trace.
sel_order = 0;
trace = struct('from', {}, 'to', {}, 'chi2', {}, 'df', {}, 'p', {}, 'accepted', {});
for k = 1:length(LL)-1
    chi2 = 2 * (LL(k+1) - LL(k));
    df   = npar(k+1) - npar(k);
    if df <= 0 || chi2 < 0
        p = 1;
    else
        p = 1 - chi2cdf(chi2, df);
    end
    accepted = p < 0.05;
    trace(end+1) = struct('from', k-1, 'to', k, ...
        'chi2', chi2, 'df', df, 'p', p, 'accepted', accepted); %#ok<AGROW>
    if accepted
        sel_order = k;
    else
        break;
    end
end
end


function [sel_order, trace] = iter_lrt_sincos(LL_sin, LL_cos, npar)
% Iterative LRT for sin/cos LMEs: at each step compute LRT for each
% component, take min(p_sin, p_cos)*2 (Bonferroni) as the joint p.
sel_order = 0;
trace = struct('from', {}, 'to', {}, ...
               'chi2_sin', {}, 'p_sin', {}, ...
               'chi2_cos', {}, 'p_cos', {}, ...
               'p_joint', {}, 'accepted', {});
for k = 1:length(LL_sin)-1
    chi2_s = 2 * (LL_sin(k+1) - LL_sin(k));
    chi2_c = 2 * (LL_cos(k+1) - LL_cos(k));
    df = npar(k+1) - npar(k);
    p_s = 1 - chi2cdf(max(chi2_s, 0), df);
    p_c = 1 - chi2cdf(max(chi2_c, 0), df);
    p_joint = min(1, 2 * min(p_s, p_c));
    accepted = p_joint < 0.05;
    trace(end+1) = struct('from', k-1, 'to', k, ...
        'chi2_sin', chi2_s, 'p_sin', p_s, ...
        'chi2_cos', chi2_c, 'p_cos', p_c, ...
        'p_joint', p_joint, 'accepted', accepted); %#ok<AGROW>
    if accepted
        sel_order = k;
    else
        break;
    end
end
end


function print_trace(trace)
for i = 1:length(trace)
    t = trace(i);
    if isfield(t, 'p_joint')
        fprintf(['  order %d -> %d:  chi2_sin=%.2f p_sin=%.4g  ', ...
                 'chi2_cos=%.2f p_cos=%.4g  joint p=%.4g  %s\n'], ...
            t.from, t.to, t.chi2_sin, t.p_sin, ...
            t.chi2_cos, t.p_cos, t.p_joint, ...
            tf_simple(t.accepted));
    else
        fprintf('  order %d -> %d:  chi2(%d) = %.2f  p = %.4g  %s\n', ...
            t.from, t.to, t.df, t.chi2, t.p, tf_simple(t.accepted));
    end
end
end


function s = tf(b)
if b, s = 'YES'; else, s = 'no'; end
end
function s = tf_simple(b)
if b, s = '[accept]'; else, s = '[stop]'; end
end
