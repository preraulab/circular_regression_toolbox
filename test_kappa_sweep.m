%TEST_KAPPA_SWEEP  Order-selection sensitivity to noise level.
%
% Re-runs the iterative LRT order-selection at fixed N = 300 for several
% values of kappa (the von Mises concentration -> 1/sqrt(kappa) is the
% additive noise SD on the latent angle).  The aim is to separate two
% explanations for the previous test_order_selection.m result that every
% method picked order 0 when truth was order 3:
%
%   (a) Noise too high.  Lift kappa, the iterative LRT recovers.
%   (b) Structural blindness to non-monotonic shapes.  Iterative LRT keeps
%       missing it even at very high SNR because the cubic's linear
%       projection onto Age is near zero in this design, so the order
%       0->1 step never clears p < .05.
%
% Reports for each kappa: selected order via iterative LRT for original /
% fixed circular regression and sin/cos-LME (Bonferroni); plus the block
% joint LRT (order 0 vs order 3) p-value as a sanity check that the
% effect IS recoverable in principle.

clear; close all;
addpath(fileparts(mfilename('fullpath')));

% =====================================================================
% Common setup
% =====================================================================
n_subjects = 300;
% Trend coefficients DOUBLED: age polynomial + electrode:age interactions
% (intercept, electrode main, sex main unchanged -- those aren't "trends").
true_b = [ 0.6;  -0.12;  0.0024;  -1.4e-5;  0.30;  -0.10;  0.008;  -4e-5;  0 ];
kappa_list = [4, 10, 20, 50, 100];
orders     = 0:4;
nO         = length(orders);

results = struct('kappa', {}, ...
                 'sel_orig', {}, 'sel_fixed', {}, 'sel_sincos', {}, ...
                 'p_block3_orig', {}, 'p_block3_fixed', {}, ...
                 'p_block3_sincos', {});

for ki = 1:length(kappa_list)
    kappa_true = kappa_list(ki);
    rng(42);   % same subjects/effects across kappas; only noise changes

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

    X_true = [ones(n,1), Age, Age.^2, Age.^3, electrode, sex, ...
              electrode.*Age, electrode.*Age.^2, electrode.*Age.^3];
    true_phase = X_true * true_b + re;
    phase_pref = wrapToPi(true_phase + randn(n,1)/sqrt(kappa_true));

    tbl = table(subj_id, Age, sex, electrode, phase_pref, ...
                'VariableNames', {'Subj_ID','Age','sex','electrode','phase_pref'});
    tbl.sin_phase = sin(tbl.phase_pref);
    tbl.cos_phase = cos(tbl.phase_pref);
    tbl.Age2 = tbl.Age.^2;
    tbl.Age3 = tbl.Age.^3;
    tbl.Age4 = tbl.Age.^4;

    csv_path = fullfile(fileparts(mfilename('fullpath')), '..', ...
        sprintf('test_kappa_sweep_K%g.csv', kappa_true));
    writetable(tbl, csv_path);

    fprintf('\n========================================================\n');
    fprintf('  kappa = %g  (per-obs angular SD ~ %.3f rad)\n', ...
            kappa_true, 1/sqrt(kappa_true));
    fprintf('========================================================\n');

    LL_orig   = nan(1, nO);
    LL_fixed  = nan(1, nO);
    LL_sin    = nan(1, nO);
    LL_cos    = nan(1, nO);
    np_circ   = nan(1, nO);
    np_sincos = nan(1, nO);

    cat_vars   = [tbl.electrode, tbl.sex];
    varnames   = {'phase_pref','Age','electrode','sex'};
    intx_full  = [true, false];

    for ii = 1:nO
        k = orders(ii);
        if k == 0
            intx_k = [false, false];
        else
            intx_k = intx_full;
        end

        np = 1 + k + size(cat_vars,2) + sum(intx_k)*k;
        b0 = zeros(np, 1);
        b0(1) = atan2(mean(sin(tbl.phase_pref)), mean(cos(tbl.phase_pref)));
        [~, mdl_o] = circular_regression(tbl.Age, tbl.phase_pref, cat_vars, ...
            varnames(2:end), k, intx_k, b0, 1000);
        LL_orig(ii) = mdl_o.LogLikelihood;
        np_circ(ii) = mdl_o.NumCoefficients;

        [~, mdl_f] = circular_regression_fixed(tbl.Age, tbl.phase_pref, ...
            'Order', k, ...
            'Categorical', cat_vars, ...
            'CategoricalNames', {'electrode','sex'}, ...
            'Interactions', intx_k, ...
            'PredictorName', 'Age', ...
            'ClusterID', tbl.Subj_ID);
        LL_fixed(ii) = mdl_f.LogLikelihood;

        fml = build_lme_formula(k);
        mdl_s = fitlme(tbl, sprintf(fml, 'sin_phase'));
        mdl_c = fitlme(tbl, sprintf(fml, 'cos_phase'));
        LL_sin(ii)  = mdl_s.LogLikelihood;
        LL_cos(ii)  = mdl_c.LogLikelihood;
        np_sincos(ii) = mdl_s.NumCoefficients;
    end

    sel_orig   = iter_lrt(LL_orig,  np_circ);
    sel_fixed  = iter_lrt(LL_fixed, np_circ);
    sel_sincos = iter_lrt_sincos(LL_sin, LL_cos, np_sincos);

    p_block3_orig   = block_p(LL_orig,  np_circ,   3+1);  % index 4 = order 3
    p_block3_fixed  = block_p(LL_fixed, np_circ,   3+1);
    p_block3_sincos = block_p_sincos(LL_sin, LL_cos, np_sincos, 3+1);

    fprintf('  iterative LRT  -> orig=%d  fixed=%d  sin/cos(Bonf)=%d   (truth=3)\n', ...
            sel_orig, sel_fixed, sel_sincos);
    fprintf('  block(0 vs 3)  -> p_orig=%.4g  p_fixed=%.4g  p_sincos=%.4g\n', ...
            p_block3_orig, p_block3_fixed, p_block3_sincos);

    results(end+1) = struct('kappa', kappa_true, ...
        'sel_orig', sel_orig, 'sel_fixed', sel_fixed, 'sel_sincos', sel_sincos, ...
        'p_block3_orig', p_block3_orig, ...
        'p_block3_fixed', p_block3_fixed, ...
        'p_block3_sincos', p_block3_sincos); %#ok<SAGROW>
end

% =====================================================================
% Summary
% =====================================================================
fprintf('\n========================================================\n');
fprintf('  SUMMARY (truth = 3, N = 300)\n');
fprintf('========================================================\n');
fprintf('%-10s | %-22s | %-22s\n', ...
        'kappa', 'iterative LRT order', 'block(0 vs 3) p-value');
fprintf('%-10s | %-7s %-7s %-7s | %-7s %-7s %-7s\n', ...
        '', 'orig', 'fixed', 'sin/cos', 'orig', 'fixed', 'sin/cos');
fprintf('%s\n', repmat('-', 1, 70));
for ki = 1:length(results)
    r = results(ki);
    fprintf('%-10g | %-7d %-7d %-7d | %-7.4g %-7.4g %-7.4g\n', ...
        r.kappa, r.sel_orig, r.sel_fixed, r.sel_sincos, ...
        r.p_block3_orig, r.p_block3_fixed, r.p_block3_sincos);
end


% =====================================================================
% Helpers
% =====================================================================
function fml = build_lme_formula(k)
base = '%s ~ 1 + electrode + sex + (1|Subj_ID)';
if k == 0, fml = base; return; end
poly_terms = {}; intx_terms = {};
for kk = 1:k
    if kk == 1, v = 'Age'; else, v = sprintf('Age%d', kk); end
    poly_terms{end+1} = v; %#ok<AGROW>
    intx_terms{end+1} = [v ':electrode']; %#ok<AGROW>
end
extra = strjoin([poly_terms intx_terms], ' + ');
fml = ['%s ~ 1 + ' extra ' + electrode + sex + (1|Subj_ID)'];
end


function sel = iter_lrt(LL, npar)
sel = 0;
for k = 1:length(LL)-1
    chi2 = 2 * (LL(k+1) - LL(k));
    df   = npar(k+1) - npar(k);
    if df <= 0 || chi2 < 0
        p = 1;
    else
        p = 1 - chi2cdf(chi2, df);
    end
    if p < 0.05, sel = k; else, break; end
end
end


function sel = iter_lrt_sincos(LL_sin, LL_cos, npar)
sel = 0;
for k = 1:length(LL_sin)-1
    chi2_s = 2 * (LL_sin(k+1) - LL_sin(k));
    chi2_c = 2 * (LL_cos(k+1) - LL_cos(k));
    df = npar(k+1) - npar(k);
    p_s = 1 - chi2cdf(max(chi2_s, 0), df);
    p_c = 1 - chi2cdf(max(chi2_c, 0), df);
    p_joint = min(1, 2 * min(p_s, p_c));
    if p_joint < 0.05, sel = k; else, break; end
end
end


function p = block_p(LL, npar, idx_high)
chi2 = 2 * (LL(idx_high) - LL(1));
df   = npar(idx_high) - npar(1);
p = 1 - chi2cdf(max(chi2,0), df);
end


function p = block_p_sincos(LL_sin, LL_cos, npar, idx_high)
chi2_s = 2 * (LL_sin(idx_high) - LL_sin(1));
chi2_c = 2 * (LL_cos(idx_high) - LL_cos(1));
df = npar(idx_high) - npar(1);
p_s = 1 - chi2cdf(max(chi2_s,0), df);
p_c = 1 - chi2cdf(max(chi2_c,0), df);
p = min(1, 2 * min(p_s, p_c));
end
