%TEST_MC_SIMULATION  Monte Carlo bias / coverage for circular LMEs.
%
% For each kappa in {4, 10, 20, 50, 100}, generate n_reps independent
% datasets from the doubled-trend DGP and fit three methods:
%   1. fitlme_circ            (sin/cos LME pair, Bonferroni)
%   2. fitcirc_lme            (vM EM with Laplace correction)
%   3. circular_regression_fixed.m (cluster-robust vM MLE, no RE)
%
% Records coefficient estimates and standard errors per replicate, then
% aggregates: empirical bias, Monte Carlo SE of the bias, and 95% CI
% coverage of the angle-scale truth.  Results saved to a CSV consumed
% by the LaTeX paper update.

clear; close all;
addpath(fileparts(mfilename('fullpath')));
rng(7);

n_reps     = 50;
kappa_list = [4, 10, 20, 50, 100];
n_subjects = 300;
n_per      = 2;          % obs per subject
sigma_phi  = 0.4;

% Doubled trend (matches the kappa sweep)
true_b = [ 0.6;  -0.12;  0.0024;  -1.4e-5;  0.30;  -0.10;  0.008;  -4e-5;  0 ];

% Term names in MATLAB Wilkinson order (matches CoefficientNames produced
% by fitlme on the formula below).
term_names = {'(Intercept)','Age','sex','electrode','Age2','Age3', ...
              'Age:electrode','electrode:Age2','electrode:Age3'};
% Map true coefficients onto term order
truth_by_term = [true_b(1); true_b(2); true_b(6); true_b(5); ...
                 true_b(3); true_b(4); true_b(7); true_b(8); true_b(9)];

n_terms = length(term_names);
fml = ['phase_pref ~ 1 + Age + Age2 + Age3 + electrode + sex + ' ...
       'Age:electrode + Age2:electrode + Age3:electrode + (1|Subj_ID)'];

% Storage
methods  = {'sincos','vMEM','clRobust'};
n_meth   = length(methods);
% [n_reps x n_terms x n_kappa x n_meth]
est_arr  = nan(n_reps, n_terms, length(kappa_list), n_meth);
se_arr   = nan(n_reps, n_terms, length(kappa_list), n_meth);
% kappa, sigma_phi recovery (only available for some methods)
kappa_hat = nan(n_reps, length(kappa_list), n_meth);
sigma_hat = nan(n_reps, length(kappa_list), n_meth);

t_start = tic;
for ki = 1:length(kappa_list)
    kappa_true = kappa_list(ki);
    fprintf('\n=== kappa = %g ===\n', kappa_true);
    for r = 1:n_reps
        rng(7 + 1000*ki + r);                 % deterministic, distinct
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

        % --- fitlme_circ ---
        try
            mdl1 = fitlme_circ(tbl, fml);
            for ti = 1:n_terms
                idx = find(strcmp(mdl1.CoefficientNames, term_names{ti}), 1);
                if ~isempty(idx)
                    % "Estimate" for the sin/cos pair: average of the
                    % component coefs, since both estimate angle * A1
                    % up to multiplicative factors and we want a single
                    % scalar per term.  This mirrors how a paper would
                    % typically report a sin/cos LME finding.
                    est_arr(r, ti, ki, 1) = ...
                        mean([mdl1.Coefficients.EstSin(idx), ...
                              mdl1.Coefficients.EstCos(idx)]);
                    se_arr(r, ti, ki, 1) = ...
                        mean([mdl1.Coefficients.SESin(idx), ...
                              mdl1.Coefficients.SECos(idx)]);
                end
            end
            kappa_hat(r, ki, 1) = mdl1.ResidualKappa;
        catch ME
            fprintf('  rep %d sincos failed: %s\n', r, ME.message);
        end

        % --- fitcirc_lme ---
        try
            mdl2 = fitcirc_lme(tbl, fml, 'MaxIter', 100);
            for ti = 1:n_terms
                idx = find(strcmp(mdl2.Coefficients.Name, term_names{ti}), 1);
                if ~isempty(idx)
                    est_arr(r, ti, ki, 2) = mdl2.Coefficients.Estimate(idx);
                    se_arr(r, ti, ki, 2) = mdl2.Coefficients.SE(idx);
                end
            end
            kappa_hat(r, ki, 2) = mdl2.Kappa;
            sigma_hat(r, ki, 2) = mdl2.SigmaPhi;
        catch ME
            fprintf('  rep %d vMEM failed: %s\n', r, ME.message);
        end

        % --- circular_regression_fixed (cluster-robust, no RE) ---
        try
            cat_vars  = [tbl.electrode, tbl.sex];
            intx_full = [true, false];
            [~, mdl3] = circular_regression_fixed(tbl.Age, tbl.phase_pref, ...
                'Order', 3, ...
                'Categorical', cat_vars, ...
                'CategoricalNames', {'electrode','sex'}, ...
                'Interactions', intx_full, ...
                'PredictorName', 'Age', ...
                'ClusterID', tbl.Subj_ID);
            % Map circular_regression_fixed coefficient names to ours.
            % Their order is: Intercept, Age, Age^2, Age^3, electrode, sex,
            %                 electrode:Age, electrode:Age^2, electrode:Age^3
            cn = mdl3.CoefficientNames;
            for ti = 1:n_terms
                t = term_names{ti};
                target = '';
                switch t
                    case '(Intercept)',     target = '(Intercept)';
                    case 'Age',             target = 'Age';
                    case 'Age2',            target = 'Age^2';
                    case 'Age3',            target = 'Age^3';
                    case 'electrode',       target = 'electrode';
                    case 'sex',             target = 'sex';
                    case 'Age:electrode',   target = 'electrode:Age';
                    case 'electrode:Age2',  target = 'electrode:Age^2';
                    case 'electrode:Age3',  target = 'electrode:Age^3';
                end
                idx = find(strcmp(cn, target), 1);
                if ~isempty(idx)
                    est_arr(r, ti, ki, 3) = mdl3.Coefficients.Estimate(idx);
                    se_arr(r, ti, ki, 3) = mdl3.Coefficients.SE(idx);
                end
            end
            kappa_hat(r, ki, 3) = mdl3.Kappa;
        catch ME
            fprintf('  rep %d clRobust failed: %s\n', r, ME.message);
        end

        if mod(r, 10) == 0
            fprintf('  rep %d/%d  elapsed %.0fs\n', r, n_reps, toc(t_start));
        end
    end
end

% ---------------------------------------------------------------------
% Aggregate: bias, MC SE, and 95% CI coverage of angle-scale truth
% ---------------------------------------------------------------------
fprintf('\n\nAggregating...\n');
bias  = nan(n_terms, length(kappa_list), n_meth);
mcse  = nan(n_terms, length(kappa_list), n_meth);
cov95 = nan(n_terms, length(kappa_list), n_meth);

for ki = 1:length(kappa_list)
    for mi = 1:n_meth
        for ti = 1:n_terms
            x  = est_arr(:, ti, ki, mi);
            se = se_arr(:, ti, ki, mi);
            ok = isfinite(x) & isfinite(se);
            if sum(ok) < 2, continue; end
            xt = truth_by_term(ti);
            bias(ti, ki, mi) = mean(x(ok)) - xt;
            mcse(ti, ki, mi) = std(x(ok)) / sqrt(sum(ok));
            % 95% Wald CI per replicate; coverage is fraction containing truth.
            % Use n_subjects-1 df for vM EM (sandwich), n-p for fitlme.
            df = n_subjects - 1;
            t_crit = tinv(0.975, df);
            lo = x(ok) - t_crit * se(ok);
            hi = x(ok) + t_crit * se(ok);
            cov95(ti, ki, mi) = mean(lo <= xt & xt <= hi);
        end
    end
end

% Save raw arrays for the paper
save_path = fullfile(fileparts(mfilename('fullpath')), '..', 'mc_results.mat');
save(save_path, 'kappa_list','term_names','truth_by_term','methods', ...
                'est_arr','se_arr','kappa_hat','sigma_hat', ...
                'bias','mcse','cov95','n_reps','-v7.3');
fprintf('Saved %s\n', save_path);

% Tabular summary CSV: one row per (term, kappa, method)
out_rows = {};
for ti = 1:n_terms
    for ki = 1:length(kappa_list)
        for mi = 1:n_meth
            out_rows(end+1, :) = { ...
                term_names{ti}, kappa_list(ki), methods{mi}, ...
                truth_by_term(ti), bias(ti, ki, mi), mcse(ti, ki, mi), ...
                cov95(ti, ki, mi) };
        end
    end
end
T = cell2table(out_rows, 'VariableNames', {'term','kappa','method', ...
    'truth','bias','mcse','coverage'});
csv_path = fullfile(fileparts(mfilename('fullpath')), '..', 'mc_results.csv');
writetable(T, csv_path);
fprintf('Saved %s\n', csv_path);

% Variance-component summary
vc_rows = {};
for ki = 1:length(kappa_list)
    for mi = 1:n_meth
        kk = kappa_hat(:, ki, mi); kk = kk(isfinite(kk));
        ss = sigma_hat(:, ki, mi); ss = ss(isfinite(ss));
        vc_rows(end+1, :) = { ...
            kappa_list(ki), methods{mi}, ...
            mean(kk), std(kk), ...
            mean(ss), std(ss), length(kk)};
    end
end
T_vc = cell2table(vc_rows, 'VariableNames', ...
    {'kappa_true','method','kappa_mean','kappa_sd','sigma_mean','sigma_sd','n_used'});
vc_path = fullfile(fileparts(mfilename('fullpath')), '..', 'mc_variance_components.csv');
writetable(T_vc, vc_path);
fprintf('Saved %s\n', vc_path);

fprintf('\nTotal time: %.1f min\n', toc(t_start)/60);
