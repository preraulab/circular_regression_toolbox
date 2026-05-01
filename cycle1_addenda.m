% CYCLE1_ADDENDA  Cycle-1 reviewer-requested addenda.
%
% (a) Verify mean(cos eta), mean(sin eta) on the eq.(5) DGP.
% (b) Compute 2x2 Fisher info condition number for (sigma_phi, kappa)
%     at (0.4, 20) on the actual design, n_j = 2 vs n_j = 10.
% (c) Compute MC order-recovery rate (Age^3 detection at alpha=0.05)
%     from existing mc_results.mat.
%
% Saves /Users/Mike/code/projects/trends_v_individual/mc_cycle1_addenda.mat

clear; close all;
addpath(fileparts(mfilename('fullpath')));
rng(7);

repo_root = fullfile(fileparts(mfilename('fullpath')), '..');

% =====================================================================
% (a) Sample means of cos(eta), sin(eta) on the eq. (5) DGP
% =====================================================================
fprintf('=== (a) Sample means of cos(eta), sin(eta) ===\n');

n_subjects = 300;
n_per      = 2;
sigma_phi  = 0.4;
true_b = [ 0.6;  -0.12;  0.0024;  -1.4e-5;  0.30;  -0.10;  0.008;  -4e-5;  0 ];

rng(7);  % match test_mc_simulation.m initial seed for first replicate
subj_age   = 7 + 73 * rand(n_subjects, 1);
subj_sex   = double(rand(n_subjects, 1) > 0.5);
Age        = repelem(subj_age, n_per);
sex        = repelem(subj_sex, n_per);
electrode  = repmat([1; 0], n_subjects, 1);

X_true = [ones(length(Age),1), Age, Age.^2, Age.^3, electrode, sex, ...
          electrode.*Age, electrode.*Age.^2, electrode.*Age.^3];
eta_true_no_re = X_true * true_b;   % do NOT add phi for the deflation
                                    % calculation: deflation is computed
                                    % over the design only.

mean_cos_eta = mean(cos(eta_true_no_re));
mean_sin_eta = mean(sin(eta_true_no_re));
fprintf('  mean(cos eta) = %.4f\n', mean_cos_eta);
fprintf('  mean(sin eta) = %.4f\n', mean_sin_eta);

A1_kappa20 = besseli(1, 20) / besseli(0, 20);
defl_factor_re = exp(-sigma_phi^2 / 2);
predicted_sin_slope = A1_kappa20 * mean_cos_eta * true_b(2) * defl_factor_re;
predicted_cos_slope = -A1_kappa20 * mean_sin_eta * true_b(2) * defl_factor_re;
fprintf('  A1(20) = %.4f, exp(-sigma^2/2) = %.4f\n', A1_kappa20, defl_factor_re);
fprintf('  Predicted sin slope = %.5f\n', predicted_sin_slope);
fprintf('  Predicted cos slope = %.5f\n', predicted_cos_slope);

% Average the two -- this is what the paper currently quotes.
predicted_avg_slope = (predicted_sin_slope + predicted_cos_slope) / 2;
fprintf('  Predicted avg slope = %.5f (truth = %.5f)\n', ...
    predicted_avg_slope, true_b(2));

% Also report scalar deflation factors
defl_sin = A1_kappa20 * mean_cos_eta * defl_factor_re;
defl_cos = A1_kappa20 * abs(mean_sin_eta) * defl_factor_re;
fprintf('  Deflation factor sin direction: %.4f\n', defl_sin);
fprintf('  Deflation factor cos direction: %.4f\n', defl_cos);
fprintf('  Mean deflation: %.4f\n', mean([defl_sin, defl_cos]));

% =====================================================================
% (b) Fisher information for (sigma_phi, kappa) at (0.4, 20)
% =====================================================================
fprintf('\n=== (b) Fisher information condition numbers ===\n');

% Use Gauss-Hermite quadrature over phi, marginal log-likelihood per subj.
% theta = (sigma_phi, kappa)
% phi_i ~ N(0, sigma_phi^2)
% y_ij | phi_i ~ vM(eta_ij + phi_i, kappa)
% l_i(theta) = log integral over phi of prod_j vM(y_ij - eta_ij - phi; kappa)
%              * N(phi; 0, sigma_phi^2) dphi

% Gauss-Hermite nodes
n_gh = 20;
[gh_x, gh_w] = local_gauss_hermite(n_gh);
% After change of variables phi = sqrt(2) * sigma_phi * z, so
% integral over phi w/ N(0, sigma_phi^2) = (1/sqrt(pi)) sum w_k * f(sqrt(2) sigma_phi z_k)
% l_i(theta) = -log(sqrt(pi)) + log sum_k exp( log(w_k) + sum_j log_vM(y_ij - eta_ij - sqrt(2) sigma_phi z_k; kappa) )

theta_true = [0.4; 20];

condition_numbers = zeros(2, 1);
det_info          = zeros(2, 1);
n_j_list          = [2, 10];

for nj_i = 1:length(n_j_list)
    n_j_use = n_j_list(nj_i);
    fprintf('\n  --- n_j = %d ---\n', n_j_use);

    % Build design with n_j obs per subject; reuse the same n_subjects
    if n_j_use == 2
        rng(7);
        sa  = 7 + 73 * rand(n_subjects, 1);
        ss_ = double(rand(n_subjects, 1) > 0.5);
        sphi = sigma_phi * randn(n_subjects, 1);
        Age_ = repelem(sa, n_j_use);
        sex_ = repelem(ss_, n_j_use);
        electrode_ = repmat([1; 0], n_subjects, 1);
        re_ = repelem(sphi, n_j_use);
        subj_id_ = repelem((1:n_subjects)', n_j_use);
    else
        % n_j = 10: replicate each row 5x in time order, alternating electrode
        % Use the same subjects/ages but stack 5 copies of the [1;0] electrode
        rng(7);
        sa  = 7 + 73 * rand(n_subjects, 1);
        ss_ = double(rand(n_subjects, 1) > 0.5);
        sphi = sigma_phi * randn(n_subjects, 1);
        Age_ = repelem(sa, n_j_use);
        sex_ = repelem(ss_, n_j_use);
        electrode_ = repmat([1; 0; 1; 0; 1; 0; 1; 0; 1; 0], n_subjects, 1);
        re_ = repelem(sphi, n_j_use);
        subj_id_ = repelem((1:n_subjects)', n_j_use);
    end
    n_obs = length(Age_);
    X_     = [ones(n_obs,1), Age_, Age_.^2, Age_.^3, electrode_, sex_, ...
              electrode_.*Age_, electrode_.*Age_.^2, electrode_.*Age_.^3];
    eta_   = X_ * true_b;

    % Simulate one replicate
    rng(42);
    y_ = wrapToPi(eta_ + re_ + randn(n_obs,1)/sqrt(theta_true(2)));

    % Per-subject info
    % We'll compute the Hessian of -l_i wrt (sigma, kappa) at the truth
    % via central differences with h_sig = 1e-3, h_kap = 1e-2 (kappa is bigger)
    h_sig = 1e-3;
    h_kap = 1e-2;
    info_total = zeros(2, 2);

    for i = 1:n_subjects
        idx_i = subj_id_ == i;
        y_i   = y_(idx_i);
        eta_i = eta_(idx_i);

        % Loglik f(theta) = log p(y_i | sigma, kappa)
        f0 = subj_loglik(y_i, eta_i, theta_true(1), theta_true(2), gh_x, gh_w);
        fpp_sig = subj_loglik(y_i, eta_i, theta_true(1)+h_sig, theta_true(2), gh_x, gh_w);
        fmp_sig = subj_loglik(y_i, eta_i, theta_true(1)-h_sig, theta_true(2), gh_x, gh_w);
        fpp_kap = subj_loglik(y_i, eta_i, theta_true(1), theta_true(2)+h_kap, gh_x, gh_w);
        fmp_kap = subj_loglik(y_i, eta_i, theta_true(1), theta_true(2)-h_kap, gh_x, gh_w);
        % cross
        fpp = subj_loglik(y_i, eta_i, theta_true(1)+h_sig, theta_true(2)+h_kap, gh_x, gh_w);
        fmm = subj_loglik(y_i, eta_i, theta_true(1)-h_sig, theta_true(2)-h_kap, gh_x, gh_w);
        fpm = subj_loglik(y_i, eta_i, theta_true(1)+h_sig, theta_true(2)-h_kap, gh_x, gh_w);
        fmp = subj_loglik(y_i, eta_i, theta_true(1)-h_sig, theta_true(2)+h_kap, gh_x, gh_w);

        d2_sig = (fpp_sig - 2*f0 + fmp_sig) / h_sig^2;
        d2_kap = (fpp_kap - 2*f0 + fmp_kap) / h_kap^2;
        d2_x   = (fpp - fpm - fmp + fmm) / (4 * h_sig * h_kap);

        H_i = [d2_sig, d2_x; d2_x, d2_kap];
        info_total = info_total + (-H_i);
    end

    % Symmetrize
    info_total = (info_total + info_total') / 2;
    eigs_ = eig(info_total);
    cond_num = max(eigs_) / min(eigs_);
    det_  = det(info_total);
    fprintf('  Fisher info matrix:\n');
    disp(info_total);
    fprintf('  eigenvalues: %s\n', mat2str(eigs_, 4));
    fprintf('  condition number = %.3g\n', cond_num);
    fprintf('  determinant      = %.3g\n', det_);
    condition_numbers(nj_i) = cond_num;
    det_info(nj_i)          = det_;
end

% =====================================================================
% (c) MC empirical Age^3 order-recovery rate
% =====================================================================
fprintf('\n=== (c) MC empirical Age^3 detection rate ===\n');

mc = load(fullfile(repo_root, 'mc_results.mat'));
% est_arr: [n_reps, n_terms, n_kappa, n_meth]
% term_names from mc; find Age3
term_idx = find(strcmp(mc.term_names, 'Age3'), 1);
fprintf('  Age3 term index: %d\n', term_idx);
n_subj_mc = 300;  % per the MC config
df_use    = n_subj_mc - 1;

n_kappa = length(mc.kappa_list);
n_meth  = length(mc.methods);
detect_rate = nan(n_kappa, n_meth);
for ki = 1:n_kappa
    for mi = 1:n_meth
        est = mc.est_arr(:, term_idx, ki, mi);
        se  = mc.se_arr(:,  term_idx, ki, mi);
        ok  = isfinite(est) & isfinite(se) & (se > 0);
        if sum(ok) < 2, continue; end
        tstat = abs(est(ok) ./ se(ok));
        p_w   = 2 * (1 - tcdf(tstat, df_use));
        detect_rate(ki, mi) = mean(p_w < 0.05);
    end
end

fprintf('\nAge3 Wald-detection rate at alpha=0.05 (rows: kappa, cols: methods)\n');
fprintf('             ');
for mi = 1:n_meth, fprintf('%-12s', mc.methods{mi}); end
fprintf('\n');
for ki = 1:n_kappa
    fprintf('  kappa=%-5g', mc.kappa_list(ki));
    for mi = 1:n_meth
        fprintf('%-12.3f', detect_rate(ki, mi));
    end
    fprintf('\n');
end

% =====================================================================
% Save all addenda
% =====================================================================
out_path = fullfile(repo_root, 'mc_cycle1_addenda.mat');
save(out_path, ...
    'mean_cos_eta', 'mean_sin_eta', ...
    'A1_kappa20', 'defl_factor_re', ...
    'predicted_sin_slope', 'predicted_cos_slope', ...
    'predicted_avg_slope', 'defl_sin', 'defl_cos', ...
    'condition_numbers', 'det_info', 'n_j_list', ...
    'detect_rate', '-v7.3');
fprintf('\nSaved %s\n', out_path);

% Also CSV summary
csv_path = fullfile(repo_root, 'mc_cycle1_addenda.csv');
fid = fopen(csv_path, 'w');
fprintf(fid, 'quantity,value\n');
fprintf(fid, 'mean_cos_eta,%.6f\n', mean_cos_eta);
fprintf(fid, 'mean_sin_eta,%.6f\n', mean_sin_eta);
fprintf(fid, 'A1_kappa20,%.6f\n', A1_kappa20);
fprintf(fid, 'exp_neg_sig2_over_2,%.6f\n', defl_factor_re);
fprintf(fid, 'predicted_sin_slope,%.6f\n', predicted_sin_slope);
fprintf(fid, 'predicted_cos_slope,%.6f\n', predicted_cos_slope);
fprintf(fid, 'predicted_avg_slope,%.6f\n', predicted_avg_slope);
fprintf(fid, 'defl_factor_sin_direction,%.6f\n', defl_sin);
fprintf(fid, 'defl_factor_cos_direction,%.6f\n', defl_cos);
fprintf(fid, 'fisher_cond_nj_2,%.6g\n', condition_numbers(1));
fprintf(fid, 'fisher_cond_nj_10,%.6g\n', condition_numbers(2));
fprintf(fid, 'fisher_det_nj_2,%.6g\n', det_info(1));
fprintf(fid, 'fisher_det_nj_10,%.6g\n', det_info(2));
for ki = 1:n_kappa
    for mi = 1:n_meth
        fprintf(fid, 'detect_age3_kappa%d_%s,%.6f\n', ...
            mc.kappa_list(ki), mc.methods{mi}, detect_rate(ki, mi));
    end
end
fclose(fid);
fprintf('Saved %s\n', csv_path);

% =====================================================================
% Helper functions
% =====================================================================

function L = subj_loglik(y_i, eta_i, sigma_phi, kappa, gh_x, gh_w)
% Marginal log-likelihood of one subject's observations under
% phi_i ~ N(0, sigma_phi^2), y_ij ~ vM(eta_ij + phi_i, kappa).
% Uses Gauss-Hermite quadrature.
%
% After change of variables phi = sqrt(2) sigma_phi z,
% integral of N(0, sigma_phi^2) f(phi) dphi
%   = (1/sqrt(pi)) * sum_k w_k f(sqrt(2) sigma_phi z_k).
%
% Use log-sum-exp for numerical stability.

n_j = length(y_i);
log_2pi_I0 = log(2 * pi) + log(besseli(0, kappa));   % normalization per obs

phi_grid = sqrt(2) * sigma_phi * gh_x(:);
log_w_h  = log(gh_w(:));

logvals = zeros(length(phi_grid), 1);
for k = 1:length(phi_grid)
    phi = phi_grid(k);
    arg = y_i - eta_i - phi;
    sum_log_p = sum(kappa * cos(arg)) - n_j * log_2pi_I0;
    logvals(k) = log_w_h(k) + sum_log_p;
end
m = max(logvals);
L = m + log(sum(exp(logvals - m))) - 0.5 * log(pi);
end

function [x, w] = local_gauss_hermite(n)
% Physicists' Hermite Gauss quadrature nodes and weights, order n.
% Computes by Golub-Welsch on Jacobi matrix.
% w_k as in int_-inf^inf f(x) exp(-x^2) dx ~ sum_k w_k f(x_k).

i = 1:(n-1);
b = sqrt(i / 2);
J = diag(b, 1) + diag(b, -1);
[V, D] = eig(J);
[x, idx] = sort(diag(D));
V = V(:, idx);
w = sqrt(pi) * V(1, :).^2;
w = w(:);
end
