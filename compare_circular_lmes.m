%COMPARE_CIRCULAR_LMES  Side-by-side: fitlme_circ vs fitcirc_lme (vs brms).
%
% Fits the same circular mixed-effects model three different ways on
% the doubled-trend kappa=20 synthetic dataset and produces:
%
%   - A summary table of fixed-effect coefficient estimates and joint
%     p-values from each method, alongside the truth.
%   - A trajectory plot overlaying each method's population-level fit
%     against truth, per electrode.
%   - A scatter of per-subject phase random effects from fitcirc_lme
%     vs the implied phase shift from fitlme_circ.
%   - Numerical comparison of kappa, sigma_phi, and goodness-of-fit on
%     the circular scale (residual kappa).
%
% brms results are read from a CSV that the companion R script
% (export_brms_summary.R) writes after pulling posterior summaries
% from the cached fit.  If that CSV does not exist, the brms columns
% are skipped with a warning.

clear; close all;
addpath(fileparts(mfilename('fullpath')));

% =====================================================================
% Load doubled-trend kappa=20 data (test_kappa_sweep.m wrote it).
% =====================================================================
csv = fullfile(fileparts(mfilename('fullpath')), '..', 'test_kappa_sweep_K20.csv');
tbl = readtable(csv);
tbl.Subj_ID = categorical(tbl.Subj_ID);
for kk = 2:4
    if ~ismember(sprintf('Age%d', kk), tbl.Properties.VariableNames)
        tbl.(sprintf('Age%d', kk)) = tbl.Age.^kk;
    end
end

% Doubled-trend truth (matches test_kappa_sweep.m)
true_b_full = [ 0.6;  -0.12;  0.0024;  -1.4e-5;  0.30;  -0.10;  0.008;  -4e-5;  0 ];
% Position layout: [intercept; Age; Age^2; Age^3; electrode; sex;
%                   electrode:Age; electrode:Age^2; electrode:Age^3]
% Map to fitlme term names (sex stays in even though it's no-trend)
true_named = struct( ...
    'Intercept',         true_b_full(1), ...
    'Age',               true_b_full(2), ...
    'Age2',              true_b_full(3), ...
    'Age3',              true_b_full(4), ...
    'electrode',         true_b_full(5), ...
    'sex',               true_b_full(6), ...
    'Age_electrode',     true_b_full(7), ...
    'Age2_electrode',    true_b_full(8), ...
    'Age3_electrode',    true_b_full(9));

% Common formula: order-3 polynomial in Age, electrode + sex,
% electrode:Age interactions, subject random intercept.
fml = ['phase_pref ~ 1 + Age + Age2 + Age3 + electrode + sex + ' ...
       'Age:electrode + Age2:electrode + Age3:electrode + (1|Subj_ID)'];

fprintf('\n========================================================\n');
fprintf('  Fitting three circular mixed-effects models\n');
fprintf('========================================================\n');

% =====================================================================
% Method 1: fitlme_circ (sin/cos LME pair, Bonferroni joint inference)
% =====================================================================
tic;
mdl1 = fitlme_circ(tbl, fml);
t1 = toc;
fprintf('\n[1] fitlme_circ                        fit in %.2f s\n', t1);
fprintf('    coefficient names: %s\n', strjoin(mdl1.CoefficientNames, ' | '));

% =====================================================================
% Method 2: fitcirc_lme (von Mises GLMM with EM + Laplace)
% =====================================================================
tic;
mdl2 = fitcirc_lme(tbl, fml, 'MaxIter', 200, 'Verbose', false);
t2 = toc;
fprintf('[2] fitcirc_lme                         fit in %.2f s  ', t2);
fprintf('(converged in %d EM iters)\n', mdl2.ConvergedIn);

% =====================================================================
% Method 3 (optional): brms posterior summary if cached
% =====================================================================
brms_csv = fullfile(fileparts(mfilename('fullpath')), 'R_outputs_kappa', ...
                    'brms_K20_order3_summary.csv');
have_brms = isfile(brms_csv);
if have_brms
    brms_tbl = readtable(brms_csv);
    fprintf('[3] brms (cached, kappa=20, order=3)    summary loaded from %s\n', brms_csv);
else
    fprintf('[3] brms summary CSV not found at %s\n', brms_csv);
    fprintf('    Run export_brms_summary.R to generate it.\n');
end

% =====================================================================
% Build a combined coefficients table.  Names won't match exactly
% across methods because of MATLAB vs brms term naming; we line them up
% by hand here.
% =====================================================================
% Term keys for the combined table (using fitlme_circ's display names)
% MATLAB sorts interaction term order alphabetically: e.g. Age2:electrode
% prints as electrode:Age2.  Use the actual sorted-form names.
terms = {'(Intercept)','Age','Age2','Age3','electrode','sex', ...
         'Age:electrode','electrode:Age2','electrode:Age3'};
truth_lookup = struct( ...
    'x_Intercept_',    true_b_full(1), ...
    'Age',             true_b_full(2), ...
    'Age2',            true_b_full(3), ...
    'Age3',            true_b_full(4), ...
    'electrode',       true_b_full(5), ...
    'sex',             true_b_full(6), ...
    'Age_electrode',   true_b_full(7), ...
    'electrode_Age2',  true_b_full(8), ...
    'electrode_Age3',  true_b_full(9));

n_terms = length(terms);
truth_vec      = zeros(n_terms,1);
sin_est        = nan(n_terms,1); sin_p = nan(n_terms,1);
cos_est        = nan(n_terms,1); cos_p = nan(n_terms,1);
joint_p        = nan(n_terms,1);
glmm_est       = nan(n_terms,1); glmm_p = nan(n_terms,1);
brms_est       = nan(n_terms,1); brms_lo = nan(n_terms,1); brms_hi = nan(n_terms,1);

c1 = mdl1.Coefficients;
c2 = mdl2.Coefficients;
truth_keys = fieldnames(truth_lookup);
for ii = 1:n_terms
    nm = terms{ii};

    % Match truth via cleaned-up key (':' -> '_', '(...)' -> 'x_..._')
    key = strrep(strrep(nm, ':', '_'), '(', 'x_'); key = strrep(key, ')', '_');
    if isfield(truth_lookup, key)
        truth_vec(ii) = truth_lookup.(key);
    end

    % Match into fitlme_circ table (Name field)
    [tf, idx] = ismember(nm, c1.Name);
    if tf
        sin_est(ii) = c1.EstSin(idx);
        sin_p(ii)   = c1.pSin(idx);
        cos_est(ii) = c1.EstCos(idx);
        cos_p(ii)   = c1.pCos(idx);
        joint_p(ii) = c1.pJointBonf(idx);
    end

    % Match into fitcirc_lme table
    [tf, idx] = ismember(nm, c2.Name);
    if tf
        glmm_est(ii) = c2.Estimate(idx);
        glmm_p(ii)   = c2.pValue(idx);
    end

    if have_brms
        % brms term names from posterior_summary use a "b_" prefix and
        % occasionally different separators; try a few variants.
        candidates = {nm, ['b_' nm], strrep(nm, ':', '.'), ...
                      ['b_' strrep(nm, ':', '.')]};
        for c = 1:length(candidates)
            [tf, idx] = ismember(candidates{c}, brms_tbl.term);
            if tf
                brms_est(ii) = brms_tbl.mean(idx);
                brms_lo(ii)  = brms_tbl.q2_5(idx);
                brms_hi(ii)  = brms_tbl.q97_5(idx);
                break
            end
        end
    end
end

T = table(terms', truth_vec, sin_est, sin_p, cos_est, cos_p, joint_p, ...
          glmm_est, glmm_p, brms_est, brms_lo, brms_hi, ...
    'VariableNames', {'term','truth', ...
                      'sincos_EstSin','sincos_pSin', ...
                      'sincos_EstCos','sincos_pCos', ...
                      'sincos_pJoint', ...
                      'glmm_Estimate','glmm_p', ...
                      'brms_mean','brms_q2_5','brms_q97_5'});
fprintf('\n--- Coefficient comparison ---\n');
disp(T);

% =====================================================================
% Variance-component comparison
% =====================================================================
fprintf('--- Variance components ---\n');
fprintf('  fitlme_circ residual kappa:        %.2f  (no single sigma_phi)\n', mdl1.ResidualKappa);
fprintf('  fitlme_circ RE SD on sin:          %.3f\n', mdl1.RandomEffectSD_Sin{1});
fprintf('  fitlme_circ RE SD on cos:          %.3f\n', mdl1.RandomEffectSD_Cos{1});
fprintf('  fitcirc_lme  kappa:                %.2f  (truth = 20)\n', mdl2.Kappa);
fprintf('  fitcirc_lme  sigma_phi:            %.3f  (truth = 0.4)\n', mdl2.SigmaPhi);
if have_brms
    [~, ki] = ismember('kappa', brms_tbl.term);
    [~, si] = ismember('sd_Subj_ID__Intercept', brms_tbl.term);
    if ki>0
        fprintf('  brms kappa posterior mean:         %.2f  (95%% CrI %.2f - %.2f)\n', ...
            brms_tbl.mean(ki), brms_tbl.q2_5(ki), brms_tbl.q97_5(ki));
    end
    if si>0
        fprintf('  brms sigma_phi (sd_Subj_ID):       %.3f  (95%% CrI %.3f - %.3f)\n', ...
            brms_tbl.mean(si), brms_tbl.q2_5(si), brms_tbl.q97_5(si));
    end
end

% =====================================================================
% Trajectory plot
% =====================================================================
age_grid = linspace(7, 80, 200)';
n_grid   = length(age_grid);

fig = figure('Position', [50 50 1100 700], 'Color', 'w');
set(fig, 'DefaultAxesFontSize', 16);
set(fig, 'DefaultTextFontSize', 16);
tl  = tiledlayout(2, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

% Truth trajectories (electrode 0 and 1, sex=1)
truth0 = wrapToPi(true_b_full(1) + true_b_full(2)*age_grid + ...
                  true_b_full(3)*age_grid.^2 + true_b_full(4)*age_grid.^3 + ...
                  true_b_full(5)*0 + true_b_full(6)*1);
truth1 = wrapToPi(true_b_full(1) + true_b_full(2)*age_grid + ...
                  true_b_full(3)*age_grid.^2 + true_b_full(4)*age_grid.^3 + ...
                  true_b_full(5)*1 + true_b_full(6)*1 + ...
                  true_b_full(7)*age_grid + true_b_full(8)*age_grid.^2 + ...
                  true_b_full(9)*age_grid.^3);

for elec = 0:1
    nexttile;
    nd = table(age_grid, age_grid.^2, age_grid.^3, ...
               elec*ones(n_grid,1), ones(n_grid,1), ...
               repmat(tbl.Subj_ID(1), n_grid, 1), ...
        'VariableNames', {'Age','Age2','Age3','electrode','sex','Subj_ID'});

    pred1 = mdl1.predict(nd, 'Conditional', false);
    pred2 = mdl2.predict(nd, 'Conditional', false);

    if elec == 0, truth = truth0; else, truth = truth1; end

    idx_e = (tbl.electrode == elec);
    hold on;
    scatter(tbl.Age(idx_e), tbl.phase_pref(idx_e), 12, [.6 .6 .6], ...
            'filled', 'MarkerFaceAlpha', 0.4);
    plot(age_grid, truth, 'k-', 'LineWidth', 2);
    plot(age_grid, pred1, 'Color', [0    0.45 0.74], 'LineWidth', 1.6);
    plot(age_grid, pred2, 'Color', [0.85 0.33 0.10], 'LineWidth', 1.6);
    legend_labels = {'data','truth','fitlme\_circ (sin/cos)','fitcirc\_lme (vM EM)'};
    if have_brms
        brms_traj = readtable(fullfile(fileparts(mfilename('fullpath')), ...
            'R_outputs_kappa', 'brms_K20_order3_trajectory.csv'));
        bidx = (brms_traj.electrode == elec);
        plot(brms_traj.Age(bidx), brms_traj.median(bidx), ...
             'Color', [0.47 0.67 0.19], 'LineWidth', 1.6, 'LineStyle', '--');
        legend_labels{end+1} = 'brms (vM Bayes, posterior median)';
    end
    ylim([-pi pi]); yticks([-pi -pi/2 0 pi/2 pi]);
    yticklabels({'-\pi','-\pi/2','0','\pi/2','\pi'});
    xlim([7 80]);
    ylabel('phase\_pref (rad)');
    title(sprintf('Electrode = %d', elec), 'FontWeight', 'normal');
    if elec == 0
        legend(legend_labels, 'Location', 'southeast');
    end
    grid on;
end
xlabel('Age (years)');
title(tl, '\kappa = 20, doubled trend, N=300: population-level fits vs truth', ...
      'FontWeight', 'bold');
trajec_path = fullfile(fileparts(mfilename('fullpath')), '..', ...
    'compare_circular_lmes_trajectory.png');
exportgraphics(fig, trajec_path, 'Resolution', 150);
fprintf('\nWrote %s\n', trajec_path);

% =====================================================================
% Per-subject random-effect comparison
% =====================================================================
% fitcirc_lme: phi_hat is a single phase per subject (radians).
% fitlme_circ: random effects are (b_s, b_c) on (sin, cos) plane per subject.
%   Effective phase shift for subject i: rotate the population (sin, cos)
%   prediction by atan2(s_i + b_s_i, c_i + b_c_i) - atan2(s_i, c_i),
%   evaluated at the population mean predictor.  We use the average
%   age * average electrode * sex=1 design row as a reference.
%
% Quick approximation: the small-RE limit gives
%     phi_eff_i ~= b_s_i * cos(theta_pop) - b_c_i * sin(theta_pop)
% where theta_pop is the population mean angle.

[B_s, ~, ~] = randomEffects(mdl1.Sin);
[B_c, ~, ~] = randomEffects(mdl1.Cos);
% randomEffects returns one entry per random-effect coefficient per
% group; with (1|Subj_ID) that's exactly one per subject in subject order.

% Compute population-level mean angle to linearize around
pop_S = mean(predict(mdl1.Sin));
pop_C = mean(predict(mdl1.Cos));
theta_pop = atan2(pop_S, pop_C);
phi_eff_sincos = B_s * cos(theta_pop) - B_c * sin(theta_pop);

phi_glmm = mdl2.PhiHat;
% Match by subject order; both methods use the same Subj_ID factor levels
% so positions match.
fprintf('\n--- Random-effect comparison ---\n');
fprintf('  fitcirc_lme  sigma_phi:                  %.3f  (truth = 0.4)\n', ...
        mdl2.SigmaPhi);
fprintf('  fitlme_circ  effective phase RE SD:      %.3f\n', std(phi_eff_sincos));
fprintf('  Pearson correlation between phi_GLMM and phi_eff_sincos:  %.3f\n', ...
        corr(phi_glmm, phi_eff_sincos));

fig2 = figure('Position', [50 50 700 700], 'Color', 'w');
hold on;
plot(phi_eff_sincos, phi_glmm, 'o', 'Color', [0.2 0.2 0.2], ...
     'MarkerFaceColor', [0.2 0.4 0.7], 'MarkerSize', 5);
lim = max(abs([phi_eff_sincos; phi_glmm])) * 1.05;
plot([-lim lim], [-lim lim], 'k--');
xlim([-lim lim]); ylim([-lim lim]);
axis square; grid on;
xlabel('Effective phase RE from fitlme\_circ (rad)');
ylabel('\phi_i from fitcirc\_lme (rad)');
title(sprintf(['Per-subject phase RE: sin/cos LME vs vM GLMM\n' ...
               'Pearson r = %.3f'], corr(phi_glmm, phi_eff_sincos)));
re_path = fullfile(fileparts(mfilename('fullpath')), '..', ...
    'compare_circular_lmes_re.png');
exportgraphics(fig2, re_path, 'Resolution', 150);
fprintf('Wrote %s\n', re_path);

% =====================================================================
% Save summary table
% =====================================================================
out_csv = fullfile(fileparts(mfilename('fullpath')), '..', ...
    'compare_circular_lmes_summary.csv');
writetable(T, out_csv);
fprintf('Wrote %s\n', out_csv);
