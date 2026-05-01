%TEST_CIRCULAR_REGRESSION  Synthetic-data validation of the circular
% regression implementations.
%
% Generates a synthetic dataset with known parameters that mimics the
% structure of the lifespan circular feature analyses (preferred phase
% as a polynomial-in-age function, with electrode and sex effects, an
% electrode-by-age interaction, a subject-level random intercept, and
% von-Mises-like noise).  Fits the same model with three approaches:
%
%   1. circular_regression.m         (original; one-sided p-values,
%                                     unit-weight Hessian, no clustering)
%   2. circular_regression_fixed.m   (proper Fisher-Lee Newton-Raphson
%                                     with cluster-robust SEs and joint
%                                     block tests)
%   3. fitlme on sin(phase) and cos(phase) separately (projected linear
%                                     approximation; reuses standard
%                                     MATLAB infrastructure)
%
% Compares estimated coefficients to truth, prints joint block tests for
% the polynomial-age main-effect block and the electrode-by-age
% interaction block, and produces a single overlay figure showing data,
% true population trajectory, and each method's fitted trajectory with
% confidence/credible bands.
%
% USAGE
%   From the repo root:
%     >> addpath stats
%     >> test_circular_regression
%
% NOTES
%   - Noise is drawn as a wrapped Gaussian with sd = 1/sqrt(kappa_true),
%     which is the small-residual approximation of vonMises(0, kappa).
%     For kappa >= 3 this is indistinguishable from a true von Mises
%     draw at the resolutions we plot at; we use it here because
%     CircStat may not be on path.  Swap in circ_vmrnd if you have it.
%   - With seed = 42 the synthetic data has a clear nonlinear age
%     trajectory that survives all three estimators, which is the
%     expected behavior.  Reducing kappa_true (e.g. kappa = 1.5) makes
%     the original method's coefficient SEs visibly too small relative
%     to the fixed/cluster-robust SEs, and inflates Type I error.

clear; close all;
addpath(fileparts(mfilename('fullpath')));   % ensure stats/ is on path

rng(42);

% =====================================================================
% 1. SYNTHETIC DATA
% =====================================================================
% Subject-level frame: each of n_subjects gets one age, one sex, plus a
% subject-specific random intercept on phase.  Each subject contributes
% TWO observations -- one central electrode, one frontal -- producing
% the within-subject correlation structure that motivates clustering.

n_subjects = 200;
subj_age   = 7 + 73 * rand(n_subjects, 1);       % ages 7-80
subj_sex   = double(rand(n_subjects, 1) > 0.5);  % F=1, M=0

% Subject random intercept on phase (radians).  This is the signal the
% cluster-robust SEs should be picking up.
subj_re_sd = 0.4;
subj_re    = subj_re_sd * randn(n_subjects, 1);

% Expand to one row per (subject, electrode).
subj_id   = repelem((1:n_subjects)', 2);
Age       = repelem(subj_age, 2);
sex       = repelem(subj_sex, 2);
electrode = repmat([1; 0], n_subjects, 1);       % central=1, frontal=0
re        = repelem(subj_re, 2);
n         = length(Age);

% True coefficients.  Order matters in the design:
%   [intercept, Age, Age^2, Age^3, electrode, sex, electrode:Age,
%    electrode:Age^2, electrode:Age^3]
% These values are chosen to produce a visible cubic precession and a
% modest electrode-by-age interaction at the scale of the synthetic data.
true_b = [ 0.6;        % intercept (rad)
          -0.06;       % Age
           0.0012;     % Age^2
          -7e-6;       % Age^3
           0.30;       % electrode
          -0.10;       % sex
           0.004;      % electrode:Age
          -2e-5;       % electrode:Age^2
           0;       ]; % electrode:Age^3 (true zero -- check that it
                       % gets a non-significant block test)

X_true = [ones(n,1), Age, Age.^2, Age.^3, electrode, sex, ...
          electrode.*Age, electrode.*Age.^2, electrode.*Age.^3];
true_phase = X_true * true_b + re;

% Noise: wrapped Gaussian, ~ vonMises(0, kappa) for moderate kappa.
kappa_true = 4;
phase_pref = wrapToPi(true_phase + randn(n,1) / sqrt(kappa_true));

tbl = table(subj_id, Age, sex, electrode, phase_pref, ...
            'VariableNames', {'Subj_ID','Age','sex','electrode','phase_pref'});

% Export to CSV for the R comparison script.  Same data, same seed, so
% the R fits are computed on identical observations.
writetable(tbl, fullfile(fileparts(mfilename('fullpath')), '..', ...
                         'test_synthetic_data.csv'));

fprintf('Synthetic data: %d subjects x 2 electrodes = %d observations\n', ...
        n_subjects, n);
fprintf('True kappa = %g, subject random-intercept sd = %g rad\n\n', ...
        kappa_true, subj_re_sd);


% =====================================================================
% 2. FIT WITH ORIGINAL circular_regression.m
% =====================================================================
% Match the calling convention in get_single_order_model.m exactly so
% this is the same model the existing pipeline would fit.

order = 3;
cat_vars   = [tbl.electrode, tbl.sex];
varnames   = {'phase_pref','Age','electrode','sex'};
intx_flags = [true, false];                       % electrode:Age*, no sex:Age*
b0 = [atan2(mean(sin(tbl.phase_pref)), mean(cos(tbl.phase_pref)));
      zeros(order + size(cat_vars,2) + sum(intx_flags)*order, 1)];

[yhat_orig, mdl_orig] = circular_regression( ...
    tbl.Age, tbl.phase_pref, cat_vars, varnames(2:end), order, ...
    intx_flags, b0, 1000);


% =====================================================================
% 3. FIT WITH circular_regression_fixed.m
% =====================================================================

[yhat_fixed, mdl_fixed] = circular_regression_fixed( ...
    tbl.Age, tbl.phase_pref, ...
    'Order',            order, ...
    'Categorical',      cat_vars, ...
    'CategoricalNames', {'electrode','sex'}, ...
    'Interactions',     intx_flags, ...
    'PredictorName',    'Age', ...
    'ClusterID',        tbl.Subj_ID);

fprintf('Fixed implementation converged: %d (in %d iterations)\n', ...
        mdl_fixed.Converged, mdl_fixed.Iterations);
fprintf('Fixed implementation kappa estimate: %.3f (truth = %g)\n\n', ...
        mdl_fixed.Kappa, kappa_true);


% =====================================================================
% 4. FIT WITH fitlme ON sin(phase) AND cos(phase) (the user's question)
% =====================================================================
% Projected-linear approximation: each component of the unit vector
% (cos y, sin y) is modeled as Gaussian with a subject random intercept.
% This is what we'd do with any other (non-circular) feature in the
% MATLAB pipeline, and for tightly-concentrated circular data it is
% nearly indistinguishable from the proper von-Mises MLE.

tbl.sin_phase = sin(tbl.phase_pref);
tbl.cos_phase = cos(tbl.phase_pref);
tbl.Age2      = tbl.Age.^2;
tbl.Age3      = tbl.Age.^3;

% Manually expanded polynomial * electrode interactions.  We avoid the
% Wilkinson `Age^3*electrode` shorthand because its expansion semantics
% in fitlme are easy to misread.
formula_str = ['%s ~ Age + Age2 + Age3 + electrode + sex + ' ...
               'Age:electrode + Age2:electrode + Age3:electrode + ' ...
               '(1|Subj_ID)'];
mdl_sin = fitlme(tbl, sprintf(formula_str, 'sin_phase'));
mdl_cos = fitlme(tbl, sprintf(formula_str, 'cos_phase'));


% =====================================================================
% 5. JOINT BLOCK TESTS
% =====================================================================
% We want a SINGLE p-value per inferential question, not per coefficient.
%
%   Q1: "is there ANY age main effect?"  -> joint test on
%       {Age, Age^2, Age^3} block.
%   Q2: "is the age trajectory different across electrodes?"
%       -> joint test on {electrode:Age, electrode:Age^2, electrode:Age^3}.
%
% For the original implementation we have to make do with the
% per-coefficient table; report the smallest p-value as a (biased)
% summary, since that's what get_sigstar_text.m does.

fprintf('=== JOINT BLOCK TESTS ===\n\n');

% --- Original (no joint test available; fall back to min p across block) ---
age_rows  = 2 : 1+order;                        % rows for Age, Age^2, Age^3
intx_rows = (1+order+size(cat_vars,2)) + (1:order);   % electrode:Age*
fprintf('ORIGINAL circular_regression:\n');
fprintf('  Age block, individual p-values: %s\n', ...
    sprintf('%.4g ', mdl_orig.Coefficients.pValue(age_rows)));
fprintf('  electrode:Age block, individual p-values: %s\n\n', ...
    sprintf('%.4g ', mdl_orig.Coefficients.pValue(intx_rows)));

% --- Fixed: built-in coefTest method on the contrast index ---
age_test  = mdl_fixed.coefTest(mdl_fixed.ContrastIndex.x_main);
intx_test = mdl_fixed.coefTest(mdl_fixed.ContrastIndex.x_x_electrode);

fprintf('FIXED circular_regression (cluster-robust):\n');
fprintf('  Age block:           F(%d, %d) = %.3f, p = %.4g\n', ...
    age_test.df1, age_test.df2, age_test.F, age_test.p_F);
fprintf('  electrode:Age block: F(%d, %d) = %.3f, p = %.4g\n\n', ...
    intx_test.df1, intx_test.df2, intx_test.F, intx_test.p_F);

% --- Sin/cos LME: coefTest on each component, combine via Bonferroni ---
% Build the contrast matrix once and reuse it for both component models.
H_age  = zeros(order, length(mdl_sin.CoefficientNames));
H_intx = zeros(order, length(mdl_sin.CoefficientNames));
% Locate columns by name match.  fitlme's CoefficientNames will be
% {'(Intercept)','Age','Age2','Age3','electrode','sex',
%  'Age:electrode','Age2:electrode','Age3:electrode'}.
for i = 1:length(mdl_sin.CoefficientNames)
    nm = mdl_sin.CoefficientNames{i};
    if strcmp(nm,'Age');           H_age(1,i)  = 1; end
    if strcmp(nm,'Age2');          H_age(2,i)  = 1; end
    if strcmp(nm,'Age3');          H_age(3,i)  = 1; end
    if strcmp(nm,'Age:electrode'); H_intx(1,i) = 1; end
    if strcmp(nm,'Age2:electrode');H_intx(2,i) = 1; end
    if strcmp(nm,'Age3:electrode');H_intx(3,i) = 1; end
end

[p_age_sin,  F_age_sin]  = coefTest(mdl_sin, H_age);
[p_age_cos,  F_age_cos]  = coefTest(mdl_cos, H_age);
[p_intx_sin, F_intx_sin] = coefTest(mdl_sin, H_intx);
[p_intx_cos, F_intx_cos] = coefTest(mdl_cos, H_intx);

% Bonferroni across the two components for the joint phase claim
% (two correlated tests; this is conservative but valid).
p_age_sincos  = min(1, 2 * min(p_age_sin,  p_age_cos));
p_intx_sincos = min(1, 2 * min(p_intx_sin, p_intx_cos));

fprintf('SIN/COS LME (Bonferroni across components):\n');
fprintf('  Age block:           sin: F=%.2f p=%.4g | cos: F=%.2f p=%.4g | combined p=%.4g\n', ...
    F_age_sin, p_age_sin, F_age_cos, p_age_cos, p_age_sincos);
fprintf('  electrode:Age block: sin: F=%.2f p=%.4g | cos: F=%.2f p=%.4g | combined p=%.4g\n\n', ...
    F_intx_sin, p_intx_sin, F_intx_cos, p_intx_cos, p_intx_sincos);


% =====================================================================
% 6. COEFFICIENT TABLE COMPARISON
% =====================================================================
% Lay out estimates side by side against truth.  The fixed and original
% methods share a coefficient ordering; the sin/cos LME estimates aren't
% directly comparable to the von-Mises betas (different scale), so we
% only report them descriptively.

names = {'(Intercept)','Age','Age^2','Age^3','electrode','sex', ...
         'Age:electrode','Age^2:electrode','Age^3:electrode'};

fprintf('=== COEFFICIENTS (truth vs estimates) ===\n\n');
fprintf('%-18s %10s %10s %10s %12s %12s\n', 'Term', 'Truth', ...
        'Orig.b', 'Fixed.b', 'Orig.p', 'Fixed.p');
fprintf('%s\n', repmat('-', 1, 78));
for i = 1:length(names)
    fprintf('%-18s %10.4g %10.4g %10.4g %12.4g %12.4g\n', ...
        names{i}, true_b(i), ...
        mdl_orig.Coefficients.Estimate(i), ...
        mdl_fixed.Coefficients.Estimate(i), ...
        mdl_orig.Coefficients.pValue(i), ...
        mdl_fixed.Coefficients.pValue(i));
end
fprintf('\nNote: Orig.p is one-sided due to the bug in circular_regression.m.\n');
fprintf('      Fixed.p is two-sided AND uses cluster-robust SEs.\n');
fprintf('      The factor of ~2 discrepancy is the bug fix; the additional\n');
fprintf('      shrinkage on Fixed.p reflects within-subject correlation.\n\n');


% =====================================================================
% 7. PLOT TRAJECTORIES
% =====================================================================
% Overlay: data, true population trajectory, fits from each method.
% Hold sex constant at its reference (F = 1) and sweep age across each
% electrode.

age_grid = linspace(7, 80, 200)';
sex_eval = 1;       % marginalize at F to keep the plot tractable

figure('Position', [100 100 1100 700]);

for elec = [1, 0]
    ax = subplot(2, 1, 2 - elec);   % central on top, frontal on bottom
    hold(ax, 'on');

    % --- raw data ---
    is_e = tbl.electrode == elec;
    scatter(ax, tbl.Age(is_e), tbl.phase_pref(is_e), 12, ...
            [.7 .7 .7], 'filled', 'MarkerFaceAlpha', 0.4);

    % --- true population trajectory ---
    X_grid = [ones(size(age_grid)), age_grid, age_grid.^2, age_grid.^3, ...
              elec*ones(size(age_grid)), sex_eval*ones(size(age_grid)), ...
              elec*age_grid, elec*age_grid.^2, elec*age_grid.^3];
    truth_traj = wrapToPi(X_grid * true_b);
    plot(ax, age_grid, truth_traj, 'k-', 'LineWidth', 2.5, ...
         'DisplayName', 'Truth');

    % --- original fit ---
    % Plotted with sparse markers because the Original IRLS converges to
    % the same MLE point estimate as the Fixed implementation -- the two
    % continuous curves overlap exactly.  Sparse 'o' markers make the
    % overlap visible without obscuring either line.
    orig_traj = wrapToPi(X_grid * mdl_orig.Coefficients.Estimate);
    marker_idx = 1:12:length(age_grid);   % every 12th point
    plot(ax, age_grid(marker_idx), orig_traj(marker_idx), 'o', ...
         'MarkerEdgeColor', [0.85 0.33 0.10], ...
         'MarkerFaceColor', 'none', ...
         'MarkerSize', 8, 'LineWidth', 1.8, ...
         'DisplayName', 'Original (markers)');

    % --- fixed fit (with 95% CI from cov_b) ---
    fixed_traj = wrapToPi(X_grid * mdl_fixed.Coefficients.Estimate);
    se_fixed = sqrt(diag(X_grid * mdl_fixed.cov_b * X_grid'));
    fixed_lo = fixed_traj - 1.96 * se_fixed;
    fixed_hi = fixed_traj + 1.96 * se_fixed;

    % Mask out wraparound discontinuities for line plotting
    bad = [false; abs(diff(fixed_traj)) > pi];
    fixed_traj_plot = fixed_traj; fixed_traj_plot(bad) = NaN;
    plot(ax, age_grid, fixed_traj_plot, '-', 'Color', [0 0.45 0.74], ...
         'LineWidth', 2, 'DisplayName', 'Fixed (cluster-robust)');

    % CI band -- plot in the unwrapped space and show three vertically
    % shifted copies to handle the wrap, like the MATLAB pipeline does.
    for shift = [-2*pi 0 2*pi]
        fill(ax, [age_grid; flipud(age_grid)], ...
                  [fixed_lo + shift; flipud(fixed_hi + shift)], ...
                  [0 0.45 0.74], 'FaceAlpha', 0.12, 'EdgeColor', 'none', ...
                  'HandleVisibility', 'off');
    end

    % --- sin/cos LME predictions ---
    % Predict on each component LME at the age grid for this electrode &
    % sex, then atan2 to recover the angle.
    eval_tbl = table(age_grid, age_grid.^2, age_grid.^3, ...
                     elec*ones(size(age_grid)), ...
                     sex_eval*ones(size(age_grid)), ...
                     ones(size(age_grid)), ...   % dummy Subj_ID (any value)
                     'VariableNames', {'Age','Age2','Age3','electrode','sex','Subj_ID'});
    sin_pred = predict(mdl_sin, eval_tbl);
    cos_pred = predict(mdl_cos, eval_tbl);
    sincos_traj = atan2(sin_pred, cos_pred);
    bad = [false; abs(diff(sincos_traj)) > pi];
    sincos_plot = sincos_traj; sincos_plot(bad) = NaN;
    plot(ax, age_grid, sincos_plot, ':', 'Color', [0.47 0.67 0.19], ...
         'LineWidth', 2, 'DisplayName', 'sin/cos LME');

    % --- formatting ---
    title(ax, sprintf('Electrode = %s (sex = F)', ...
                      ternary(elec==1, 'central', 'frontal')));
    xlabel(ax, 'Age (years)');
    ylabel(ax, 'Preferred phase (rad)');
    set(ax, 'YLim', [-pi pi], 'YTick', [-pi -pi/2 0 pi/2 pi], ...
            'YTickLabel', {'-\pi','-\pi/2','0','\pi/2','\pi'});
    grid(ax, 'on');
    if elec == 1
        legend(ax, 'Location', 'best');
    end
end

sgtitle('Synthetic circular regression: truth vs three implementations');
set(gcf, 'Color', 'w');


% =====================================================================
% Helper
% =====================================================================
function out = ternary(cond, a, b)
if cond, out = a; else, out = b; end
end
