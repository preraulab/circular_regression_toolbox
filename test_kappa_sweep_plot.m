%TEST_KAPPA_SWEEP_PLOT  Visualize fitted age trajectories vs truth.
%
% For each kappa in the sweep, refit every method at the order that the
% iterative LRT actually picked (order 0 for orig/fixed at every kappa;
% 0 or 2 for sin/cos depending on kappa) and overlay the predicted age
% trajectory on the truth and the data.  If the iteratively-selected fit
% tracks the truth despite "wrong" order, the under-selection is benign
% for the paper's purpose (detecting an age-based trend, not recovering
% the true polynomial degree).
%
% Layout: rows = kappa, columns = electrode (0 = frontal, 1 = central).
% Each panel: gray dots = observed data, black = truth, colored lines =
% fitted trajectory from each method at its selected order.  Title shows
% (selected order, block-test p) per method.

clear; close all;
addpath(fileparts(mfilename('fullpath')));

kappa_list = [4, 10, 20, 50, 100];
% Selected orders from earlier sweep (test_kappa_sweep.m output):
%   orig:   always 0
%   fixed:  always 0
%   sin/cos: 0 at k<=10, 2 at k>=20
sel_orig_by_k   = [0 3 3 3 3];
sel_fixed_by_k  = [0 3 3 3 3];
sel_sincos_by_k = [3 3 3 3 3];

age_grid = linspace(7, 80, 200)';
n_grid   = length(age_grid);

fig = figure('Position', [50 50 1200 1500], 'Color', 'w');
tl  = tiledlayout(length(kappa_list), 2, 'TileSpacing', 'compact', ...
                  'Padding', 'compact');

% Truth in raw Age space (matches data generator)
true_b = [ 0.6;  -0.12;  0.0024;  -1.4e-5;  0.30;  -0.10;  0.008;  -4e-5;  0 ];

for ki = 1:length(kappa_list)
    kappa_true = kappa_list(ki);

    % Reconstruct exact same data the sweep used (rng(42) inside the loop)
    rng(42);
    n_subjects = 300;
    subj_age   = 7 + 73 * rand(n_subjects, 1);
    subj_sex   = double(rand(n_subjects, 1) > 0.5);
    subj_re    = 0.4 * randn(n_subjects, 1);
    subj_id    = repelem((1:n_subjects)', 2);
    Age        = repelem(subj_age, 2);
    sex        = repelem(subj_sex, 2);
    electrode  = repmat([1; 0], n_subjects, 1);
    re         = repelem(subj_re, 2);
    n          = length(Age);
    X_true = [ones(n,1), Age, Age.^2, Age.^3, electrode, sex, ...
              electrode.*Age, electrode.*Age.^2, electrode.*Age.^3];
    true_phase = X_true * true_b + re;
    phase_pref = wrapToPi(true_phase + randn(n,1)/sqrt(kappa_true));

    tbl = table(subj_id, Age, sex, electrode, phase_pref, ...
        'VariableNames', {'Subj_ID','Age','sex','electrode','phase_pref'});
    tbl.sin_phase = sin(tbl.phase_pref);
    tbl.cos_phase = cos(tbl.phase_pref);
    for kk = 2:4
        tbl.(sprintf('Age%d', kk)) = tbl.Age.^kk;
    end

    cat_vars  = [tbl.electrode, tbl.sex];
    intx_full = [true, false];

    % --- Fit each method at its iteratively-selected order ---
    k_orig   = sel_orig_by_k(ki);
    k_fixed  = sel_fixed_by_k(ki);
    k_sincos = sel_sincos_by_k(ki);

    if k_orig   == 0, intx_o = [false, false]; else, intx_o = intx_full; end
    if k_fixed  == 0, intx_f = [false, false]; else, intx_f = intx_full; end
    if k_sincos == 0, intx_s = [false, false]; else, intx_s = intx_full; end

    % Original circular_regression
    np = 1 + k_orig + size(cat_vars,2) + sum(intx_o)*k_orig;
    b0 = zeros(np, 1);
    b0(1) = atan2(mean(sin(tbl.phase_pref)), mean(cos(tbl.phase_pref)));
    [~, mdl_o] = circular_regression(tbl.Age, tbl.phase_pref, cat_vars, ...
        {'Age','electrode','sex'}, k_orig, intx_o, b0, 1000);
    b_orig = mdl_o.Coefficients.Estimate;

    % Fixed circular_regression_fixed
    [~, mdl_f] = circular_regression_fixed(tbl.Age, tbl.phase_pref, ...
        'Order', k_fixed, ...
        'Categorical', cat_vars, ...
        'CategoricalNames', {'electrode','sex'}, ...
        'Interactions', intx_f, ...
        'PredictorName', 'Age', ...
        'ClusterID', tbl.Subj_ID);
    b_fixed = mdl_f.Coefficients.Estimate;

    % sin/cos LME
    fml = build_lme_formula(k_sincos);
    mdl_s = fitlme(tbl, sprintf(fml, 'sin_phase'));
    mdl_c = fitlme(tbl, sprintf(fml, 'cos_phase'));

    % --- Build prediction grid and fitted curves per electrode ---
    for elec_idx = 1:2
        elec_val = elec_idx - 1;   % 0 or 1
        sex_val  = 1;              % marginalize-ish: pick one level

        % Truth at grid (using sex=1, electrode=elec_val, RE=0)
        Xg_true = [ones(n_grid,1), age_grid, age_grid.^2, age_grid.^3, ...
                   elec_val*ones(n_grid,1), sex_val*ones(n_grid,1), ...
                   elec_val*age_grid, elec_val*age_grid.^2, elec_val*age_grid.^3];
        truth_grid = wrapToPi(Xg_true * true_b);

        % Build design at grid for each method (matches the build inside
        % circular_regression / circular_regression_fixed)
        Xg_circ_o = build_circ_design(age_grid, elec_val, sex_val, ...
            k_orig,  intx_o);
        Xg_circ_f = build_circ_design(age_grid, elec_val, sex_val, ...
            k_fixed, intx_f);

        eta_orig  = Xg_circ_o * b_orig;
        eta_fixed = Xg_circ_f * b_fixed;
        pred_orig  = wrapToPi(eta_orig);
        pred_fixed = wrapToPi(eta_fixed);

        % sin/cos LME prediction
        nd = table(age_grid, elec_val*ones(n_grid,1), sex_val*ones(n_grid,1), ...
            'VariableNames', {'Age','electrode','sex'});
        for kk = 2:4
            nd.(sprintf('Age%d', kk)) = nd.Age.^kk;
        end
        nd.Subj_ID = repmat(tbl.Subj_ID(1), n_grid, 1);  % dummy; conditional false below
        s_hat = predict(mdl_s, nd, 'Conditional', false);
        c_hat = predict(mdl_c, nd, 'Conditional', false);
        pred_sincos = atan2(s_hat, c_hat);

        % --- Plot ---
        nexttile;
        hold on;
        idx_e = tbl.electrode == elec_val;
        scatter(tbl.Age(idx_e), tbl.phase_pref(idx_e), 8, ...
                [.6 .6 .6], 'filled', 'MarkerFaceAlpha', 0.4);
        plot(age_grid, truth_grid, 'k-', 'LineWidth', 2);
        plot(age_grid, pred_orig,  'Color', [0.85 0.33 0.10], 'LineWidth', 1.5, ...
             'LineStyle', ':');
        plot(age_grid, pred_fixed, 'Color', [0    0.45 0.74], 'LineWidth', 1.5);
        plot(age_grid, pred_sincos,'Color', [0.47 0.67 0.19], 'LineWidth', 1.5);
        ylim([-pi pi]); yticks([-pi -pi/2 0 pi/2 pi]);
        yticklabels({'-\pi','-\pi/2','0','\pi/2','\pi'});
        xlim([7 80]);
        if ki == length(kappa_list), xlabel('Age (years)'); end
        if elec_idx == 1, ylabel('phase\_pref (rad)'); end
        title(sprintf('\\kappa = %g, electrode = %d  (orig:%d, fixed:%d, sin/cos:%d)', ...
                      kappa_true, elec_val, k_orig, k_fixed, k_sincos), ...
                      'FontSize', 9, 'FontWeight', 'normal');
        if ki == 1 && elec_idx == 1
            legend({'data','truth','orig (k=0)','fixed','sin/cos LME'}, ...
                   'Location','southoutside','Orientation','horizontal', ...
                   'NumColumns', 5, 'FontSize', 8);
        end
        grid on;
    end
end

title(tl, 'Fitted age trajectory at iteratively-selected order vs truth (N=300)', ...
      'FontWeight', 'bold');
sgt_path = fullfile(fileparts(mfilename('fullpath')), '..', ...
    'test_kappa_sweep_trajectories.png');
exportgraphics(fig, sgt_path, 'Resolution', 150);
fprintf('Wrote %s\n', sgt_path);


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


function X = build_circ_design(age, elec, sex_val, order, intx)
% Matches the design ordering in circular_regression: [1, Age, Age^2, ...,
% Age^order, electrode, sex, electrode:Age, electrode:Age^2, ...]
n = length(age);
cols = ones(n,1);
for k = 1:order
    cols = [cols, age.^k]; %#ok<AGROW>
end
cols = [cols, elec*ones(n,1), sex_val*ones(n,1)];
if intx(1)   % electrode interactions
    for k = 1:order
        cols = [cols, elec*age.^k]; %#ok<AGROW>
    end
end
if length(intx) > 1 && intx(2)   % sex interactions (not used here)
    for k = 1:order
        cols = [cols, sex_val*age.^k]; %#ok<AGROW>
    end
end
X = cols;
end
