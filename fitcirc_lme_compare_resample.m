function fitcirc_lme_compare_resample(results_root, target_order, B, keep_frac)
%FITCIRC_LME_COMPARE_RESAMPLE  Three-way overlay of legacy / cluster-bootstrap-median /
% subsample-80%-median fitcirc_lme trajectories on real preferred-phase data.
% Prints per-panel angular L2 displacement of each resampled trajectory
% from legacy, and writes a side-by-side overlay PNG.
%
%   fitcirc_lme_compare_resample(results_root, target_order, B, keep_frac)
%
%   B          : number of resamples (default 80)
%   keep_frac  : fraction of subjects retained for the without-replacement
%                subsample (default 0.8)

if nargin < 1 || isempty(results_root)
    results_root = fullfile(fileparts(mfilename('fullpath')), 'results');
end
if nargin < 2 || isempty(target_order), target_order = 3; end
if nargin < 3 || isempty(B),            B = 80; end
if nargin < 4 || isempty(keep_frac),    keep_frac = 0.8; end

if ~java.io.File(results_root).isAbsolute()
    results_root = char(java.io.File(results_root).getCanonicalPath());
end

mfile = fullfile(results_root, sprintf('sweep_manifest_o%d.mat', target_order));
M = load(mfile);
runs = M.runs;
ok = arrayfun(@(r) r.success, runs);
runs = runs(ok);
for k = 1:numel(runs)
    [~, base] = fileparts(runs(k).results_dir);
    runs(k).results_dir = fullfile(results_root, base);
end
n_panel = numel(runs);

color_legacy = [0.20 0.20 0.20];   % dashed black
color_cboot  = [0.85 0.20 0.20];   % solid red
color_sub80  = [0.20 0.45 0.85];   % solid blue
[nr, nc] = best_grid(n_panel);

fig = figure('Position',[60 60 1700 1100],'Color','w', ...
             'Name', sprintf('fitcirc_lme legacy vs cluster-boot vs subsample — order %d', target_order));
sgtitle(fig, sprintf(['\\bf{fitcirc\\_lme}  order %d   ' ...
    '\\color[rgb]{0.2,0.2,0.2}black = legacy   ' ...
    '\\color[rgb]{0.85,0.2,0.2}red = cluster-boot median (B=%d)   ' ...
    '\\color[rgb]{0.2,0.45,0.85}blue = subsample-%d%% median   ' ...
    '\\color[rgb]{0,0,0}|   solid = central electrode, dashed = frontal'], ...
    target_order, B, round(100*keep_frac)), ...
    'Interpreter','tex','FontSize',12);

x_eval = (7:80)';
rng(11);  % deterministic resample picks

for k = 1:n_panel
    run = runs(k);
    feat = run.combo{1}; cluster = run.combo{2};

    meta = jsondecode(fileread(fullfile(run.results_dir, 'meta.json')));
    S = load(meta.dump);
    T = S.tbl_full_save;
    keep = ~isnan(T.Age) & ~isnan(T.(feat));
    if ismember('electrode', T.Properties.VariableNames), keep = keep & ~isnan(T.electrode); end
    if ismember('sex',       T.Properties.VariableNames), keep = keep & ~isnan(T.sex); end
    T = T(keep,:);
    has_elec = ismember('electrode', T.Properties.VariableNames) && numel(unique(T.electrode))>=2;
    has_sex  = ismember('sex',       T.Properties.VariableNames) && numel(unique(T.sex))>=2;
    T.Subj_ID = double(T.Subj_ID);

    [theta_shift, ~] = circ_shift_min_var(T.(feat));
    T_shifted = T;
    T_shifted.(feat) = wrap_pi(T.(feat) - theta_shift);
    formula_full = build_formula(meta.feature, meta.order, has_elec, has_sex);

    % --- Legacy ---
    m0 = fitcirc_lme(T_shifted, formula_full);
    yc_legacy = predict_central(m0, build_eval(T, x_eval, 0, has_elec, has_sex), theta_shift);
    if has_elec
        yf_legacy = predict_central(m0, build_eval(T, x_eval, 1, has_elec, has_sex), theta_shift);
    end

    % --- Cluster bootstrap ---
    Yc_boot = zeros(numel(x_eval), B);
    Yf_boot = zeros(numel(x_eval), B);
    subj   = unique(T_shifted.Subj_ID);
    n_subj = numel(subj);
    feat_name = meta.feature;
    t0 = tic;
    for b = 1:B
        pick   = subj(randi(n_subj, n_subj, 1));
        T_b    = resample_subjects(T_shifted, pick, feat_name);
        m      = fitcirc_lme(T_b, formula_full);
        Yc_boot(:, b) = predict_central(m, build_eval(T, x_eval, 0, has_elec, has_sex), theta_shift);
        if has_elec
            Yf_boot(:, b) = predict_central(m, build_eval(T, x_eval, 1, has_elec, has_sex), theta_shift);
        end
    end
    yc_cboot = circ_median_columns(Yc_boot);
    if has_elec, yf_cboot = circ_median_columns(Yf_boot); end
    t_cboot = toc(t0);

    % --- Subsample 80% ---
    n_keep = round(keep_frac * n_subj);
    Yc_sub = zeros(numel(x_eval), B);
    Yf_sub = zeros(numel(x_eval), B);
    t0 = tic;
    for b = 1:B
        pick = subj(randperm(n_subj, n_keep));
        T_s  = T_shifted(ismember(T_shifted.Subj_ID, pick), :);
        m    = fitcirc_lme(T_s, formula_full);
        Yc_sub(:, b) = predict_central(m, build_eval(T, x_eval, 0, has_elec, has_sex), theta_shift);
        if has_elec
            Yf_sub(:, b) = predict_central(m, build_eval(T, x_eval, 1, has_elec, has_sex), theta_shift);
        end
    end
    yc_sub80 = circ_median_columns(Yc_sub);
    if has_elec, yf_sub80 = circ_median_columns(Yf_sub); end
    t_sub = toc(t0);

    d_cboot = wrap_pi(yc_cboot - yc_legacy);
    d_sub80 = wrap_pi(yc_sub80 - yc_legacy);
    rmsd_cboot = sqrt(mean(d_cboot.^2));
    rmsd_sub80 = sqrt(mean(d_sub80.^2));
    fprintf('  panel %2d %s c%d   cboot RMSD=%.3f rad (%.0fs)   sub80 RMSD=%.3f rad (%.0fs)\n', ...
        k, feat, cluster, rmsd_cboot, t_cboot, rmsd_sub80, t_sub);

    % Channel-coded scatter so the per-channel structure is visible.
    color_central = [0.95 0.55 0.16];
    color_frontal = [0.00 0.59 1.00];
    ax = subplot(nr, nc, k); hold(ax,'on');
    if has_elec
        scatter_circ(ax, T.Age(T.electrode==0), T.(feat)(T.electrode==0), color_central);
        scatter_circ(ax, T.Age(T.electrode==1), T.(feat)(T.electrode==1), color_frontal);
    else
        scatter_circ(ax, T.Age, T.(feat), [.7 .7 .7]);
    end
    % Solid = central electrode trajectory; dashed = frontal. Color = method.
    plot_circ_line(ax, x_eval, yc_legacy, color_legacy, '-',  1.4);
    plot_circ_line(ax, x_eval, yc_cboot,  color_cboot,  '-',  1.6);
    plot_circ_line(ax, x_eval, yc_sub80,  color_sub80,  '-',  1.6);
    if has_elec
        plot_circ_line(ax, x_eval, yf_legacy, color_legacy, '--', 1.4);
        plot_circ_line(ax, x_eval, yf_cboot,  color_cboot,  '--', 1.6);
        plot_circ_line(ax, x_eval, yf_sub80,  color_sub80,  '--', 1.6);
    end

    ylim(ax,[-pi pi]); yticks(ax,[-pi 0 pi]); yticklabels(ax,{'-\pi','0','\pi'});
    xlim(ax,[5 85]); grid(ax,'on');
    title(ax, sprintf('%s c%d  \\Delta_{cb}=%.2f  \\Delta_{s80}=%.2f', ...
        pretty_feature_name(feat), cluster, rmsd_cboot, rmsd_sub80), ...
        'Interpreter','tex','FontSize',9);
    if mod(k-1, nc) == 0, ylabel(ax,'phase (rad)','FontSize',8); end
    if k > (nr-1)*nc,    xlabel(ax,'Age (years)','FontSize',8); end
end

out = fullfile(results_root, sprintf('fitcirc_lme_compare_resample_o%d.png', target_order));
set(fig,'Renderer','painters');
exportgraphics(fig, out, 'Resolution', 130);
fprintf('  wrote %s\n', out);
system(sprintf('open %s', out));
end


%% ===== local helpers =====

function T_out = resample_subjects(T, pick, feat_name) %#ok<INUSD>
% With-replacement bootstrap: each pick gets a fresh unique Subj_ID so
% duplicates are independent clusters in the random-intercept model.
parts = cell(numel(pick),1);
next_id = 0;
for k = 1:numel(pick)
    rows = T(T.Subj_ID == pick(k), :);
    next_id = next_id + 1;
    rows.Subj_ID = repmat(next_id, height(rows), 1);
    parts{k} = rows;
end
T_out = vertcat(parts{:});
end


function ymed = circ_median_columns(Y)
% Componentwise circular median across columns of Y. Uses
% atan2(median(sin), median(cos)) — robust enough at this scale.
ymed = atan2(median(sin(Y), 2), median(cos(Y), 2));
end


function yhat = predict_central(mdl, nd, theta_shift)
X_new = design_matrix_for_newdata(mdl, nd);
yhat_shifted = X_new * mdl.Beta;
yhat = wrap_pi(yhat_shifted + theta_shift);
end

function X_new = design_matrix_for_newdata(mdl, nd)
base = mdl.TrainingData;
nd2  = nd;
missing_cols = setdiff(base.Properties.VariableNames, nd2.Properties.VariableNames);
for c = missing_cols
    cn = c{1};
    if iscategorical(base.(cn))
        nd2.(cn) = repmat(base.(cn)(1), height(nd2), 1);
    elseif iscell(base.(cn))
        nd2.(cn) = repmat(base.(cn)(1), height(nd2), 1);
    else
        nd2.(cn) = zeros(height(nd2), 1, 'like', base.(cn));
    end
end
nd2 = nd2(:, base.Properties.VariableNames);
combined = [base; nd2];
tmp = fitlme(combined, mdl.Formula);
X_all = designMatrix(tmp, 'Fixed');
X_new = X_all(height(base)+1:end, :);
end

function nd = build_eval(T, x_eval, electrode_val, has_elec, has_sex)
nd = table();
nd.Age = x_eval;
if has_elec, nd.electrode = electrode_val * ones(numel(x_eval),1); end
if has_sex,  nd.sex       = zeros(numel(x_eval),1); end
nd.Subj_ID = double(repmat(T.Subj_ID(1), numel(x_eval), 1));
end

function fml = build_formula(feat, ord, has_elec, has_sex)
if ord == 0, rhs = '1'; else, rhs = sprintf('Age^%d', ord); end
if has_elec, rhs = ['electrode * ' rhs]; end
if has_sex,  rhs = [rhs ' + sex']; end
fml = sprintf('%s ~ %s + (1|Subj_ID)', feat, rhs);
end

function w = wrap_pi(x)
w = ((x + pi) - 2*pi*floor((x + pi) / (2*pi))) - pi;
end

function [nr, nc] = best_grid(n)
nc = ceil(sqrt(n)); nr = ceil(n / nc);
end

function s = pretty_feature_name(feat)
switch lower(feat)
    case 'pref_phase', s = 'PrefPhase';
    case 'phase',      s = 'SOPhase';
    case 'theta',      s = 'Theta';
    otherwise,         s = strrep(feat, '_', ' ');
end
end

function scatter_circ(ax, x, y, col)
scatter(ax, x, y,        4, col, 'filled', 'MarkerFaceAlpha', 0.12, 'HandleVisibility','off');
scatter(ax, x, y + 2*pi, 4, col, 'filled', 'MarkerFaceAlpha', 0.12, 'HandleVisibility','off');
scatter(ax, x, y - 2*pi, 4, col, 'filled', 'MarkerFaceAlpha', 0.12, 'HandleVisibility','off');
end

function plot_circ_line(ax, x, y, col, style, lw)
[xb, yb] = break_at_jumps(x(:), y(:));
plot(ax, xb, yb,        style, 'Color', col, 'LineWidth', lw, 'HandleVisibility','off');
plot(ax, xb, yb + 2*pi, style, 'Color', col, 'LineWidth', lw, 'HandleVisibility','off');
plot(ax, xb, yb - 2*pi, style, 'Color', col, 'LineWidth', lw, 'HandleVisibility','off');
end

function [xo, yo] = break_at_jumps(x, y)
xo = x; yo = y;
d = abs(diff(y));
for j = sort(find(d > pi),'descend')'
    xo = [xo(1:j); NaN; xo(j+1:end)];
    yo = [yo(1:j); NaN; yo(j+1:end)];
end
end
