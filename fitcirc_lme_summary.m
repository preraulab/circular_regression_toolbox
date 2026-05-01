function fitcirc_lme_summary(results_root, target_order, dump_dir)
%FITCIRC_LME_SUMMARY  Per-method figure dedicated to MATLAB fitcirc_lme.
% Each subplot shows the (variance-min shifted) data + central/frontal
% trajectory + 95% Wald CI ribbons. Title carries Wald-block significance
% stars for: Age polynomial, sex, electrode (main + Age*electrode
% interactions).
%
%   fitcirc_lme_summary(results_root, target_order, dump_dir)

if nargin < 1 || isempty(results_root)
    results_root = fullfile(fileparts(mfilename('fullpath')), 'results');
end
if nargin < 2 || isempty(target_order), target_order = 2; end
if nargin < 3 || isempty(dump_dir)
    dump_dir = '/Users/Mike/Desktop/phase_fit_dumps';
end

% Resolve results_root to an absolute path so manifest entries (which
% may be stored relatively) are dereferenced correctly.
if ~java.io.File(results_root).isAbsolute()
    results_root = char(java.io.File(results_root).getCanonicalPath());
end

mfile = fullfile(results_root, sprintf('sweep_manifest_o%d.mat', target_order));
M = load(mfile);
runs = M.runs;
ok = arrayfun(@(r) r.success, runs);
runs = runs(ok);
% Re-anchor each run's results_dir to the (now absolute) results_root,
% in case the manifest was written with a relative path.
for k = 1:numel(runs)
    [~, base] = fileparts(runs(k).results_dir);
    runs(k).results_dir = fullfile(results_root, base);
end
n_panel = numel(runs);

color_central = [0.95 0.55 0.16];
color_frontal = [0.00 0.59 1.00];
[nr, nc] = best_grid(n_panel);

fig = figure('Position',[50 50 1700 1100],'Color','w', ...
             'Name', sprintf('MATLAB fitcirc_lme (vM-GLMM, EM) — order %d', target_order));
sgtitle(fig, sprintf('\\bf{MATLAB fitcirc\\_lme}   order %d   stars: Age block / sex / electrode (main + Age*elec)', target_order), ...
        'Interpreter','tex','FontSize',15);

for k = 1:n_panel
    run = runs(k);
    feat = run.combo{1}; cluster = run.combo{2};

    % Load dump, drop NaNs, apply variance-min shift, refit.
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
    mdl = fitcirc_lme(T_shifted, formula_full);

    % Wald block p-values
    p_age  = block_pvalue(mdl, 'x_main');
    p_sex  = single_coef_pvalue(mdl, 'sex');
    p_elec = electrode_block_pvalue(mdl);

    % Predict + asymptotic CI on the eval grid
    x_eval = (7:80)';
    [yc, yc_lo, yc_hi] = predict_with_ci(mdl, build_eval(T, x_eval, 0, has_elec, has_sex), theta_shift);
    if has_elec
        [yf, yf_lo, yf_hi] = predict_with_ci(mdl, build_eval(T, x_eval, 1, has_elec, has_sex), theta_shift);
    else
        yf = []; yf_lo = []; yf_hi = [];
    end

    % Decide y-limits: if data + CIs fit comfortably inside a small
    % window, zoom in so the trend and CI bands are visible. Otherwise
    % keep the full circular range and use the wraparound rendering.
    y_extent_data = max(abs(T.(feat)));
    y_extent_pred = max([abs(yc(:)); abs(yc_lo(:)); abs(yc_hi(:)); ...
                         abs(yf(:)); abs(yf_lo(:)); abs(yf_hi(:))]);
    y_extent = max(y_extent_data, y_extent_pred);
    zoom_panel = y_extent < (pi/3);  % ~60 deg; gives a clear view

    % --- Render panel ---
    ax = subplot(nr, nc, k); hold(ax,'on');
    if zoom_panel
        if has_elec
            scatter(ax, T.Age(T.electrode==0), T.(feat)(T.electrode==0), 4, color_central, 'filled', 'MarkerFaceAlpha',0.2,'HandleVisibility','off');
            scatter(ax, T.Age(T.electrode==1), T.(feat)(T.electrode==1), 4, color_frontal, 'filled', 'MarkerFaceAlpha',0.2,'HandleVisibility','off');
            fill_ci_flat(ax, x_eval, yc_lo, yc_hi, color_central);
            plot(ax, x_eval, yc, '-',  'Color', color_central, 'LineWidth', 1.8, 'HandleVisibility','off');
            fill_ci_flat(ax, x_eval, yf_lo, yf_hi, color_frontal);
            plot(ax, x_eval, yf, '--', 'Color', color_frontal, 'LineWidth', 1.8, 'HandleVisibility','off');
        else
            scatter(ax, T.Age, T.(feat), 4, [.5 .5 .5], 'filled', 'MarkerFaceAlpha',0.2,'HandleVisibility','off');
            fill_ci_flat(ax, x_eval, yc_lo, yc_hi, color_central);
            plot(ax, x_eval, yc, '-', 'Color', color_central, 'LineWidth', 1.8, 'HandleVisibility','off');
        end
        ypad = max(0.05, 1.2 * y_extent);
        ylim(ax, [-ypad ypad]);
        yticks(ax, 'auto');
    else
        if has_elec
            scatter_circ(ax, T.Age(T.electrode==0), T.(feat)(T.electrode==0), color_central);
            scatter_circ(ax, T.Age(T.electrode==1), T.(feat)(T.electrode==1), color_frontal);
            fill_ci_circ(ax, x_eval, yc_lo, yc_hi, color_central);
            plot_circ_line(ax, x_eval, yc, color_central, '-',  1.8);
            fill_ci_circ(ax, x_eval, yf_lo, yf_hi, color_frontal);
            plot_circ_line(ax, x_eval, yf, color_frontal, '--', 1.8);
        else
            scatter_circ(ax, T.Age, T.(feat), [.5 .5 .5]);
            fill_ci_circ(ax, x_eval, yc_lo, yc_hi, color_central);
            plot_circ_line(ax, x_eval, yc, color_central, '-', 1.8);
        end
        ylim(ax,[-pi pi]);
        yticks(ax,[-pi 0 pi]); yticklabels(ax,{'-\pi','0','\pi'});
    end

    xlim(ax,[5 85]);
    grid(ax,'on');
    title_str = sprintf('%s, cluster %d   Age%s sex%s elec%s', ...
        pretty_feature_name(feat), cluster, ...
        stars(p_age), stars(p_sex), stars(p_elec));
    title(ax, title_str, 'Interpreter','tex','FontSize',9);
    if mod(k-1, nc) == 0, ylabel(ax,'phase (rad)','FontSize',8); end
    if k > (nr-1)*nc,    xlabel(ax,'Age (years)','FontSize',8); end
end

out = fullfile(results_root, sprintf('fitcirc_lme_summary_o%d.png', target_order));
set(fig,'Renderer','painters');
exportgraphics(fig, out, 'Resolution', 130);
fprintf('  wrote %s\n', out);
system(sprintf('open %s', out));
end


function p = block_pvalue(mdl, key)
ci = mdl.ContrastIndex;
if ~isfield(ci, key) || isempty(ci.(key))
    p = NaN; return;
end
res = mdl.coefTest(ci.(key));
p = res.pValue;
end


function p = single_coef_pvalue(mdl, name)
% Find the single coefficient whose name equals `name` (exact match) or
% startsWith name. Returns NaN if not found.
T = mdl.Coefficients;
nm = string(T.Name);
hit = nm == string(name);
if ~any(hit)
    hit = startsWith(nm, name);
end
if ~any(hit)
    p = NaN; return;
end
p = T.pValue(find(hit, 1));
end


function p = electrode_block_pvalue(mdl)
% Joint Wald test combining the electrode main coefficient AND its
% Age*electrode interaction coefficients. Falls back to whatever exists.
ci = mdl.ContrastIndex;
T  = mdl.Coefficients;
nm = string(T.Name);
idx = [];
elec_main = find(nm == "electrode");
if ~isempty(elec_main), idx(end+1,1) = elec_main(1); end
% Pull interaction indices out of x_x_electrode
fld = 'x_x_electrode';
if isfield(ci, fld), idx = [idx; ci.(fld)(:)]; end
% Fallback: any name containing 'electrode'
if isempty(idx)
    idx = find(contains(nm, "electrode"));
end
idx = unique(idx);
if isempty(idx)
    p = NaN; return;
end
res = mdl.coefTest(idx);
p = res.pValue;
end


function s = stars(p)
if isnan(p),       s = '_n/a'; return; end
if p < 1e-4,       s = '****';   return; end
if p < 1e-3,       s = '***';    return; end
if p < 1e-2,       s = '**';     return; end
if p < 0.05,       s = '*';      return; end
s = '_{ns}';
end


function [yhat, lo, hi] = predict_with_ci(mdl, nd, theta_shift)
% Predict on nd in shifted frame, then unshift. CI is +/- 1.96 * SE on
% the linear predictor (angle scale). All three quantities are wrapped
% to (-pi, pi] before return. Where the half-width exceeds pi/2 (i.e.,
% the CI covers more than half the circle), uncertainty is essentially
% total — return NaN so downstream rendering skips those points.
X_new = design_matrix_for_newdata(mdl, nd);
yhat_shifted = X_new * mdl.Beta;
se = sqrt(max(0, sum((X_new * mdl.cov_b) .* X_new, 2)));
half = 1.96 * se;
bad  = half > (pi/2);

lo_shifted = yhat_shifted - half;
hi_shifted = yhat_shifted + half;
yhat = wrap_pi(yhat_shifted + theta_shift);
lo   = wrap_pi(lo_shifted   + theta_shift);
hi   = wrap_pi(hi_shifted   + theta_shift);
lo(bad) = NaN; hi(bad) = NaN;
end


function X_new = design_matrix_for_newdata(mdl, nd)
% Replicate the trick used inside fitcirc_lme.predict: combine training
% data with new rows, run fitlme to build the design, take only the
% new rows' columns. This handles categorical predictors and any
% missing columns by filling with reference values.
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
    case 'pref_phase', s = 'Phase Preference';
    case 'phase',      s = 'SO Phase';
    case 'theta',      s = 'Theta phase';
    otherwise,         s = strrep(feat, '_', ' ');
end
end


function scatter_circ(ax, x, y, col)
scatter(ax, x, y,        4, col, 'filled', 'MarkerFaceAlpha', 0.15, 'HandleVisibility','off');
scatter(ax, x, y + 2*pi, 4, col, 'filled', 'MarkerFaceAlpha', 0.15, 'HandleVisibility','off');
scatter(ax, x, y - 2*pi, 4, col, 'filled', 'MarkerFaceAlpha', 0.15, 'HandleVisibility','off');
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


function fill_ci_flat(ax, x, lo, hi, col)
% Non-wrapping CI ribbon for zoomed-in panels. lo/hi are already on the
% original (unshifted) angle scale; we don't need the +/-2pi triple plot.
x = x(:); lo = lo(:); hi = hi(:);
mask = isfinite(lo) & isfinite(hi) & ((hi - lo) <= pi);
x = x(mask); lo = lo(mask); hi = hi(mask);
if numel(x) < 2, return; end
xring = [x; flipud(x)];
yring = [lo; flipud(hi)];
fill(ax, xring, yring, col, 'EdgeColor','none', 'FaceAlpha', 0.18, 'HandleVisibility','off');
end


function fill_ci_circ(ax, x, lo, hi, col)
mask = isfinite(lo) & isfinite(hi) & ((hi - lo) <= pi);
if ~any(mask), return; end
x = x(mask); lo = lo(mask); hi = hi(mask);
runs = split_runs(x, lo, hi);
for r = 1:numel(runs)
    xr = runs(r).x; lr = runs(r).lo; hr = runs(r).hi;
    if numel(xr) < 2, continue; end
    xring = [xr; flipud(xr)];
    yring = [lr; flipud(hr)];
    fill(ax, xring, yring,        col, 'EdgeColor','none', 'FaceAlpha', 0.13, 'HandleVisibility','off');
    fill(ax, xring, yring + 2*pi, col, 'EdgeColor','none', 'FaceAlpha', 0.13, 'HandleVisibility','off');
    fill(ax, xring, yring - 2*pi, col, 'EdgeColor','none', 'FaceAlpha', 0.13, 'HandleVisibility','off');
end
end


function runs = split_runs(x, lo, hi)
x = x(:); lo = lo(:); hi = hi(:);
break_after = false(numel(x),1); break_after(end) = true;
if numel(x) >= 2
    big_gap = diff(x) > 1.5;
    seam_lo = abs(diff(lo)) > pi;
    seam_hi = abs(diff(hi)) > pi;
    break_after(1:end-1) = break_after(1:end-1) | big_gap | seam_lo | seam_hi;
end
ends = find(break_after); starts = [1; ends(1:end-1)+1];
runs = struct('x',{},'lo',{},'hi',{});
for k = 1:numel(starts)
    a = starts(k); b = ends(k);
    if b - a < 1, continue; end
    runs(end+1) = struct('x', x(a:b), 'lo', lo(a:b), 'hi', hi(a:b)); %#ok<AGROW>
end
end
