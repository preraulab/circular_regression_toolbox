function per_method_summary(results_root, target_order, dump_dir)
%PER_METHOD_SUMMARY  Build one figure per method showing every
% (feature, cluster) panel that the sweep ran. Each subplot shows the
% data scatter (in original frame) + the method's central/frontal
% trajectory, with the +-2*pi wraparound trick applied so seam
% crossings render continuously.
%
%   per_method_summary()
%   per_method_summary(results_root, target_order, dump_dir)

if nargin < 1 || isempty(results_root)
    results_root = fullfile(fileparts(mfilename('fullpath')), 'results');
end
if nargin < 2 || isempty(target_order), target_order = 4; end
if nargin < 3 || isempty(dump_dir)
    dump_dir = '/Users/Mike/Desktop/phase_fit_dumps';
end

mfile = fullfile(results_root, sprintf('sweep_manifest_o%d.mat', target_order));
M = load(mfile);
runs = M.runs;
ok = arrayfun(@(r) r.success, runs);
runs = runs(ok);
n_panel = numel(runs);
fprintf('Drawing per-method summaries for %d panels at order %d\n', n_panel, target_order);

methods = {
    'fitcirc_lme',  'MATLAB fitcirc\_lme (vM-GLMM, EM)',           'matlab', 1;
    'fitlme_circ',  'MATLAB fitlme\_circ (sin/cos)',                'matlab', 2;
    'brms',         'R brms vM-GLMM (Stan, tan\_half link)',        'R',      3;
    'lme4',         'R lme4 sin/cos',                               'R',      4;
    'bpnreg',       'R bpnreg projected-normal (Stan)',             'R',      5};

color_central = [0.95 0.55 0.16];
color_frontal = [0.00 0.59 1.00];

[nr, nc] = best_subplot_grid(n_panel);

for mi = 1:size(methods,1)
    fig = figure('Position',[50 50 1700 1100],'Color','w', ...
                 'Name', sprintf('Per-method summary: %s', methods{mi,1}));
    sgt = sgtitle(fig, methods{mi,2}, 'Interpreter','tex','FontSize',16,'FontWeight','bold');

    for k = 1:n_panel
        run = runs(k);
        feat = run.combo{1}; cluster = run.combo{2};
        rd = run.results_dir;
        ax = subplot(nr, nc, k); hold(ax, 'on');

        % Pull this run's data + this method's predictions from disk
        [T, theta_shift, has_elec] = load_run_data(rd, dump_dir);
        if isempty(T)
            title(ax, sprintf('%s c=%d (no dump)', feat, cluster), 'Interpreter','none');
            continue
        end
        [yc, age_c, yc_lo, yc_hi, yf, age_f] = load_method_pred(rd, methods{mi,1}, methods{mi,3});

        % Data scatter (original frame, plotted thrice for wraparound)
        if has_elec
            scatter_circ(ax, T.Age(T.electrode==0), T.(feat)(T.electrode==0), color_central);
            scatter_circ(ax, T.Age(T.electrode==1), T.(feat)(T.electrode==1), color_frontal);
        else
            scatter_circ(ax, T.Age, T.(feat), [.5 .5 .5]);
        end

        % Method trajectory (also tripled with seam-jump break)
        if ~isempty(yc)
            plot_circ_line(ax, age_c, yc, color_central, '-', 1.8);
            if ~isempty(yc_lo)
                fill_ci_circ(ax, age_c, yc_lo, yc_hi, color_central);
            end
        end
        if ~isempty(yf)
            plot_circ_line(ax, age_f, yf, color_frontal, '--', 1.8);
        end

        xlim(ax, [5 85]); ylim(ax, [-pi pi]);
        yticks(ax, [-pi 0 pi]); yticklabels(ax, {'-\pi','0','\pi'});
        grid(ax, 'on');
        title(ax, sprintf('%s, cluster %d  (n=%d, %s=%+.2f)', ...
              pretty_feature_name(feat), cluster, height(T), '\theta_{shift}', theta_shift), ...
              'Interpreter','tex','FontSize',9);
        if mod(k-1, nc) == 0, ylabel(ax,'phase (rad)','FontSize',8); end
        if k > (nr-1)*nc,    xlabel(ax,'Age (years)','FontSize',8); end
    end

    out = fullfile(results_root, sprintf('per_method_%s_o%d.png', methods{mi,1}, target_order));
    set(fig,'Renderer','painters');
    exportgraphics(fig, out, 'Resolution', 130);
    fprintf('  wrote %s\n', out);
    system(sprintf('open %s', out));
end
end


function [T, theta_shift, has_elec] = load_run_data(rd, dump_dir)
% Load original-frame data table for this run from the dump file.
meta_path = fullfile(rd, 'meta.json');
T = []; theta_shift = 0; has_elec = false;
if ~exist(meta_path,'file'), return; end
meta = jsondecode(fileread(meta_path));
if ~isfield(meta,'dump') || ~exist(meta.dump,'file'), return; end
S = load(meta.dump);
T = S.tbl_full_save;
keep = ~isnan(T.Age) & ~isnan(T.(meta.feature));
T = T(keep, :);
if isfield(meta, 'theta_shift'), theta_shift = double(meta.theta_shift); end
has_elec = ismember('electrode', T.Properties.VariableNames) && ...
           numel(unique(T.electrode)) >= 2;
end


function [yc, age_c, yc_lo, yc_hi, yf, age_f] = load_method_pred(rd, method_key, source)
% Pull this method's central/frontal trajectory + CI from disk.
yc = []; age_c = []; yc_lo = []; yc_hi = []; yf = []; age_f = [];
switch method_key
    case 'fitcirc_lme'
        path = fullfile(rd, 'fitcirc_lme_predictions.csv');  col = 'mean';
    case 'fitlme_circ'
        path = fullfile(rd, 'fitlme_circ_predictions.csv');  col = 'mean';
    case 'brms'
        path = fullfile(rd, 'brms_predictions.csv');         col = 'mean';
    case 'lme4'
        path = fullfile(rd, 'lme4_predictions.csv');         col = 'yhat';
    case 'bpnreg'
        path = fullfile(rd, 'bpnreg_predictions.csv');       col = 'mean';
    otherwise, return
end
if ~exist(path, 'file'), return; end
P = readtable(path);
if ~ismember('electrode', P.Properties.VariableNames)
    % monolevel
    [age_c, ix] = sort(P.Age);
    yc = P.(col)(ix);
    if ismember('lo', P.Properties.VariableNames), yc_lo = P.lo(ix); end
    if ismember('hi', P.Properties.VariableNames), yc_hi = P.hi(ix); end
    return
end
elec_levels = unique(P.electrode);
mask0 = P.electrode == elec_levels(1);
[age_c, ix] = sort(P.Age(mask0));
yc = P.(col)(mask0); yc = yc(ix);
if ismember('lo', P.Properties.VariableNames)
    yc_lo = P.lo(mask0); yc_lo = yc_lo(ix);
    yc_hi = P.hi(mask0); yc_hi = yc_hi(ix);
end
if numel(elec_levels) >= 2
    mask1 = P.electrode == elec_levels(2);
    [age_f, ix] = sort(P.Age(mask1));
    yf = P.(col)(mask1); yf = yf(ix);
end
end


function [nr, nc] = best_subplot_grid(n)
% Pick a roughly-square grid that fits n panels.
nc = ceil(sqrt(n));
nr = ceil(n / nc);
end


function s = pretty_feature_name(feat)
switch lower(feat)
    case 'pref_phase', s = 'Phase Preference';
    case 'phase',      s = 'SO Phase';
    case 'theta',      s = 'Theta phase';
    case 'stdfreq',    s = 'STD frequency';
    case 'stdphase',   s = 'STD phase';
    otherwise,         s = strrep(feat, '_', ' ');
end
end


function w = wrap_pi(x)
w = ((x + pi) - 2*pi*floor((x + pi) / (2*pi))) - pi;
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
d  = abs(diff(y));
for j = sort(find(d > pi),'descend')'
    xo = [xo(1:j); NaN; xo(j+1:end)];
    yo = [yo(1:j); NaN; yo(j+1:end)];
end
end


function fill_ci_circ(ax, x, lo, hi, col)
mask = (hi - lo) <= pi;
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
break_after = false(numel(x),1);
break_after(end) = true;
if numel(x) >= 2
    big_gap = diff(x) > 1.5;
    seam_lo = abs(diff(lo)) > pi;
    seam_hi = abs(diff(hi)) > pi;
    break_after(1:end-1) = break_after(1:end-1) | big_gap | seam_lo | seam_hi;
end
ends   = find(break_after);
starts = [1; ends(1:end-1)+1];
runs = struct('x',{},'lo',{},'hi',{});
for k = 1:numel(starts)
    a = starts(k); b = ends(k);
    if b - a < 1, continue; end
    runs(end+1) = struct('x', x(a:b), 'lo', lo(a:b), 'hi', hi(a:b)); %#ok<AGROW>
end
end
