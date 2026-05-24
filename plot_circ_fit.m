function ax = plot_circ_fit(results, tbl, opts)
%PLOT_CIRC_FIT  Plot one or more circ_fit results from the uniform schema.
%
%   ax = plot_circ_fit(result, tbl)
%   ax = plot_circ_fit({r1, r2, ...}, tbl, opts)
%
% Draws each result's stored Trajectory (and CI band) using the triple-line
% trick — the curve is replicated at value, +2*pi, and -2*pi so it is
% continuous across the +-pi seam — and overlays multiple backends for the
% paper comparison figure. Trajectories are in the original (unshifted)
% angle frame, matching the raw data scatter.
%
% INPUTS
%   results  a single result struct (make_circ_result) or a cell array of them
%   tbl      data table for the scatter (needs x_col + the response; optional electrode)
%   opts     struct, all optional:
%     .ax        axes to draw into (default: new figure)
%     .feature   response column (default: results{1}.ResponseName)
%     .x_col     predictor column (default 'Age')
%     .plot_CI   draw CI bands (default true)
%     .scatter   draw the data scatter (default true)
%     .colors    Nx3 RGB rows, one per result (default: lines colormap)
%     .labels    cellstr legend labels (default: each result's Backend)
%
% Electrode level 0 is drawn solid, level 1 dashed (if present).

if ~iscell(results), results = {results}; end
N = numel(results);
if nargin < 3, opts = struct(); end

if isfield(opts,'ax') && ~isempty(opts.ax), ax = opts.ax; else, figure('Color','w'); ax = axes(); end
feature = getopt(opts, 'feature', results{1}.ResponseName);
x_col   = getopt(opts, 'x_col', 'Age');
plot_CI = getopt(opts, 'plot_CI', true);
do_scat = getopt(opts, 'scatter', true);
colors  = getopt(opts, 'colors', lines(max(N,1)));
labels  = getopt(opts, 'labels', cellfun(@(r) r.Backend, results, 'UniformOutput', false));

hold(ax, 'on');

% --- data scatter (faint, triple-plotted) ---
if do_scat && ~isempty(tbl) && all(ismember({x_col, feature}, tbl.Properties.VariableNames))
    xs = tbl.(x_col); ys = tbl.(feature);
    good = ~isnan(xs) & ~isnan(ys);
    for off = [0, 2*pi, -2*pi]
        scatter(ax, xs(good), ys(good) + off, 5, [.7 .7 .7], 'filled', ...
            'MarkerFaceAlpha', 0.12, 'HandleVisibility', 'off');
    end
end

% --- trajectories ---
leg_h = gobjects(N,1);
for i = 1:N
    r = results{i};
    Tr = r.Trajectory;
    col = colors(min(i,size(colors,1)), :);
    elecs = unique(Tr.electrode(:))';
    first_for_legend = true;
    for e = elecs
        sel = Tr.electrode == e;
        x  = Tr.Age(sel);   [x, si] = sort(x);
        m  = Tr.mean(sel);  m  = m(si);
        lo = Tr.lo(sel);    lo = lo(si);
        hi = Tr.hi(sel);    hi = hi(si);
        style = '-'; if e == 1, style = '--'; end
        for off = [0, 2*pi, -2*pi]
            if plot_CI && any(hi ~= m)
                patch(ax, [x(:); flipud(x(:))], [lo(:)+off; flipud(hi(:)+off)], col, ...
                    'FaceAlpha', 0.12, 'EdgeColor', 'none', 'HandleVisibility', 'off');
            end
            h = plot(ax, x, m + off, style, 'Color', col, 'LineWidth', 1.6, ...
                'HandleVisibility', 'off');
            if first_for_legend && off == 0, leg_h(i) = h; first_for_legend = false; end
        end
    end
end

ylim(ax, [-pi pi]); yticks(ax, [-pi 0 pi]); yticklabels(ax, {'-\pi','0','\pi'});
xlabel(ax, x_col); ylabel(ax, sprintf('%s (rad)', strrep(feature,'_','\_')));
grid(ax, 'on');
valid = isgraphics(leg_h);
if any(valid)
    legend(ax, leg_h(valid), labels(valid), 'Location', 'best', 'Interpreter', 'none');
end
end


function v = getopt(opts, name, default)
if isfield(opts, name) && ~isempty(opts.(name))
    v = opts.(name);
else
    v = default;
end
end
