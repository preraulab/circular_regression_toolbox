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
% Electrode level 0 is drawn solid, level 1 dashed (if present). Sex
% level 0 keeps the result's base color; sex level 1 (if present) is
% drawn in a lighter variant of the same color so a single fit produces
% two visually distinguishable curves when sex is in the formula.

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

% --- data scatter (triple-plotted across the +-2*pi seam) ---
% When the table carries a `sex` column, points are colored to match
% the trajectory shades (sex=0 in the result's base color, sex=1 in
% the lighter variant) so the scatter and the curves are visually
% paired. Without a sex column the points are mid-grey.
if do_scat && ~isempty(tbl) && all(ismember({x_col, feature}, tbl.Properties.VariableNames))
    xs   = tbl.(x_col);
    ys   = tbl.(feature);
    good = ~isnan(xs) & ~isnan(ys);
    base_col = colors(1, :);
    if ismember('sex', tbl.Properties.VariableNames)
        sex_col = double(tbl.sex);
        light_col = 0.55 * base_col + 0.45 * [1 1 1];
        is_f = good & (sex_col == 0);
        is_m = good & (sex_col == 1);
        for off = [0, 2*pi, -2*pi]
            scatter(ax, xs(is_f), ys(is_f) + off, 14, base_col,  'filled', ...
                'MarkerFaceAlpha', 0.45, 'MarkerEdgeColor', 'none', ...
                'HandleVisibility', 'off');
            scatter(ax, xs(is_m), ys(is_m) + off, 14, light_col, 'filled', ...
                'MarkerFaceAlpha', 0.45, 'MarkerEdgeColor', 'none', ...
                'HandleVisibility', 'off');
        end
    else
        for off = [0, 2*pi, -2*pi]
            scatter(ax, xs(good), ys(good) + off, 12, [.35 .35 .35], 'filled', ...
                'MarkerFaceAlpha', 0.45, 'MarkerEdgeColor', 'none', ...
                'HandleVisibility', 'off');
        end
    end
end

% --- trajectories ---
leg_h     = gobjects(0);
leg_label = {};
for i = 1:N
    r  = results{i};
    Tr = r.Trajectory;
    base_col = colors(min(i,size(colors,1)), :);
    elecs    = unique(Tr.electrode(:))';
    sexes    = unique(Tr.sex(:))';
    for e = elecs
        for s = sexes
            sel = Tr.electrode == e & Tr.sex == s;
            if ~any(sel), continue; end
            x  = Tr.Age(sel);   [x, si] = sort(x);
            m  = Tr.mean(sel);  m  = m(si);
            lo = Tr.lo(sel);    lo = lo(si);
            hi = Tr.hi(sel);    hi = hi(si);
            % Electrode -> line style. Sex -> color shade (sex level 1
            % is a lighter variant of the base color so the two sex
            % curves of a single fit are distinguishable but clearly
            % related).
            style = '-'; if e == 1, style = '--'; end
            col   = base_col;
            if s == 1, col = 0.55 * base_col + 0.45 * [1 1 1]; end
            for off = [0, 2*pi, -2*pi]
                if plot_CI && any(hi ~= m)
                    patch(ax, [x(:); flipud(x(:))], [lo(:)+off; flipud(hi(:)+off)], col, ...
                        'FaceAlpha', 0.12, 'EdgeColor', 'none', 'HandleVisibility', 'off');
                end
                h = plot(ax, x, m + off, style, 'Color', col, 'LineWidth', 1.6, ...
                    'HandleVisibility', 'off');
                if off == 0
                    suffix = '';
                    if numel(sexes) > 1
                        if s == 0, suffix = ' (sex 0)';
                        else,      suffix = ' (sex 1)'; end
                    end
                    if numel(elecs) > 1
                        if e == 0, suffix = [suffix ' [elec 0]']; %#ok<AGROW>
                        else,      suffix = [suffix ' [elec 1]']; end %#ok<AGROW>
                    end
                    leg_h(end+1)     = h;            %#ok<AGROW>
                    leg_label{end+1} = [labels{i}, suffix]; %#ok<AGROW>
                end
            end
        end
    end
end

ylim(ax, [-pi pi]); yticks(ax, [-pi 0 pi]); yticklabels(ax, {'-\pi','0','\pi'});
xlabel(ax, x_col); ylabel(ax, sprintf('%s (rad)', strrep(feature,'_','\_')));
grid(ax, 'on');
valid = isgraphics(leg_h);
if any(valid)
    legend(ax, leg_h(valid), leg_label(valid), 'Location', 'best', 'Interpreter', 'none');
end
end


function v = getopt(opts, name, default)
if isfield(opts, name) && ~isempty(opts.(name))
    v = opts.(name);
else
    v = default;
end
end
