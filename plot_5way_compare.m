function plot_5way_compare(dump_file, results_dir, out_png)
%PLOT_5WAY_COMPARE  Five-way circular-regression comparison panel:
%   1) MATLAB fitcirc_lme   (vM-GLMM, frequentist EM, vM(0,kphi) prior)
%   2) MATLAB fitlme_circ   (sin/cos parallel LMEs, frequentist projected-Gaussian)
%   3) R brms               (Bayesian vM-GLMM via Stan, identity link)
%   4) R lme4 sin/cos       (frequentist projected-Gaussian, parallel LMEs)
%   5) R bpnreg::bpnme      (Bayesian projected-normal, Stan)
%
% Plots central + frontal trajectories per method when electrode is in the
% data. Computes per-method goodness-of-fit summary stats (LL, circular
% R^2, mean angular error).

if nargin < 2, results_dir = fullfile(fileparts(dump_file), 'results'); end
if nargin < 3, out_png     = fullfile(results_dir, 'five_way_compare.png'); end

addpath(genpath('/Users/Mike/code/projects/trends_v_individual/stats'));

S = load(dump_file);
T = S.tbl_full_save;
meta = S.meta;
feat = meta.feature; ord = meta.order;

has_elec = ismember('electrode', T.Properties.VariableNames);
has_sex  = ismember('sex',       T.Properties.VariableNames);

% Drop NaNs
keep = ~isnan(T.Age) & ~isnan(T.(feat));
if has_elec, keep = keep & ~isnan(T.electrode); end
if has_sex,  keep = keep & ~isnan(T.sex); end
T = T(keep, :);

% Fall back if only one level present (rank-deficient design)
if has_elec && numel(unique(T.electrode)) < 2
    fprintf('plot_5way_compare: dropping electrode (only one level present)\n');
    has_elec = false;
end
if has_sex && numel(unique(T.sex)) < 2
    has_sex = false;
end

% Build the FULL formula in MATLAB Wilkinson notation
if ord == 0
    rhs = '1';
else
    rhs = sprintf('Age^%d', ord);
end
if has_elec, rhs = ['electrode * ' rhs]; end
if has_sex,  rhs = [rhs ' + sex']; end
formula_full = sprintf('%s ~ %s + (1|Subj_ID)', feat, rhs);
T.Subj_ID = double(T.Subj_ID);

fprintf('=== %s ===\n', feat);
fprintf('formula: %s\n', formula_full);
fprintf('n=%d  n_subj=%d  has_elec=%d\n', height(T), numel(unique(T.Subj_ID)), has_elec);

% --- MATLAB fits ---
% Apply variance-minimizing shift before fitting so the data is on a
% Cartesian-friendly frame (no seam crossings); unshift predictions
% afterward so plots and stats are reported in the original frame.
[theta_shift, ~] = circ_shift_min_var(T.(feat));
T_shifted = T;
T_shifted.(feat) = wrap_pi(T.(feat) - theta_shift);
fprintf('theta_shift = %+.3f rad   (variance-minimizing shift applied to all methods)\n', theta_shift);

m1 = fitcirc_lme(T_shifted, formula_full);
m2 = fitlme_circ(T_shifted, formula_full);

% Eval grid
x_eval = (7:80)';
nd_c = build_eval(T, x_eval, 0, has_elec, has_sex);  % central (or only level)
y1c = wrap_pi(m1.predict(nd_c, 'Conditional', false) + theta_shift);
y2c = wrap_pi(m2.predict(nd_c, 'Conditional', false) + theta_shift);
if has_elec
    nd_f = build_eval(T, x_eval, 1, has_elec, has_sex);
    y1f = wrap_pi(m1.predict(nd_f, 'Conditional', false) + theta_shift);
    y2f = wrap_pi(m2.predict(nd_f, 'Conditional', false) + theta_shift);
else
    nd_f = nd_c;  % no second electrode
    y1f = []; y2f = [];
end

% --- R predictions (graceful degradation if a method failed) ---
brms   = read_pred(fullfile(results_dir,'brms_predictions.csv'));
lme4   = read_pred(fullfile(results_dir,'lme4_predictions.csv'));
bpnreg = read_pred(fullfile(results_dir,'bpnreg_predictions.csv'));

% Decide electrode levels off whatever R outputs we have, else MATLAB grid
ref = brms; if isempty(ref), ref = lme4; end; if isempty(ref), ref = bpnreg; end
if ~isempty(ref)
    elec_levels = unique(ref.electrode);
else
    elec_levels = 0;
end
elec0 = elec_levels(1);

[y3c, brms_c_age, y3c_lo, y3c_hi] = pick_pred(brms, elec0, 'mean');
y4c                             = pick_pred(lme4,   elec0, 'yhat');
y5c                             = pick_pred(bpnreg, elec0, 'mean');

if numel(elec_levels) >= 2
    elec1 = elec_levels(2);
    [y3f, brms_f_age]           = pick_pred(brms, elec1, 'mean');
    y4f                         = pick_pred(lme4, elec1, 'yhat');
    y5f                         = pick_pred(bpnreg, elec1, 'mean');
else
    brms_f_age = []; y3f = []; y4f = []; y5f = [];
end

% --- Binned circular truth per electrode (or once if monolevel) ---
bins = 7.5:5:80;
bin_centers = (bins(1:end-1)+bins(2:end))/2;
if has_elec
    [mu_c, lo_c, hi_c] = binned_circ(T.Age(T.electrode==0), T.(feat)(T.electrode==0), bins);
    [mu_f, lo_f, hi_f] = binned_circ(T.Age(T.electrode==1), T.(feat)(T.electrode==1), bins);
else
    [mu_c, lo_c, hi_c] = binned_circ(T.Age, T.(feat), bins);
    mu_f = []; lo_f = []; hi_f = [];
end
R_overall = abs(mean(exp(1i*T.(feat))));

% --- Goodness-of-fit per method (training set, marginal predictions) ---
% Pass the shifted table to MATLAB methods so residuals are computed in
% the same frame the model was fit in; unshift not needed since
% angular residuals are rotation-invariant.
stats = goodness_of_fit_table(T_shifted, feat, m1, m2, results_dir);
stats_str = strjoin(arrayfun(@(k) sprintf('%-30s LL=%9.1f  R^2=%5.3f  MAE=%5.3f', ...
    stats.method{k}, stats.LL(k), stats.R2_circ(k), stats.mae_angular(k)), ...
    1:height(stats), 'uni', 0), '\n');
fprintf('\n--- Goodness of fit ---\n%s\n', stats_str);

% --- Plot ---
fig = figure('Position', [50 50 1700 950], 'Color', 'w');
ax = axes(fig, 'Position', [0.06 0.18 0.66 0.74]);   % leave room: legend right, stats below
hold(ax,'on');

color_central = [0.95 0.55 0.16];
color_frontal = [0.00 0.59 1.00];

% Data scatter
if has_elec
    scatter(ax, T.Age(T.electrode==0), T.(feat)(T.electrode==0), 5, color_central, 'filled', 'MarkerFaceAlpha', 0.15, 'DisplayName','data central');
    scatter(ax, T.Age(T.electrode==1), T.(feat)(T.electrode==1), 5, color_frontal, 'filled', 'MarkerFaceAlpha', 0.15, 'DisplayName','data frontal');
    plot_bin(ax, bin_centers, mu_c, lo_c, hi_c, color_central);
    plot_bin(ax, bin_centers, mu_f, lo_f, hi_f, color_frontal);
else
    scatter(ax, T.Age, T.(feat), 5, [.5 .5 .5], 'filled', 'MarkerFaceAlpha', 0.18, 'DisplayName','data');
    plot_bin(ax, bin_centers, mu_c, lo_c, hi_c, [0 0 0]);
end

% brms 95% CI (central only, to keep ribbon legible)
if ~isempty(brms_c_age) && ~isempty(y3c_lo)
    fill(ax, [brms_c_age; flipud(brms_c_age)], [y3c_lo; flipud(y3c_hi)], ...
         color_central, 'EdgeColor','none', 'FaceAlpha', 0.15, 'DisplayName','brms 95% CI (central)');
end

% Trajectories — central solid, frontal dashed for each method
methods = {
    'fitcirc\_lme (vM-GLMM, EM)',           y1c, y1f, [0.10 0.40 0.85];
    'fitlme\_circ (sin/cos)',                y2c, y2f, [0.10 0.65 0.30];
    'brms vM-GLMM (Stan)',                   y3c, y3f, [0.85 0.30 0.55];
    'lme4 sin/cos',                          y4c, y4f, [0.50 0.20 0.85];
    'bpnreg projected-normal (Stan)',        y5c, y5f, [0.85 0.55 0.10] };

plot_one(ax, methods{1,1}, y1c, y1f, methods{1,4}, x_eval,     x_eval,     has_elec);
plot_one(ax, methods{2,1}, y2c, y2f, methods{2,4}, x_eval,     x_eval,     has_elec);
if ~isempty(y3c), plot_one(ax, methods{3,1}, y3c, y3f, methods{3,4}, brms_c_age, brms_f_age, has_elec); end
if ~isempty(y4c), plot_one(ax, methods{4,1}, y4c, y4f, methods{4,4}, x_eval,     x_eval,     has_elec); end
if ~isempty(y5c), plot_one(ax, methods{5,1}, y5c, y5f, methods{5,4}, x_eval,     x_eval,     has_elec); end

% Pretty-print feature label and pull mode-cluster id from the data so
% the user can see at a glance which mode this dump came from. Title
% renders in LaTeX so the formula and stats line up cleanly.
feat_pretty = pretty_feature_name(feat);
cluster_str = '';
if ismember('mode_cluster', T.Properties.VariableNames)
    uc = unique(T.mode_cluster);
    if numel(uc) == 1
        cluster_str = sprintf(', cluster %d', uc);
    else
        cluster_str = sprintf(', clusters \\{%s\\}', strjoin(string(uc(:)'), ','));
    end
end
% Title gets the human-readable info with LaTeX math; the formula
% string (with its raw '_' and '^') goes into the stats annotation
% below, which is rendered with Interpreter='none' so it shows
% literally without further escaping.
title_line1 = sprintf('\\textbf{%s}%s', feat_pretty, cluster_str);
title_line2 = sprintf('$n = %d$ obs / $%d$ subj, $\\bar{R} = %.2f$, $\\theta_{\\mathrm{shift}} = %+.2f$ rad', ...
    height(T), numel(unique(T.Subj_ID)), R_overall, theta_shift);
xlabel(ax,'Age (years)','Interpreter','latex');
ylabel(ax,sprintf('%s (rad)', feat_pretty),'Interpreter','latex');
title(ax, {title_line1, title_line2}, 'Interpreter','latex','FontSize',13);
xlim(ax,[5 85]); ylim(ax,[-pi pi]);
yticks(ax,[-pi -pi/2 0 pi/2 pi]);
yticklabels(ax,{'-\pi','-\pi/2','0','\pi/2','\pi'});
grid(ax,'on');
legend(ax,'Location','eastoutside','FontSize',9,'NumColumns',1,'Interpreter','tex');

% Stats panel as text annotation, full-width below the axes so it doesn't
% collide with the x-axis labels. Includes the Wilkinson formula at the
% top (rendered with Interpreter='none' so '_' and '^' come through
% literally without LaTeX escaping).
panel_str = sprintf('formula: %s\n%s', formula_full, stats_str);
ann = annotation(fig,'textbox',[0.06 0.02 0.88 0.14], ...
    'String', panel_str, ...
    'EdgeColor',[0.85 0.85 0.85],'BackgroundColor',[0.97 0.97 0.97], ...
    'FontName','Courier New','FontSize',10, ...
    'VerticalAlignment','top','Interpreter','none', ...
    'Margin', 6);

set(fig,'Renderer','painters');
exportgraphics(fig, out_png, 'Resolution', 150);
fprintf('Wrote %s\n', out_png);

% Save stats CSV
writetable(stats, fullfile(results_dir, 'fit_stats.csv'));
end


function plot_one(ax, label, yc, yf, col, age_x_c, age_x_f, has_elec_local)
if has_elec_local
    plot(ax, age_x_c, yc, '-',  'Color', col, 'LineWidth', 2.0, 'DisplayName', [label ' (central)']);
    plot(ax, age_x_f, yf, '--', 'Color', col, 'LineWidth', 2.0, 'DisplayName', [label ' (frontal)']);
else
    plot(ax, age_x_c, yc, '-', 'Color', col, 'LineWidth', 2.0, 'DisplayName', label);
end
end


function nd = build_eval(T, x_eval, electrode_val, has_elec, has_sex)
nd = table();
nd.Age = x_eval;
if has_elec, nd.electrode = electrode_val * ones(numel(x_eval),1); end
if has_sex,  nd.sex       = zeros(numel(x_eval),1); end
nd.Subj_ID = double(repmat(T.Subj_ID(1), numel(x_eval), 1));
end


function [mu_b, lo_b, hi_b] = binned_circ(x, y, bins)
nb = numel(bins) - 1;
mu_b = nan(nb,1); lo_b = nan(nb,1); hi_b = nan(nb,1);
for b = 1:nb
    mask = x >= bins(b) & x < bins(b+1);
    nbi = nnz(mask);
    if nbi < 5, continue; end
    Sb = sum(sin(y(mask))); Cb = sum(cos(y(mask)));
    mu_b(b) = atan2(Sb, Cb);
    R_b = sqrt(Sb^2 + Cb^2)/nbi;
    if R_b > 0
        se = sqrt((1 - R_b)/(nbi*R_b));
        lo_b(b) = mu_b(b) - 1.96*se;
        hi_b(b) = mu_b(b) + 1.96*se;
    end
end
end


function plot_bin(ax, bc, mu, lo, hi, color)
% Draw binned-circular-mean error bars. Skip bars where the CI half-width
% exceeds pi/2 — those bins have too much circular dispersion for a
% Cartesian vertical bar to make sense (it would span most of [-pi,pi]
% and dominate the figure visually). Plot the dot regardless so the bin
% mean is still shown.
for b = 1:numel(bc)
    if ~isnan(lo(b)) && (hi(b) - lo(b)) <= pi
        line(ax, [bc(b) bc(b)], [lo(b) hi(b)], 'Color', color, 'LineWidth', 1.0, 'HandleVisibility','off');
    end
end
plot(ax, bc, mu, 'o', 'MarkerSize', 5, 'MarkerFaceColor', color, 'MarkerEdgeColor', color, 'HandleVisibility','off');
end


function S = goodness_of_fit_table(T, feat, m1, m2, results_dir)
% Compute training-set goodness of fit for each method.
% MATLAB methods compute on the spot; R methods read from *_stats.json.
y = T.(feat);
mu_y = atan2(mean(sin(y)), mean(cos(y)));

% MATLAB fits — predict marginally (random effect = 0) so the
% comparison is apples-to-apples with lme4 / brms / bpnreg, all of which
% use re.form=NA / population-level prediction.
nd_train = T(:, intersect(T.Properties.VariableNames, ...
    {'Age','electrode','sex','Subj_ID'}));

% fitcirc_lme: defaults to Conditional=false already, but pass it
% explicitly so the intent is documented at the call site.
yhat1 = m1.predict(nd_train, 'Conditional', false);
ar1 = wrapToPi(y - yhat1);
LL1 = m1.LogLikelihood;
R2_1 = 1 - sum(1 - cos(ar1)) / max(sum(1 - cos(wrapToPi(y - mu_y))), 1e-12);
mae1 = mean(abs(ar1));

% fitlme_circ: forwards to fitlme.predict, which DEFAULTS to
% Conditional=true (uses random effects). Force marginal prediction so
% R² reflects population-level fit, not subject-specific offsets.
yhat2 = m2.predict(nd_train, 'Conditional', false);
ar2 = wrapToPi(y - yhat2);
% fitlme_circ stores per-component LL; sum them
try
    LL2 = m2.LogLikelihood;
catch
    LL2 = NaN;
end
R2_2 = 1 - sum(1 - cos(ar2)) / max(sum(1 - cos(wrapToPi(y - mu_y))), 1e-12);
mae2 = mean(abs(ar2));

% Read R stats jsons
brms_s   = read_or_nan(fullfile(results_dir,'brms_stats.json'));
lme4_s   = read_or_nan(fullfile(results_dir,'lme4_stats.json'));
bpnreg_s = read_or_nan(fullfile(results_dir,'bpnreg_stats.json'));

LL_brms   = field_or_nan(brms_s, 'LL');
LL_lme4   = field_or_nan(lme4_s, 'LL_sin_plus_cos');
LL_bpn    = field_or_nan(bpnreg_s, 'LL');
if isnan(LL_bpn), LL_bpn = field_or_nan(bpnreg_s, 'lppd'); end

S = table();
S.method      = {'MATLAB fitcirc_lme'; 'MATLAB fitlme_circ'; 'R brms vM-GLMM'; 'R lme4 sin/cos'; 'R bpnreg projected-normal'};
S.LL          = [LL1; LL2; LL_brms; LL_lme4; LL_bpn];
S.R2_circ     = [R2_1; R2_2; field_or_nan(brms_s,'R2_circ'); field_or_nan(lme4_s,'R2_circ'); field_or_nan(bpnreg_s,'R2_circ')];
S.mae_angular = [mae1; mae2; field_or_nan(brms_s,'mae_angular'); field_or_nan(lme4_s,'mae_angular'); field_or_nan(bpnreg_s,'mae_angular')];
end


function v = field_or_nan(s, name)
if isstruct(s) && isfield(s, name) && ~isempty(s.(name)) && isnumeric(s.(name))
    v = double(s.(name));
else
    v = NaN;
end
end


function s = read_or_nan(path)
if exist(path,'file')
    s = jsondecode(fileread(path));
else
    s = struct('LL',NaN,'R2_circ',NaN,'mae_angular',NaN);
end
end


function tbl = read_pred(path)
% Return a prediction table or [] if the file doesn't exist or is empty.
if exist(path,'file')
    tbl = readtable(path);
    if height(tbl) == 0, tbl = []; end
else
    tbl = [];
end
end


function varargout = pick_pred(tbl, elec, value_col)
% Return [yhat, age, lo, hi] for the requested electrode level,
% sorted by Age, with empty arrays if tbl is empty.
varargout = {[], [], [], []};
if isempty(tbl), return; end
if ismember('electrode', tbl.Properties.VariableNames)
    mask = tbl.electrode == elec;
else
    mask = true(height(tbl),1);
end
if ~any(mask), return; end
sub = tbl(mask, :);
[age, ix] = sort(sub.Age);
y = sub.(value_col)(ix);
varargout{1} = y;
varargout{2} = age;
if ismember('lo', sub.Properties.VariableNames)
    varargout{3} = sub.lo(ix);
end
if ismember('hi', sub.Properties.VariableNames)
    varargout{4} = sub.hi(ix);
end
end


function w = wrap_pi(x)
% Wrap to (-pi, pi].
w = ((x + pi) - 2*pi*floor((x + pi) / (2*pi))) - pi;
end


function s = latex_escape(s)
% Make a Wilkinson formula string safe for MATLAB's LaTeX interpreter
% as plain text mixed with math segments. Outside math mode underscores
% must be escaped, and the formula's '~' and 'X^k' need to be in math.
s = regexprep(s, '([A-Za-z]+)\^(\d+)', '$1$$^{$2}$$'); % Age^2 -> Age$^{2}$
s = strrep(s, '~', '$\sim$');
s = strrep(s, '_', '\_');
end


function s = pretty_feature_name(feat)
% Map the snake_case dump feature names to human-readable labels for
% titles and y-axis. MATLAB's tex interpreter treats '_' as a subscript
% command, so for tex-rendered titles we prefer these clean strings.
switch lower(feat)
    case 'pref_phase'
        s = 'Phase Preference';
    case 'phase'
        s = 'SO Phase';
    case 'theta'
        s = 'Theta phase';
    case 'stdfreq'
        s = 'STD frequency';
    case 'stdphase'
        s = 'STD phase';
    case 'stdpower'
        s = 'STD power';
    case 'sopower'
        s = 'SO power';
    case 'frequency'
        s = 'Frequency';
    case 'amplitude'
        s = 'Amplitude';
    otherwise
        % Generic fallback: replace underscores with spaces
        s = strrep(feat, '_', ' ');
end
end
