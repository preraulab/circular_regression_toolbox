function meta = write_circ_contract(T, feature, order, results_dir, opts)
%WRITE_CIRC_CONTRACT  Write the data.csv / eval_grid.csv / meta.json contract
% that the R circular-regression worker (circ_fit.R; legacy
% compare_*.R) read. Takes a plain table + feature + order, so it can be
% driven directly from a fit call (not just from a saved dump).
%
%   meta = write_circ_contract(T, feature, order, results_dir, opts)
%
% INPUTS
%   T            table with columns Age (x_col), <feature>, Subj_ID, and
%                optionally electrode, sex.
%   feature      response variable name (char)
%   order        max (Select=true) or single polynomial order
%   results_dir  output directory (created if missing)
%   opts         struct, all fields optional:
%     .Shift        'circmean' (default; subtract circular mean, circ_center)
%                   | 'minvar'  (circ_shift_min_var; legacy harness)
%     .x_col        predictor name (default 'Age')
%     .eval_ages    prediction grid ages (default 7:80)
%     .backend      'brms'|'lme4'|'bpnreg'|'' (written to meta)
%     .select       logical (internal order selection); default true
%     .max_order    written to meta as max_order (default = order)
%     .chains/.iter/.warmup/.seed/.adapt_delta/.band   sampler opts -> meta
%     .dump         provenance path (default '')
%
% OUTPUT
%   meta  struct of everything written to meta.json.
%
% The response is renamed to 'y' and pre-shifted by theta_shift before
% writing; predictions are unshifted (y + theta_shift) downstream.

if nargin < 5, opts = struct(); end
if ~exist(results_dir, 'dir'), mkdir(results_dir); end

x_col     = getopt(opts, 'x_col', 'Age');
shift_kind= getopt(opts, 'Shift', 'circmean');
eval_ages = getopt(opts, 'eval_ages', (7:80)');
eval_ages = eval_ages(:);

% --- Required columns ---
need = {x_col, feature, 'Subj_ID'};
for cc = need
    assert(ismember(cc{1}, T.Properties.VariableNames), ...
        'write_circ_contract:missingColumn', '%s missing from table', cc{1});
end
has_elec = ismember('electrode', T.Properties.VariableNames);
has_sex  = ismember('sex',       T.Properties.VariableNames);

% --- Drop NaN rows in any used numeric column ---
keep = ~isnan(T.(x_col)) & ~isnan(T.(feature));
if has_elec, keep = keep & ~isnan(T.electrode); end
if has_sex,  keep = keep & ~isnan(T.sex); end
T = T(keep, :);

% --- Drop single-level factors ---
if has_elec && numel(unique(T.electrode)) < 2
    fprintf('write_circ_contract: only one electrode level — dropping electrode\n');
    has_elec = false;
end
if has_sex && numel(unique(T.sex)) < 2
    fprintf('write_circ_contract: only one sex level — dropping sex\n');
    has_sex = false;
end

% --- Canonical shift + response rename ---
switch lower(shift_kind)
    case 'minvar'
        [theta_shift, y_shifted] = circ_shift_min_var(T.(feature));
    otherwise   % 'circmean'
        [theta_shift, y_shifted] = circ_center(T.(feature));
end
out = table();
out.y   = y_shifted;
out.(x_col) = T.(x_col);
if has_elec, out.electrode = T.electrode; end
if has_sex,  out.sex       = T.sex; end
out.Subj_ID = T.Subj_ID;
writetable(out, fullfile(results_dir, 'data.csv'));

% --- Eval grid: ages x electrode levels x sex 0 ---
grid = table();
elec_levels = 0; if has_elec, elec_levels = [0 1]; end
for elec = elec_levels
    G = table();
    G.(x_col) = eval_ages;
    if has_elec, G.electrode = elec * ones(numel(eval_ages),1); end
    if has_sex,  G.sex       = zeros(numel(eval_ages),1); end
    G.Subj_ID = repmat(T.Subj_ID(1), numel(eval_ages), 1);
    grid = [grid; G]; %#ok<AGROW>
end
writetable(grid, fullfile(results_dir, 'eval_grid.csv'));

% --- Full formula: y ~ x^order [* electrode] [+ sex] + (1|Subj_ID) ---
if order == 0, rhs = '1'; else, rhs = sprintf('%s^%d', x_col, order); end
if has_elec, rhs = [rhs ' * electrode']; end
if has_sex,  rhs = [rhs ' + sex']; end
formula_full = sprintf('y ~ %s + (1|Subj_ID)', rhs);

% --- meta.json (jsonencode; extra keys are ignored by the R readers) ---
meta = struct();
meta.formula       = formula_full;
meta.x_col         = x_col;
meta.feature       = feature;
meta.order         = order;
meta.has_electrode = double(has_elec);
meta.has_sex       = double(has_sex);
meta.theta_shift   = theta_shift;
meta.shift_kind    = lower(shift_kind);
meta.backend       = getopt(opts, 'backend', '');
meta.select        = double(getopt(opts, 'select', true));
meta.max_order     = getopt(opts, 'max_order', order);
meta.chains        = getopt(opts, 'chains', 4);
meta.iter          = getopt(opts, 'iter', 2000);
meta.warmup        = getopt(opts, 'warmup', 1000);
meta.seed          = getopt(opts, 'seed', 1);
meta.adapt_delta   = getopt(opts, 'adapt_delta', 0.95);
meta.band          = double(getopt(opts, 'band', true));
meta.dump          = getopt(opts, 'dump', '');

fid = fopen(fullfile(results_dir, 'meta.json'), 'w');
fprintf(fid, '%s\n', jsonencode(meta));
fclose(fid);

fprintf('write_circ_contract: %s order<=%d (n=%d, has_elec=%d, shift=%s %+.3f) -> %s\n', ...
    feature, order, height(out), has_elec, lower(shift_kind), theta_shift, results_dir);
end


function v = getopt(opts, name, default)
if isfield(opts, name) && ~isempty(opts.(name))
    v = opts.(name);
else
    v = default;
end
end
