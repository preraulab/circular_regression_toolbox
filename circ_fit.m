function result = circ_fit(tbl, formula, backend, opts)
%CIRC_FIT  Unified dispatcher for circular-regression backends.
%
%   result = circ_fit(tbl, formula, backend, opts)
%
% Returns the common result struct (see make_circ_result) for any backend:
%   'fitcirc_lme'  native EM von-Mises GLMM      -> circ_fit_fitcirc (MATLAB)
%   'brms'         Stan vM-GLMM (LOO selection)  -> R worker circ_fit.R
%   'lme4'         sin/cos projected-Gaussian    -> R worker circ_fit.R
%   'bpnreg'       Bayesian projected-normal     -> R worker circ_fit.R
%
% INPUTS
%   tbl      table with response, Age, Subj_ID, optional electrode/sex
%   formula  Wilkinson string (feature ~ ... + (1|Subj_ID)); the Age^k order
%            in it is the MAX order when opts.Select is true, else the single
%            order to fit. Used to seed feature/order if not in opts.
%   backend  one of the names above (default 'fitcirc_lme')
%   opts     struct; common fields: Select, MaxOrder, Order, x_col, feature,
%            categorical_varnames, xcol_categorical_interactions, Resample, B,
%            KeepFrac, eval_ages, Chains, Iter, Warmup, Seed, AdaptDelta, Band,
%            WorkDir, RscriptPath, BrmsFallback.
%
% On R-backend failure (missing toolchain, nonzero exit, missing outputs) and
% opts.BrmsFallback (default true), warns and falls back to fitcirc_lme.

if nargin < 3 || isempty(backend), backend = 'fitcirc_lme'; end
if nargin < 4, opts = struct(); end
backend = lower(char(backend));

% --- Seed feature/order/x_col from the formula if not supplied ---
[feat0, ord0] = parse_formula(formula);
if ~isfield(opts,'feature') || isempty(opts.feature), opts.feature = feat0; end
if ~isfield(opts,'x_col')   || isempty(opts.x_col),   opts.x_col   = 'Age'; end
if ~isfield(opts,'Order')   || isempty(opts.Order),   opts.Order   = ord0;  end
if ~isfield(opts,'MaxOrder')|| isempty(opts.MaxOrder),opts.MaxOrder= ord0;  end
if ~isfield(opts,'Select'),  opts.Select = false; end

if strcmp(backend, 'fitcirc_lme')
    result = circ_fit_fitcirc(tbl, opts);
    return;
end
if ~ismember(backend, {'brms','lme4','bpnreg'})
    error('circ_fit:UnknownBackend', 'Unknown backend "%s".', backend);
end

% --- R-backed path ---
fallback = getopt(opts, 'BrmsFallback', true);
try
    result = run_r_backend(tbl, backend, opts);
catch ME
    if fallback
        warning('circ_fit:RFallback', ...
            '%s backend failed (%s); falling back to fitcirc_lme.', backend, ME.message);
        result = circ_fit_fitcirc(tbl, opts);
    else
        rethrow(ME);
    end
end
end


% ===================== local helpers =====================

function result = run_r_backend(tbl, backend, opts)
this_dir = fileparts(mfilename('fullpath'));
rscript  = getopt(opts, 'RscriptPath', '/usr/local/bin/Rscript');
worker   = fullfile(this_dir, 'circ_fit.R');

feature  = opts.feature;
order    = opts.MaxOrder;     % R worker sweeps 0..order when select=true
select   = getopt(opts, 'Select', false);

% Stable per-(feature, slice) cache dir so brms .rds caches persist.
work = getopt(opts, 'WorkDir', '');
if isempty(work)
    tag  = slice_tag(tbl, feature, order);
    work = fullfile(this_dir, 'results', 'circ_cache', tag);
end

contract_opts = struct( ...
    'Shift',      'circmean', ...
    'x_col',      opts.x_col, ...
    'eval_ages',  getopt(opts, 'eval_ages', (7:80)'), ...
    'backend',    backend, ...
    'select',     select, ...
    'max_order',  order, ...
    'chains',     getopt(opts, 'Chains', 4), ...
    'iter',       getopt(opts, 'Iter', 2000), ...
    'warmup',     getopt(opts, 'Warmup', 1000), ...
    'seed',       getopt(opts, 'Seed', 1), ...
    'adapt_delta',getopt(opts, 'AdaptDelta', 0.95), ...
    'band',       getopt(opts, 'Band', true));
meta = write_circ_contract(tbl, feature, order, work, contract_opts);

cmd = sprintf('%s "%s" "%s" %s', rscript, worker, work, backend);
[status, out] = system(cmd);
if status ~= 0
    error('circ_fit:RscriptFailed', 'Rscript exit %d:\n%s', status, tail_lines(out, 25));
end

result = read_circ_result(work, backend, meta);
end


function [feature, order] = parse_formula(formula)
formula = char(formula);
ti = strfind(formula, '~');
feature = strtrim(formula(1:ti(1)-1));
pw = regexp(formula, '\^(\d+)', 'tokens');
if ~isempty(pw)
    order = max(cellfun(@(t) str2double(t{1}), pw));
elseif ~isempty(regexp(formula, '\bAge\b', 'once'))
    order = 1;
else
    order = 0;
end
end


function tag = slice_tag(tbl, feature, order)
% Short, data-dependent tag so different cluster slices get separate dirs.
cols = intersect({'Subj_ID','Age', feature}, tbl.Properties.VariableNames, 'stable');
v = [];
for c = cols
    x = tbl.(c{1});
    if isnumeric(x), v = [v; x(~isnan(x))]; end %#ok<AGROW>
end
h = mod(round(sum(double(v)) * 1e3) + numel(v)*7919, 1e8);
tag = sprintf('%s_o%d_n%d_%08d', matlab.lang.makeValidName(feature), order, height(tbl), h);
end


function s = tail_lines(str, k)
lines = strsplit(str, newline);
lines = lines(max(1, end-k+1):end);
s = strjoin(lines, newline);
end


function v = getopt(opts, name, default)
if isfield(opts, name) && ~isempty(opts.(name))
    v = opts.(name);
else
    v = default;
end
end
