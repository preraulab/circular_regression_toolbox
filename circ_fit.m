function result = circ_fit(tbl, formula, backend, varargin)
%CIRC_FIT  Fit a circular regression and return a uniform result struct.
%
%   result = circ_fit(tbl, formula)
%   result = circ_fit(tbl, formula, backend)
%   result = circ_fit(tbl, formula, backend, Name, Value, ...)
%   result = circ_fit(tbl, formula, backend, optsStruct)
%
% This is the entry point of the toolbox. It takes a tidy data table, a
% formula in the same Wilkinson grammar `fitlme` uses, and the name of
% one of four estimator backends, and it hands back a single struct
% with the same field names regardless of which backend did the work.
% That uniform output lets you swap estimators (sensitivity check,
% backend comparison, etc.) without changing any downstream code.
%
% Options can be supplied as Name-Value pairs (the MATLAB-idiomatic
% form, matching fitlme / fitglm / fitnlm / etc.) OR as a single
% struct (the legacy form, still accepted for backward compatibility).
% The two are equivalent: `'Select', true, 'MaxOrder', 2` works
% identically to `struct('Select', true, 'MaxOrder', 2)`.
%
% INPUTS
%   tbl      a MATLAB table. Required columns: the response (an angle in
%            radians; values can sit on any 2*pi interval -- the toolbox
%            re-centers them internally), the predictor named on the
%            right side of the formula, and a `Subj_ID` column when the
%            formula carries a random intercept `(1|Subj_ID)`. Optional
%            columns can be added freely (e.g. electrode, sex, race);
%            anything mentioned on the formula will be picked up.
%   formula  Wilkinson string, same grammar fitlme uses. Example:
%              'Phase ~ 1 + Age^2 + electrode + sex + (1|Subj_ID)'
%            Notes:
%              * `^k` denotes a polynomial in that predictor up to
%                degree k. With `opts.Select = true`, the toolbox treats
%                this as the MAX order to consider and chooses the best
%                k by step-up likelihood-ratio test; otherwise it fits
%                exactly the order written.
%              * The random term must be `(1|<column>)` with a single
%                grouping variable; nested or crossed random effects are
%                not supported.
%   backend  one of:
%              'fitcirc_lme'  native EM von-Mises GLMM (MATLAB only;
%                             default; recommended for most use)
%              'brms'         Stan vM-GLMM (LOO order selection)
%              'bpnreg'       Bayesian projected-normal mixed model
%            The R-backed backends require R + the named package on the
%            machine; see the toolbox README's "Dependencies" section.
%   Name-Value pairs (or fields of optsStruct; any omitted take the
%            defaults shown in parentheses). The most useful options:
%              .Select      (false) true -> step-up LRT order selection
%                                    up to MaxOrder
%              .MaxOrder    (parsed from formula) cap for order selection
%              .Order       (parsed from formula) fixed order when
%                                    Select = false
%              .feature     (parsed from formula) response column name
%              .x_col       ('Age') base predictor name; the polynomial
%                                    is built on this column
%              .categorical_varnames  cellstr of factor columns
%              .xcol_categorical_interactions  1xC logical, which of
%                                    those factors interact with x_col
%              .Resample    ('none' | 'cboot' | 'sub80') resampling
%                                    bagging mode for fitcirc_lme
%              .B           bag size for resampling
%              .eval_ages   x-axis grid for the returned trajectory
%              .Chains, .Iter, .Warmup, .Seed, .AdaptDelta  Stan
%                                    sampler options (brms only)
%              .WorkDir     scratch dir for the R worker contract; if
%                                    empty, a stable per-slice cache
%                                    path is computed
%              .RscriptPath ('/usr/local/bin/Rscript') override if your
%                                    R is elsewhere
%              .BrmsFallback (true) if the R backend errors out, warn
%                                    and silently fall back to
%                                    fitcirc_lme
%
% OUTPUT
%   result   the uniform circ_result struct, validated by
%            make_circ_result. Highlights (full spec in
%            docs/result_schema.md):
%              .Backend, .Formula, .ResponseName, .SelectedOrder
%              .Coefficients          table {Name, Estimate, SE, pValue}
%              .AgeEffect.pValue      omnibus joint Wald that ALL
%                                    age-involving coefficients are zero
%              .GOF.R2_circ_marginal  fixed-effects-only fit quality
%              .GOF.R2_circ_conditional fixed + subject random intercept
%              .Trajectory            evaluation-grid table for plotting
%
% USAGE TIPS
%   * For a hands-on walkthrough with a synthetic dataset where every
%     true parameter is known, run tutorial.m (`run('tutorial.m')`).
%   * To overlay multiple backends on the same data, fit each
%     separately and pass them as a cell array to plot_circ_fit:
%       r1 = circ_fit(tbl, fml, 'fitcirc_lme', 'Select', true, 'MaxOrder', 2);
%       r2 = circ_fit(tbl, fml, 'brms',        'Select', true, 'MaxOrder', 2);
%       plot_circ_fit({r1, r2}, tbl);
%
% On R-backend failure (missing toolchain, nonzero exit, missing
% outputs) the function warns and falls back to fitcirc_lme provided
% opts.BrmsFallback is true (the default). Set it to false to surface
% the underlying R error instead.
%
% SEE ALSO  fitcirc_lme, circular_regression, make_circ_result,
%           plot_circ_fit, circ_vmrnd, tutorial.

if nargin < 3 || isempty(backend), backend = 'fitcirc_lme'; end
backend = lower(char(backend));

% Parse remaining args into an opts struct. Two accepted forms:
%   circ_fit(..., backend)                          -- no options
%   circ_fit(..., backend, optsStruct)              -- legacy single struct
%   circ_fit(..., backend, 'Name', value, ...)      -- name-value pairs
% Name-value pairs and struct fields are equivalent; mixing them in one
% call is not supported (a single struct is the only thing accepted as
% the fourth positional argument).
opts = parse_options(varargin);

% --- Seed feature/order/x_col/categorical_varnames from the formula if
% not supplied. Anything on the RHS that is not the polynomial-in-x_col
% block and not the random-intercept term is treated as a categorical
% main-effect covariate; callers can override by passing an explicit
% categorical_varnames or by passing continuous_covariates for terms
% that should be treated as continuous.
if ~isfield(opts,'x_col') || isempty(opts.x_col), opts.x_col = 'Age'; end
[feat0, ord0, cats0] = parse_formula(formula, opts.x_col);
if ~isfield(opts,'feature') || isempty(opts.feature),  opts.feature  = feat0; end
if ~isfield(opts,'Order')   || isempty(opts.Order),    opts.Order    = ord0;  end
if ~isfield(opts,'MaxOrder')|| isempty(opts.MaxOrder), opts.MaxOrder = ord0;  end
if ~isfield(opts,'Select'),  opts.Select = false; end
if ~isfield(opts,'categorical_varnames') || isempty(opts.categorical_varnames)
    cont = {}; if isfield(opts,'continuous_covariates') && ~isempty(opts.continuous_covariates), cont = cellstr(opts.continuous_covariates); end
    opts.categorical_varnames = setdiff(cats0, cont, 'stable');
end

if strcmp(backend, 'fitcirc_lme')
    result = circular_regression(tbl, opts);
    return;
end
if ~ismember(backend, {'brms','bpnreg'})
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
        result = circular_regression(tbl, opts);
    else
        rethrow(ME);
    end
end
end


% ===================== local helpers =====================

function result = run_r_backend(tbl, backend, opts)
this_dir = fileparts(mfilename('fullpath'));
rscript  = getopt(opts, 'RscriptPath', '/usr/local/bin/Rscript');
worker   = fullfile(this_dir, 'R', 'circ_fit.R');

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
    'adapt_delta',getopt(opts, 'AdaptDelta', 0.95));
meta = write_circ_contract(tbl, feature, order, work, contract_opts);

cmd = sprintf('%s "%s" "%s" %s', rscript, worker, work, backend);
[status, out] = system(cmd);
if status ~= 0
    error('circ_fit:RscriptFailed', 'Rscript exit %d:\n%s', status, tail_lines(out, 25));
end

result = read_circ_result(work, backend, meta);
end


function [feature, order, cats] = parse_formula(formula, x_col)
% Pull from the Wilkinson formula the response name, the polynomial
% order on the base predictor x_col, and the list of remaining RHS
% terms (covariates / categoricals). The random-intercept block, the
% literal `1`, polynomial-in-x_col terms, and any interaction-with-
% x_col terms are removed; what's left is returned in `cats` so the
% caller can forward them as categorical_varnames.
if nargin < 2 || isempty(x_col), x_col = 'Age'; end
formula = char(formula);
ti      = strfind(formula, '~');
feature = strtrim(formula(1:ti(1)-1));
rhs     = formula(ti(1)+1:end);

% Strip the random-intercept block(s) before tokenizing.
rhs_noran = regexprep(rhs, '\([^)]*\|[^)]*\)', '');

% Split on `+` and trim each piece.
parts = strsplit(rhs_noran, '+');
parts = cellfun(@strtrim, parts, 'UniformOutput', false);

% Polynomial order: largest k where `x_col^k` appears; otherwise 1 if a
% bare x_col token is present; otherwise 0.
order = 0;
poly_pat = ['^' regexptranslate('escape', x_col) '\^(\d+)$'];
has_bare = false;
for k = 1:numel(parts)
    p = parts{k};
    if isempty(p), continue; end
    if strcmp(p, x_col), has_bare = true; continue; end
    tok = regexp(p, poly_pat, 'tokens', 'once');
    if ~isempty(tok)
        order = max(order, str2double(tok{1}));
    end
end
if order == 0 && has_bare, order = 1; end

% Keep tokens that are valid variable names AND are not the intercept,
% not the x_col main effect, not a polynomial in x_col, and not an
% interaction. Anything left is a categorical / covariate main effect
% the caller may forward as categorical_varnames.
cats = {};
for k = 1:numel(parts)
    p = parts{k};
    if isempty(p) || strcmp(p, '1') || strcmp(p, '0')
        continue;
    end
    if strcmp(p, x_col) || ~isempty(regexp(p, poly_pat, 'once'))
        continue;
    end
    % Drop products / interactions: x:y, x*y. Keep main effects only.
    if any(ismember(p, ':*'))
        continue;
    end
    if isvarname(p) && ~ismember(p, cats)
        cats{end+1} = p; %#ok<AGROW>
    end
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


function opts = parse_options(args)
% Accept either a single struct or a flat list of Name-Value pairs.
% Returns a struct with one field per supplied option.
if isempty(args)
    opts = struct();
    return;
end
if numel(args) == 1 && isstruct(args{1})
    opts = args{1};
    return;
end
if mod(numel(args), 2) ~= 0
    error('circ_fit:OddNVargs', ...
          'Name-Value arguments must come in pairs (got %d).', numel(args));
end
opts = struct();
for k = 1:2:numel(args)
    name = args{k};
    if ~(ischar(name) || isstring(name)) || ~isvarname(char(name))
        error('circ_fit:BadOptName', ...
              'Name-Value option name at position %d is not a valid identifier.', k);
    end
    opts.(char(name)) = args{k+1};
end
end
