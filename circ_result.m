classdef circ_result < matlab.mixin.CustomDisplay
%CIRC_RESULT  Uniform result object returned by circ_fit.
%
% A circ_result holds the output of any backend in the same schema. Field
% access uses ordinary property syntax (`r.GOF.R2_circ_marginal`), so
% existing code that read the previous struct schema continues to work.
% Typing the object at the command prompt prints a fitlme-style model
% summary (formula, sample size, fixed-effect coefficient table, omnibus
% age test, R^2_circ marginal and conditional).
%
% The schema is the one validated by make_circ_result. Required fields
% are always populated; optional fields default to empty when the
% backend does not expose them (e.g. bpnreg does not carry a
% single coefficient vector, so .Coefficients is empty for it).
%
% USAGE
%   r = circ_fit(tbl, 'Phase ~ Age^2 + (1|Subj_ID)', 'fitcirc_lme', ...
%                'Select', true, 'MaxOrder', 2);
%   r                       % prints model summary
%   r.GOF.R2_circ_marginal  % ordinary property access
%   r.AgeEffect.pValue
%   fields = fieldnames(r); % list every populated field name
%   s = struct(r);          % convert to a plain struct
%
% SEE ALSO  make_circ_result, circ_fit, circular_regression, plot_circ_fit.

properties
    % --- required schema fields ---
    Backend          = ''
    Formula          = ''
    ResponseName     = ''
    Order            = NaN
    ThetaShift       = 0
    Trajectory       = table()
    GOF              = struct()
    AgeEffect        = struct('pValue', NaN, 'stat', NaN, 'df', NaN, 'Method', '')
    OrderTable       = table()
    SelectedOrder    = NaN
    SelectCriterion  = ''
    Diagnostics      = struct()
    Converged        = false

    % --- optional schema fields (populated by backends that expose them) ---
    Coefficients     = []   % table or []
    CoefficientNames = {}   % cellstr / string array
    Beta             = []   % vector or []
    cov_b            = []   % matrix or []
    ContrastIndex    = struct()
    NumCoefficients  = NaN
    NumObservations  = NaN
    NumSubjects      = NaN
    DFE              = NaN
    WorkDir          = ''
    Raw              = []   % native backend handle/object or []
end

methods
    function obj = circ_result(s)
    %CIRC_RESULT  Construct a circ_result from a struct of schema fields.
    %
    %   r = circ_result(s)   copies field-by-field; absent fields take
    %                        the class defaults declared above.
    if nargin == 0, return; end
    if ~isstruct(s)
        error('circ_result:BadInput', ...
              'circ_result constructor expects a struct (got %s).', class(s));
    end
    props = properties(obj);
    fns   = fieldnames(s);
    for k = 1:numel(fns)
        f = fns{k};
        if any(strcmp(f, props))
            try
                obj.(f) = s.(f);
            catch err
                warning('circ_result:AssignFailed', ...
                        'Could not assign field "%s" (%s).', f, err.message);
            end
        end
    end
    end

    % Struct-compatibility helpers -----------------------------------
    function out = struct(obj)
    %STRUCT  Convert to a plain struct for legacy callers.
    out = struct();
    props = properties(obj);
    for k = 1:numel(props)
        out.(props{k}) = obj.(props{k});
    end
    end

    function tf = isfield(obj, name)
    %ISFIELD  True when `name` is a public property of the object.
    if iscell(name)
        tf = false(size(name));
        for k = 1:numel(name), tf(k) = isfield(obj, name{k}); end
    else
        tf = any(strcmp(char(name), properties(obj)));
    end
    end

    function out = fieldnames(obj)
    %FIELDNAMES  List the schema fields (same as properties).
    out = properties(obj);
    end

    function ax = plot(obj, tbl, opts)
    %PLOT  Plot the fitted trajectory and CI band over the raw data.
    %
    %   plot(r)              draw the trajectory on a fresh figure with
    %                        no scatter overlay
    %   plot(r, tbl)         overlay the table's raw x/y scatter
    %                        behind the trajectory
    %   plot(r, tbl, opts)   pass-through options for plot_circ_fit
    %                        (e.g. opts.ax, opts.plot_CI, opts.colors)
    %
    % Dispatches to plot_circ_fit (the toolbox plotter) with one
    % result. For overlaying multiple backends, call the standalone
    % function: plot_circ_fit({r1, r2, ...}, tbl).
    if nargin < 2, tbl = table(); end
    if nargin < 3, opts = struct(); end
    % Reuse the current figure / axes if the caller already has one
    % open (so `figure(); plot(r, T)` lands in that figure rather
    % than spawning a second blank one); otherwise gca opens a fresh
    % figure as needed.
    if ~isfield(opts, 'ax') || isempty(opts.ax)
        opts.ax = gca;
    end
    if nargout > 0
        ax = plot_circ_fit(obj, tbl, opts);
    else
        plot_circ_fit(obj, tbl, opts);
    end
    end
end

methods (Access = protected)
    function displayScalarObject(obj)
    % Pretty-print the model when the object is typed at the prompt
    % or passed to disp. Output mirrors the layout MATLAB's
    % LinearMixedModel.disp uses: header, model formula, sample size,
    % polynomial-order line, fixed-effect coefficient table, omnibus
    % age test, R^2_circ marginal and conditional.

    header = sprintf('  Circular mixed-effects regression (backend: %s)', obj.Backend);
    disp(' ');
    disp(header);
    disp(repmat('-', 1, length(header)));
    disp(' ');

    % --- model summary block ---
    fprintf('  Formula            %s\n', obj.Formula);
    n_obs  = local_nan(obj.NumObservations);
    n_subj = local_nan(obj.NumSubjects);
    if isfinite(n_obs) && isfinite(n_subj)
        fprintf('  N obs / groups     %d / %d\n', n_obs, n_subj);
    elseif isfinite(n_obs)
        fprintf('  N obs              %d\n', n_obs);
    end
    if ~isnan(obj.SelectedOrder)
        sel_str = sprintf('  Polynomial order   %d', obj.SelectedOrder);
        if ~isempty(obj.SelectCriterion)
            sel_str = sprintf('%s   (selected by %s)', sel_str, obj.SelectCriterion);
        end
        disp(sel_str);
    end

    % --- coefficient table ---
    if istable(obj.Coefficients) && ~isempty(obj.Coefficients)
        disp(' ');
        disp('  Fixed-effect coefficients:');
        print_coef_table(obj.Coefficients);
    end

    % --- omnibus age effect ---
    if isstruct(obj.AgeEffect) && isfield(obj.AgeEffect, 'pValue') && ...
            isfinite(obj.AgeEffect.pValue)
        disp(' ');
        F   = field_or(obj.AgeEffect, 'stat',   NaN);
        df  = field_or(obj.AgeEffect, 'df',     NaN);
        p   = obj.AgeEffect.pValue;
        meth = field_or(obj.AgeEffect, 'Method', '');
        if isfinite(F) && isfinite(df)
            fprintf('  Omnibus age effect F(%g) = %.3f,  p = %s   [%s]\n', ...
                    df, F, fmt_pvalue(p), meth);
        else
            fprintf('  Omnibus age effect p = %s   [%s]\n', fmt_pvalue(p), meth);
        end
    end

    % --- GOF block ---
    if isstruct(obj.GOF) && ~isempty(fieldnames(obj.GOF))
        disp(' ');
        r2m = field_or(obj.GOF, 'R2_circ_marginal', ...
                       field_or(obj.GOF, 'R2_circ', NaN));
        r2c = field_or(obj.GOF, 'R2_circ_conditional', NaN);
        mae = field_or(obj.GOF, 'MAE_angular', NaN);
        if isfinite(r2m)
            fprintf('  R^2_circ marginal  %.3f\n', r2m);
        end
        if isfinite(r2c)
            fprintf('  R^2_circ condit. %.3f\n', r2c);
        end
        if isfinite(mae)
            fprintf('  MAE_angular        %.3f rad   (= %.1f deg)\n', mae, rad2deg(mae));
        end
    end
    disp(' ');
    end
end
end


% ===================== local helpers =====================

function v = local_nan(x)
if isempty(x) || (isnumeric(x) && all(isnan(x))), v = NaN; else, v = x; end
end


function v = field_or(s, name, default)
v = default;
if isstruct(s) && isfield(s, name) && ~isempty(s.(name))
    v = s.(name);
end
end


function s = fmt_pvalue(p)
if isnan(p)
    s = 'NaN';
elseif p < eps
    % Exact zero on the float scale -- report as the smallest p-value
    % MATLAB doubles can represent rather than as the misleading "0".
    s = '< 1e-308';
elseif p < 1e-16
    s = sprintf('< %.0e', 1e-16);
elseif p < 1e-4
    s = sprintf('%.1e', p);
elseif p < 1e-3
    s = sprintf('%.4f', p);
else
    s = sprintf('%.3f', p);
end
end


function print_coef_table(T)
% Compact fitlme-style coefficient block. Reads Estimate, SE,
% tStat (or NaN), pValue from the input table.
names = T.Properties.VariableNames;
get   = @(n, d) get_or(T, n, d);
N     = height(T);

est   = get('Estimate', nan(N,1));
se    = get('SE',       nan(N,1));
tval  = get('tStat',    nan(N,1));
pval  = get('pValue',   nan(N,1));

% circular_regression's result keeps {Name, Estimate, SE, pValue} and
% drops tStat to match the uniform cross-backend schema. Compute
% tStat as Estimate / SE wherever the table did not supply it so the
% display block reports a t-like statistic for each coefficient.
missing_t = isnan(tval) & isfinite(se) & se > 0;
tval(missing_t) = est(missing_t) ./ se(missing_t);
if ismember('Name', names)
    nm = string(T.Name);
elseif ismember('Row', names)
    nm = string(T.Row);
else
    nm = string(T.Properties.RowNames);
end

% Compute name column width.
namew = max([10, max(strlength(nm)) + 2]);
fmt_h = sprintf('   %%-%ds %%10s %%10s %%10s %%10s\n', namew);
fmt_r = sprintf('   %%-%ds %%10.4g %%10.4g %%10.4g %%10s\n', namew);
fprintf(fmt_h, 'Name', 'Estimate', 'SE', 'tStat', 'pValue');
for k = 1:N
    fprintf(fmt_r, char(nm(k)), est(k), se(k), tval(k), fmt_pvalue(pval(k)));
end
end


function out = get_or(T, name, default)
if ismember(name, T.Properties.VariableNames) && isnumeric(T.(name))
    out = T.(name);
else
    out = default;
end
end
