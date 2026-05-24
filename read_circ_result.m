function result = read_circ_result(work_dir, backend, meta)
%READ_CIRC_RESULT  Assemble the uniform circ_fit result struct from the
% unified R-worker outputs in work_dir.
%
%   result = read_circ_result(work_dir, backend, meta)
%
% Reads:
%   <backend>_predictions.csv  Age, electrode, sex, mean, lo, hi (unwrapped)
%   <backend>_stats.json       LL, R2_circ, mae_angular, AIC, BIC, n_obs,
%                              n_subj, chosen_order, select_criterion,
%                              age_effect{pValue,stat,df,method}, diagnostics
%   <backend>_order_table.csv  order, n_par, LogLikelihood, R2_circ,
%                              criterion_value, selected
%   <backend>_coefs.csv  + <backend>_cov_b.csv   (brms only)
%
% `meta` is the struct returned by write_circ_contract (formula, feature,
% theta_shift, x_col, ...).

P = @(f) fullfile(work_dir, f);
backend = lower(char(backend));

% --- stats.json ---
st = jsondecode(fileread(P([backend '_stats.json'])));
GOF = struct( ...
    'R2_circ',       fieldor(st, 'R2_circ', NaN), ...
    'MAE_angular',   fieldor(st, 'mae_angular', NaN), ...
    'LogLikelihood', fieldor(st, 'LL', NaN), ...
    'AIC',           fieldor(st, 'AIC', NaN), ...
    'BIC',           fieldor(st, 'BIC', NaN));

ae = fieldor(st, 'age_effect', struct());
AgeEffect = struct( ...
    'pValue', fieldor(ae, 'pValue', NaN), ...
    'stat',   fieldor(ae, 'stat',   NaN), ...
    'df',     fieldor(ae, 'df',     NaN), ...
    'Method', char(fieldor(ae, 'method', backend)));

chosen_order = fieldor(st, 'chosen_order', meta.order);
sel_crit     = char(fieldor(st, 'select_criterion', 'none'));

% --- predictions -> Trajectory ---
Traj = readtable(P([backend '_predictions.csv']));
if ~ismember('electrode', Traj.Properties.VariableNames)
    Traj.electrode = zeros(height(Traj),1);
end
if ~ismember('sex', Traj.Properties.VariableNames)
    Traj.sex = zeros(height(Traj),1);
end
Traj = Traj(:, {'Age','electrode','sex','mean','lo','hi'});

% --- order table ---
OrderTable = readtable(P([backend '_order_table.csv']));

% --- diagnostics ---
Diagnostics = fieldor(st, 'diagnostics', struct());
converged   = true;
if isfield(Diagnostics, 'rhat_max'),  converged = converged && Diagnostics.rhat_max  < 1.05; end
if isfield(Diagnostics, 'divergent'), converged = converged && Diagnostics.divergent == 0;    end
if isfield(Diagnostics, 'converged'), converged = converged && logical(Diagnostics.converged); end

% --- assemble required tier ---
s = struct();
s.Backend         = backend;
s.Formula         = char(fieldor(meta, 'formula', ''));
s.ResponseName    = char(fieldor(meta, 'feature', ''));
s.Order           = chosen_order;
s.ThetaShift      = fieldor(meta, 'theta_shift', 0);
s.Trajectory      = Traj;
s.GOF             = GOF;
s.AgeEffect       = AgeEffect;
s.OrderTable      = OrderTable;
s.SelectedOrder   = chosen_order;
s.SelectCriterion = sel_crit;
s.Diagnostics     = Diagnostics;
s.Converged       = converged;
s.WorkDir         = work_dir;
s.NumObservations = fieldor(st, 'n_obs', NaN);
s.NumSubjects     = fieldor(st, 'n_subj', NaN);

% --- optional per-coefficient tier (brms only) ---
if exist(P([backend '_coefs.csv']), 'file') && exist(P([backend '_cov_b.csv']), 'file')
    C = readtable(P([backend '_coefs.csv']));
    names = cellstr(string(C.name));
    coef = table(names, C.estimate, C.se, C.pvalue, ...
        'VariableNames', {'Name','Estimate','SE','pValue'});
    s.Coefficients     = coef;
    s.CoefficientNames = names;
    s.Beta             = C.estimate;
    s.NumCoefficients  = height(C);
    cov_b = readmatrix(P([backend '_cov_b.csv']));
    s.cov_b            = cov_b;
    s.ContrastIndex    = contrast_index_from_names(names, char(fieldor(meta,'x_col','Age')));
    s.DFE              = max(fieldor(st,'n_subj',NaN) - 1, 1);
end

result = make_circ_result(s);
end


% ===================== local helpers =====================

function v = fieldor(s, f, default)
if isstruct(s) && isfield(s, f) && ~isempty(s.(f))
    v = s.(f);
else
    v = default;
end
end


function ci = contrast_index_from_names(names, x_col)
% Build ContrastIndex.x_main (+ interaction blocks) from Wilkinson coef
% names. Mirrors fitcirc_lme's name-only logic.
ci = struct();
is_poly = @(nm) strcmp(nm, x_col) || ...
    ~isempty(regexp(nm, ['^' regexptranslate('escape', x_col) '\^?\d+$'], 'once'));
main_idx = [];
for k = 1:numel(names)
    nm = strtrim(names{k});
    if strcmp(nm, '(Intercept)'), continue; end
    if is_poly(nm), main_idx(end+1,1) = k; end %#ok<AGROW>
end
ci.x_main = main_idx;
end
