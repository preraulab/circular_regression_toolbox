function result = make_circ_result(s)
%MAKE_CIRC_RESULT  Construct + validate the uniform circular-fit result struct.
%
%   result = make_circ_result(s)
%
% Single source of truth for the schema every circular-regression backend
% (circ_fit_fitcirc / circ_fit via R) must return. Pass a struct `s` with
% the fields below; missing optional fields are filled with defaults, and
% the required tier is validated.
%
% REQUIRED of every backend
%   Backend         char  'fitcirc_lme' | 'brms' | 'lme4' | 'bpnreg'
%   Formula         char  Wilkinson formula actually fit
%   ResponseName    char
%   Order           scalar  selected (or fixed) polynomial order
%   ThetaShift      scalar  circular-mean shift applied before fitting
%   Trajectory      table {Age, electrode, sex, mean, lo, hi}  (mean UNWRAPPED per electrode;
%                         lo=hi=mean when a band is unavailable)
%   GOF             struct {R2_circ, MAE_angular, LogLikelihood, AIC, BIC}
%                         (AIC/BIC may be NaN for Bayesian backends)
%   AgeEffect       struct {pValue, stat, df, Method}  uniform age-effect test
%   OrderTable      table {order, n_par, LogLikelihood, R2_circ, criterion_value, selected}
%   SelectedOrder   scalar
%   SelectCriterion char  'LRT' | 'LRT-sincos' | 'LOO' | 'WAIC' | 'none'
%   Diagnostics     struct  backend-specific
%   Converged       logical
%
% OPTIONAL (per-coefficient inference; populated only for fitcirc_lme + brms,
% [] otherwise)
%   Coefficients    table {Name, Estimate, SE, pValue}
%   CoefficientNames cellstr
%   Beta            vector       cov_b  matrix
%   ContrastIndex   struct (.x_main, ...)
%   NumCoefficients scalar
%   NumObservations scalar       NumSubjects scalar    DFE scalar
%   WorkDir         char         Raw  (native handle/object)

required = { 'Backend','Formula','ResponseName','Order','ThetaShift', ...
             'Trajectory','GOF','AgeEffect','OrderTable','SelectedOrder', ...
             'SelectCriterion','Diagnostics','Converged' };
optional_defaults = struct( ...
    'Coefficients',    [], ...
    'CoefficientNames',{{}}, ...
    'Beta',            [], ...
    'cov_b',           [], ...
    'ContrastIndex',   struct(), ...
    'NumCoefficients', NaN, ...
    'NumObservations', NaN, ...
    'NumSubjects',     NaN, ...
    'DFE',             NaN, ...
    'WorkDir',         '', ...
    'Raw',             []);

% --- Required-tier presence check ---
for k = 1:numel(required)
    f = required{k};
    if ~isfield(s, f) || (isempty(s.(f)) && ~istable(s.(f)) && ~islogical(s.(f)))
        error('make_circ_result:MissingField', ...
              'Required field "%s" is missing or empty.', f);
    end
end

% --- Type checks on the structured required fields ---
assert(istable(s.Trajectory), 'make_circ_result:BadTrajectory', 'Trajectory must be a table.');
need_traj = {'Age','mean','lo','hi'};
assert(all(ismember(need_traj, s.Trajectory.Properties.VariableNames)), ...
    'make_circ_result:BadTrajectory', 'Trajectory needs columns Age, mean, lo, hi.');
assert(istable(s.OrderTable), 'make_circ_result:BadOrderTable', 'OrderTable must be a table.');
for f = {'GOF','AgeEffect','Diagnostics'}
    assert(isstruct(s.(f{1})), 'make_circ_result:BadField', '%s must be a struct.', f{1});
end
for f = {'R2_circ','MAE_angular','LogLikelihood','AIC','BIC'}
    assert(isfield(s.GOF, f{1}), 'make_circ_result:BadGOF', 'GOF.%s missing.', f{1});
end
for f = {'pValue','Method'}
    assert(isfield(s.AgeEffect, f{1}), 'make_circ_result:BadAgeEffect', 'AgeEffect.%s missing.', f{1});
end

% --- Assemble the schema as a plain struct first ---
assembled = struct();
for k = 1:numel(required)
    assembled.(required{k}) = s.(required{k});
end
opt = fieldnames(optional_defaults);
for k = 1:numel(opt)
    f = opt{k};
    if isfield(s, f)
        assembled.(f) = s.(f);
    else
        assembled.(f) = optional_defaults.(f);
    end
end

% --- Wrap as a circ_result object so it prints a fitlme-style summary
% when typed at the prompt; property access is unchanged for callers
% that read fields directly (`result.GOF.R2_circ_marginal` etc.).
result = circ_result(assembled);
end
