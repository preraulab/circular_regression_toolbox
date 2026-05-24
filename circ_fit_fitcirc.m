function result = circ_fit_fitcirc(tbl, opts)
%CIRC_FIT_FITCIRC  Native EM von-Mises GLMM backend, in the uniform circ_fit schema.
%
%   result = circ_fit_fitcirc(tbl, opts)
%
% Wraps fitcirc_lme into the common result struct (see make_circ_result).
% Centers the response by its circular mean (circ_center), optionally
% selects polynomial order by LRT, optionally bags over subject resamples,
% builds an unwrapped per-electrode trajectory + CI, and computes the
% cross-backend GOF (population/marginal R2_circ + MAE via circ_gof).
%
% opts fields (see circ_fit for the full list):
%   .x_col (default 'Age'), .feature, .categorical_varnames,
%   .xcol_categorical_interactions, .Select, .MaxOrder, .Order,
%   .Resample ('none'|'cboot'|'sub80'; 'legacy' aliases to 'none'),
%   .B, .KeepFrac, .eval_ages (default 7:80)

x_col   = getopt(opts, 'x_col', 'Age');
feature = getopt(opts, 'feature', '');
cats    = getopt(opts, 'categorical_varnames', {});
intx    = getopt(opts, 'xcol_categorical_interactions', []);
resamp  = normalize_resample(getopt(opts, 'Resample', 'none'));
B       = getopt(opts, 'B', 60);
keep_fr = getopt(opts, 'KeepFrac', 0.8);
do_sel  = getopt(opts, 'Select', false);
max_ord = getopt(opts, 'MaxOrder', 2);
fix_ord = getopt(opts, 'Order', max_ord);
eval_ages = getopt(opts, 'eval_ages', (7:80)'); eval_ages = eval_ages(:);

% --- Restrict to used columns and drop NaN rows ---
keep_cols = unique([{x_col, feature, 'Subj_ID'}, cats(:)'], 'stable');
keep_cols = keep_cols(ismember(keep_cols, tbl.Properties.VariableNames));
sub = tbl(:, keep_cols);
good = true(height(sub),1);
for ii = 1:numel(keep_cols)
    col = sub.(keep_cols{ii});
    if isnumeric(col), good = good & ~isnan(col); end
end
sub = sub(good, :);
n   = height(sub);

has_elec = ismember('electrode', sub.Properties.VariableNames) && numel(unique(sub.electrode)) >= 2;
has_sex  = ismember('sex',       sub.Properties.VariableNames) && numel(unique(sub.sex))       >= 2;

% --- Canonical preprocessing: subtract circular mean ---
theta_shift = circ_center(sub.(feature));

% --- Orders to consider ---
if do_sel, orders = 0:max_ord; else, orders = fix_ord; end
n_ord = numel(orders);

fits  = cell(n_ord,1);
LLs   = nan(n_ord,1);
npars = nan(n_ord,1);
R2s   = nan(n_ord,1);
for i = 1:n_ord
    fml = [build_model_formula(orders(i), x_col, feature, cats, intx) ' + (1|Subj_ID)'];
    m   = fit_one_order(sub, fml, theta_shift, resamp, B, keep_fr);
    fits{i}  = m;
    LLs(i)   = m.LogLikelihood;
    npars(i) = m.NumCoefficients;
    yhat_pop = wrap_pi(m.X_design * m.Beta);
    R2s(i)   = circ_gof_R2(sub.(feature), yhat_pop);
end

% --- LRT order selection (reuse get_LLR with residual-df-style df) ---
crit = nan(n_ord,1);
sel  = false(n_ord,1);
chosen_i = 1;
for i = 2:n_ord
    p = get_LLR(LLs(i-1), LLs(i), n - npars(i-1), n - npars(i));  % p for order_i vs order_{i-1}
    crit(i) = p;
    if p < 0.05 && ~isnan(p)
        chosen_i = i;       % accept higher order, keep going
    else
        break;              % first non-significant step stops the climb
    end
end
sel(chosen_i) = true;
chosen       = fits{chosen_i};
chosen_order = orders(chosen_i);

OrderTable = table(orders(:), npars(:), LLs(:), R2s(:), crit(:), sel(:), ...
    'VariableNames', {'order','n_par','LogLikelihood','R2_circ','criterion_value','selected'});

% --- Trajectory (population, unwrapped per electrode) + CI ---
names = chosen.Coefficients.Name;
Traj  = build_trajectory(names, chosen.Beta, chosen.cov_b, x_col, eval_ages, has_elec, has_sex);

% --- GOF (population/marginal, comparable to the R backends) ---
yhat_pop = wrap_pi(chosen.X_design * chosen.Beta);
g        = circ_gof(sub.(feature), yhat_pop, chosen.NumCoefficients);
GOF      = struct('R2_circ', g.R2_circ, 'R2_adj', g.R2_adj, 'MAE_angular', g.MAE_angular, ...
                  'LogLikelihood', chosen.LogLikelihood, 'AIC', chosen.AIC, 'BIC', chosen.BIC);

% --- Uniform age-effect test: joint Wald on the x_col block ---
AgeEffect = struct('pValue', NaN, 'stat', NaN, 'df', NaN, 'Method', 'Wald');
if chosen_order >= 1 && isfield(chosen.ContrastIndex, 'x_main') && ~isempty(chosen.ContrastIndex.x_main)
    try
        jt = chosen.coefTest('x_main');
        AgeEffect = struct('pValue', jt.pValue, 'stat', jt.Fstat, 'df', jt.df1, 'Method', 'Wald');
    catch ME
        warning('circ_fit_fitcirc:AgeEffect', 'Wald age-effect test failed: %s', ME.message);
    end
end

% --- Assemble uniform result ---
s = struct();
s.Backend         = 'fitcirc_lme';
s.Formula         = chosen.Formula;
s.ResponseName    = feature;
s.Order           = chosen_order;
s.ThetaShift      = theta_shift;
s.Trajectory      = Traj;
s.GOF             = GOF;
s.AgeEffect       = AgeEffect;
s.OrderTable      = OrderTable;
s.SelectedOrder   = chosen_order;
s.SelectCriterion = 'LRT';
s.Diagnostics     = struct('ConvergedIn', chosen.ConvergedIn, 'Resample', resamp, ...
                           'Kappa', chosen.Kappa, 'KappaPhi', chosen.KappaPhi);
s.Converged       = chosen.ConvergedIn < 100;     % MaxIter default in fitcirc_lme
% optional per-coefficient tier
s.Coefficients     = chosen.Coefficients(:, {'Name','Estimate','SE','pValue'});
s.CoefficientNames = chosen.Coefficients.Name;
s.Beta             = chosen.Beta;
s.cov_b            = chosen.cov_b;
s.ContrastIndex    = chosen.ContrastIndex;
s.NumCoefficients  = chosen.NumCoefficients;
s.NumObservations  = chosen.NumObservations;
s.NumSubjects      = chosen.NumSubjects;
s.DFE              = chosen.DFE;
s.Raw              = chosen;

result = make_circ_result(s);
end


% ===================== local helpers =====================

function m = fit_one_order(sub, fml, theta_shift, resamp, B, keep_fr)
% Fit one order with the given circular-mean shift; bag over subject
% resamples when resamp is 'cboot' / 'sub80'.
base = fitcirc_lme(sub, fml, 'ThetaShift', theta_shift);
if strcmp(resamp, 'none')
    m = base;
    return;
end
P     = numel(base.Beta);
subj  = unique(sub.Subj_ID);
nS    = numel(subj);
Bm    = zeros(P, B);
ok    = false(B,1);
for b = 1:B
    if strcmp(resamp, 'cboot')
        pick = subj(randi(nS, nS, 1));
        Tb   = resample_subjects(sub, pick);
    else  % sub80
        nk   = max(2, round(keep_fr * nS));
        pick = subj(randperm(nS, nk));
        Tb   = sub(ismember(sub.Subj_ID, pick), :);
    end
    try
        mb = fitcirc_lme(Tb, fml, 'ThetaShift', theta_shift);
        if numel(mb.Beta) == P, Bm(:,b) = mb.Beta; ok(b) = true; end
    catch
    end
end
m = base;
if any(ok)
    Bm = Bm(:, ok);
    m.Beta  = median(Bm, 2);
    m.cov_b = cov(Bm', 1);
    se = sqrt(diag(m.cov_b));
    t  = m.Beta ./ se;
    df = max(m.NumSubjects - 1, 1);
    m.Coefficients.Estimate = m.Beta;
    m.Coefficients.SE       = se;
    m.Coefficients.tStat    = t;
    m.Coefficients.pValue   = 2 * (1 - tcdf(abs(t), df));
end
end


function Tout = resample_subjects(T, pick)
% With-replacement bootstrap: each pick gets a fresh unique Subj_ID.
parts = cell(numel(pick),1);
for k = 1:numel(pick)
    rows = T(T.Subj_ID == pick(k), :);
    rows.Subj_ID = repmat(k, height(rows), 1);
    parts{k} = rows;
end
Tout = vertcat(parts{:});
end


function Traj = build_trajectory(names, beta, cov_b, x_col, ages, has_elec, has_sex)
% Population trajectory + 95% CI on the eval grid, per electrode level.
elec_levels = 0; if has_elec, elec_levels = [0 1]; end
rows = cell(numel(elec_levels),1);
for e = 1:numel(elec_levels)
    cv = struct();
    if has_elec, cv.electrode = elec_levels(e); end
    if has_sex,  cv.sex = 0; end
    X  = design_from_names(names, x_col, ages, cv);
    eta = X * beta;                                   % continuous linear predictor
    se  = sqrt(max(diag(X * cov_b * X'), 0));
    eta = unwrap(eta);                                % no-op if already continuous
    lo  = eta - 1.96*se;
    hi  = eta + 1.96*se;
    elc = elec_levels(e) * ones(numel(ages),1);
    rows{e} = table(ages, elc, zeros(numel(ages),1), eta, lo, hi, ...
        'VariableNames', {'Age','electrode','sex','mean','lo','hi'});
end
Traj = vertcat(rows{:});
end


function X = design_from_names(names, x_col, ages, catvals)
% Build a design matrix from Wilkinson coefficient names, evaluating the
% x_col polynomial at `ages` and categoricals at the scalars in catvals.
ages = ages(:);
n = numel(ages);
p = numel(names);
X = zeros(n, p);
for j = 1:p
    nm = strtrim(names{j});
    if strcmp(nm, '(Intercept)')
        X(:,j) = 1;
        continue;
    end
    col = ones(n,1);
    for f = strsplit(nm, ':')
        ff = strtrim(f{1});
        if startsWith(ff, x_col)
            tok = regexp(ff, ['^' x_col '\^(\d+)$'], 'tokens');
            if isempty(tok), pw = 1; else, pw = str2double(tok{1}{1}); end
            col = col .* (ages.^pw);
        elseif isfield(catvals, ff)
            col = col .* (catvals.(ff) * ones(n,1));
        else
            error('circ_fit_fitcirc:UnknownFactor', 'Factor "%s" not in catvals.', ff);
        end
    end
    X(:,j) = col;
end
end


function R2 = circ_gof_R2(y, yhat)
g = circ_gof(y, yhat);
R2 = g.R2_circ;
end


function r = normalize_resample(s)
s = lower(char(s));
if strcmp(s, 'legacy'), s = 'none'; end
if ~ismember(s, {'none','cboot','sub80'})
    error('circ_fit_fitcirc:BadResample', 'Resample must be none|cboot|sub80 (got %s).', s);
end
r = s;
end


function v = getopt(opts, name, default)
if isfield(opts, name) && ~isempty(opts.(name))
    v = opts.(name);
else
    v = default;
end
end


function w = wrap_pi(x)
w = ((x + pi) - 2*pi*floor((x + pi) / (2*pi))) - pi;
end
