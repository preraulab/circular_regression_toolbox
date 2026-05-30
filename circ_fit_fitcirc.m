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

% --- Build orthogonal polynomial basis for x_col at the maximum order
% being considered, and add columns x_col_op1..x_col_opK to the
% subtable. Orthogonal polynomials replace the raw power basis (Age,
% Age^2, Age^3) because:
%   (a) Numerical conditioning: raw columns are O(1)..O(1e5) and
%       cross-correlated > 0.99 over [7, 80], which makes IRLS inside
%       the von-Mises EM jitter into local optima far from the MLE.
%   (b) Per-coefficient interpretability: with raw collinear columns,
%       beta_Age's Wald p-value tests "linear after quadratic absorbed
%       what it could", not "linear effect"; orthogonal columns give
%       each beta a standalone marginal meaning in the supplemental
%       table.
%   (c) Nested-order consistency: QR is greedy, so cols 1..j of the
%       order-K basis are identical to the order-j basis. Warm-start
%       by name therefore transfers real structure between orders
%       instead of seeding zeros into a reparameterized space.
% The fitted curve and the joint Wald test are unchanged by this
% reparameterization. Concretely: if T is the nonsingular linear map
% from the orthogonal basis to the raw power basis (so beta_raw =
% T*beta_op), and the omnibus null is R*beta_op = 0 in the orthogonal
% basis, then it is equivalently (R*inv(T))*beta_raw = 0 in the raw
% basis. The Wald quadratic form beta'*R'*(R*cov_b*R')^{-1}*R*beta is
% invariant under this nonsingular linear reparameterization (standard
% quadratic-form algebra; see e.g. the discussion of Wald-statistic
% invariance under linear vs nonlinear reparameterization in the
% econometrics literature -- Lafontaine & White 1986 establish the
% nonlinear case is NOT invariant, which is why we are careful to keep
% the reparameterization here strictly linear). Per-row coefficient
% p's gain a clean standalone interpretation because cross-correlation
% in the design is removed. The ortho_info struct is propagated to
% build_trajectory so predictions at eval_ages use the same basis
% transformation.
x_raw = sub.(x_col);
max_basis = max(1, max_ord);
[P_basis, ortho_info] = ortho_poly_basis(x_raw, max_basis);
for j = 1:max_basis
    sub.(sprintf('%s_op%d', x_col, j)) = P_basis(:, j);
end

% --- Orders to consider ---
if do_sel, orders = 0:max_ord; else, orders = fix_ord; end
n_ord = numel(orders);

fits  = cell(n_ord,1);
LLs   = nan(n_ord,1);
npars = nan(n_ord,1);
R2s   = nan(n_ord,1);
% Robust-init chain: at each order, fit twice -- once cold and once
% warm-started from the previous order's converged Beta (matched by
% coefficient name; new columns start at 0). Keep whichever fit attains
% the higher marginal LL. This handles two failure modes simultaneously:
%   (a) raw-Age design columns are wildly different scales (Age, Age^2,
%       Age^3 = O(1e1), O(1e3), O(1e5)), so cold IRLS often lands in a
%       basin where the new high-order coefficients are stuck near 0;
%   (b) cold IRLS for higher orders sometimes finds a worse local
%       optimum than the intercept-only fit (LL decreases with k),
%       which the LRT then incorrectly rejects -> SelectedOrder=0 and
%       AgeEffect.pValue=NaN.
% The "max-LL of {cold, warm}" rule is what we actually want for the
% LRT to be principled: under the null, both inits converge to the
% same MLE; under the alternative, one of them finds it.
warm = struct('Beta', [], 'Names', {{}});
for i = 1:n_ord
    fml = [build_ortho_formula(orders(i), x_col, feature, cats, intx) ' + (1|Subj_ID)'];
    m_cold = fit_one_order(sub, fml, theta_shift, resamp, B, keep_fr, ...
                           struct('Beta',[],'Names',{{}}));
    if i == 1 || isempty(warm.Beta)
        m = m_cold;
    else
        m_warm = fit_one_order(sub, fml, theta_shift, resamp, B, keep_fr, warm);
        if m_warm.LogLikelihood > m_cold.LogLikelihood
            m = m_warm;
        else
            m = m_cold;
        end
    end
    fits{i}  = m;
    LLs(i)   = m.LogLikelihood;
    npars(i) = m.NumCoefficients;
    yhat_pop = wrap_pi(m.X_design * m.Beta);
    R2s(i)   = circ_gof_R2(sub.(feature), yhat_pop);
    % Seed the next order from this order's Beta (only Beta -- carrying
    % Kappa/KappaPhi forward poisoned the EM in clusters with weak
    % low-order signal).
    warm.Beta  = m.Beta;
    warm.Names = m.Coefficients.Name;
end

% --- LRT order selection (reuse get_LLR with residual-df-style df) ---
% Step up the polynomial order while each nested likelihood-ratio test
% comparing (order_i, order_{i-1}) is significant at p < 0.05; stop at
% the first non-significant step. This is the methodologically standard
% selector (matched to the lme/blme path in get_best_iterative_order.m)
% and what the paper's Methods will report.
%
% NB: when the EM's marginal LL is non-monotone in order (a known
% boundary issue for the random-effects concentration kappa_phi -- see
% Stram & Lee 1994 on testing variance components at the boundary), the
% LRT will conservatively reject the higher order even when the actual
% angular fit improves substantially. Symptom: AgeEffect.pValue = NaN
% for clusters where R2_circ visibly jumps with order but the marginal
% LL drops. The principled fix is to regularize kappa_phi away from the
% upper boundary (cap or half-Cauchy prior on sigma_phi); this is
% recorded as a follow-up and is not done here. The OrderTable records
% R2_circ alongside LL so the diagnostic is visible to downstream
% audits.
crit = nan(n_ord, 1);
sel  = false(n_ord, 1);
chosen_i = 1;
for i = 2:n_ord
    p = get_LLR(LLs(i-1), LLs(i), n - npars(i-1), n - npars(i));
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
% Pass ortho_info so the design at eval_ages is built in the same
% orthogonal polynomial basis used for fitting; the Traj.Age column
% itself stays raw.
names = chosen.Coefficients.Name;
Traj  = build_trajectory(names, chosen.Beta, chosen.cov_b, x_col, eval_ages, has_elec, has_sex, ortho_info);

% --- GOF (population/marginal, comparable to the R backends) ---
yhat_pop = wrap_pi(chosen.X_design * chosen.Beta);
g        = circ_gof(sub.(feature), yhat_pop, chosen.NumCoefficients);
GOF      = struct('R2_circ', g.R2_circ, 'R2_adj', g.R2_adj, 'MAE_angular', g.MAE_angular, ...
                  'LogLikelihood', chosen.LogLikelihood, 'AIC', chosen.AIC, 'BIC', chosen.BIC);

% --- Uniform age-effect test: joint Wald on the x_col block ---
% Omnibus age effect: joint Wald that ALL age terms (polynomial main
% effect + every age-interaction) are zero -> one "any age effect" p.
AgeEffect = struct('pValue', NaN, 'stat', NaN, 'df', NaN, 'Method', 'Wald');
if chosen_order >= 1 && isfield(chosen.ContrastIndex, 'x_age') && ~isempty(chosen.ContrastIndex.x_age)
    try
        jt = chosen.coefTest('x_age');
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
s.OrthoInfo       = ortho_info;   % basis transform for x_col polynomial
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

function m = fit_one_order(sub, fml, theta_shift, resamp, B, keep_fr, warm)
% Fit one order with the given circular-mean shift; bag over subject
% resamples when resamp is 'cboot' / 'sub80'. `warm` (optional) carries
% the previous order's converged Beta/Names/Kappa/KappaPhi for warm-
% starting the EM (see the order loop in the caller).
if nargin < 7 || isempty(warm), warm = struct('Beta',[],'Names',{{}},'Kappa',[],'KappaPhi',[]); end
nv = {'ThetaShift', theta_shift};
if ~isempty(warm.Beta) && ~isempty(warm.Names)
    nv = [nv, {'InitBeta', warm.Beta, 'InitBetaNames', warm.Names}];
end
if isfield(warm,'Kappa')    && ~isempty(warm.Kappa),    nv = [nv, {'InitKappa',    warm.Kappa}];    end
if isfield(warm,'KappaPhi') && ~isempty(warm.KappaPhi), nv = [nv, {'InitKappaPhi', warm.KappaPhi}]; end
base = fitcirc_lme(sub, fml, nv{:});
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


function Traj = build_trajectory(names, beta, cov_b, x_col, ages, has_elec, has_sex, ortho_info)
% Population trajectory + 95% CI on the eval grid, per electrode level.
% ortho_info carries the orthogonal polynomial basis transformation used
% during fitting; the design at eval_ages is built in the SAME basis so
% predictions are valid. Traj.Age stays raw (degrees-of-time axis).
if nargin < 8, ortho_info = []; end
% Evaluate the orthogonal basis at eval_ages once. P_eval(:,j) is the
% j-th orthogonal-polynomial column at each eval age. Empty ortho_info
% means caller did not standardize the design -- shouldn't happen in
% practice but kept as a safety net.
if ~isempty(ortho_info)
    P_eval = ortho_poly_basis(ages(:), ortho_info.k, ortho_info);
else
    P_eval = ages(:) .^ (1:max(1,3));   % fallback: raw power basis
end
elec_levels = 0; if has_elec, elec_levels = [0 1]; end
rows = cell(numel(elec_levels),1);
for e = 1:numel(elec_levels)
    cv = struct();
    if has_elec, cv.electrode = elec_levels(e); end
    if has_sex,  cv.sex = 0; end
    X  = design_from_names(names, x_col, P_eval, cv);
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


function fml = build_ortho_formula(order, x_col, feature, cats, intx)
% Wilkinson formula in the orthogonal-polynomial basis. Uses explicit
% column names `<x_col>_op1`, `<x_col>_op2`, ..., `<x_col>_opK` (added to
% the table by circ_fit_fitcirc before this call) instead of the
% polynomial syntax `Age^k` which fitlme would expand to a correlated
% raw-power basis.
parts = {'1'};
for j = 1:order
    parts{end+1} = sprintf('%s_op%d', x_col, j); %#ok<AGROW>
end
for k = 1:numel(cats)
    parts{end+1} = cats{k}; %#ok<AGROW>
end
if ~isempty(intx)
    if islogical(intx), mask = intx; else, mask = logical(intx); end
    inter_cats = cats(mask);
    for k = 1:numel(inter_cats)
        for j = 1:order
            parts{end+1} = sprintf('%s_op%d:%s', x_col, j, inter_cats{k}); %#ok<AGROW>
        end
    end
end
fml = [feature ' ~ ' strjoin(parts, ' + ')];
end


function X = design_from_names(names, x_col, P_eval, catvals)
% Build a design matrix from coefficient names, using the orthogonal
% polynomial basis evaluated at eval_ages (P_eval, n x K) for the x_col
% main effects and interactions. Coefficient names look like
% '(Intercept)', '<x_col>_opJ', '<x_col>_opJ:<cat>', or '<cat>'.
n = size(P_eval, 1);
p = numel(names);
X = zeros(n, p);
op_pat = ['^' x_col '_op(\d+)$'];
for j = 1:p
    nm = strtrim(names{j});
    if strcmp(nm, '(Intercept)')
        X(:,j) = 1;
        continue;
    end
    col = ones(n,1);
    for f = strsplit(nm, ':')
        ff = strtrim(f{1});
        tok = regexp(ff, op_pat, 'tokens');
        if ~isempty(tok)
            jj = str2double(tok{1}{1});
            if jj < 1 || jj > size(P_eval, 2)
                error('circ_fit_fitcirc:OrthoDegree', ...
                      'Coefficient "%s" requires basis degree %d but only %d available.', ...
                      ff, jj, size(P_eval,2));
            end
            col = col .* P_eval(:, jj);
        elseif isfield(catvals, ff)
            col = col .* (catvals.(ff) * ones(n,1));
        else
            error('circ_fit_fitcirc:UnknownFactor', 'Factor "%s" not in catvals or basis.', ff);
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
