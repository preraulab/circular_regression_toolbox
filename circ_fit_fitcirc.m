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
%   .continuous_covariates (cellstr, default {}) extra continuous
%       predictors entered as main effects, e.g. {'bmi','ahi'}. Rows with
%       a missing value in any of them are dropped; they are held at their
%       cohort means when the trajectory is drawn. Categorical covariates
%       (e.g. race) go in .categorical_varnames instead.
%   .KappaPhiPrior (default 'halfcauchy'), .KappaPhiPriorScale (default 8)
%       passed straight to fitcirc_lme; these set the weakly-informative
%       prior on the subject-spread concentration kappa_phi. The same
%       prior is used at every polynomial order, so the order-selection
%       likelihood-ratio test compares like with like.

x_col   = getopt(opts, 'x_col', 'Age');
feature = getopt(opts, 'feature', '');
cats    = getopt(opts, 'categorical_varnames', {});
intx    = getopt(opts, 'xcol_categorical_interactions', []);
% Continuous adjustment covariates (e.g. {'bmi','ahi'}): entered as
% main-effect terms alongside the x_col polynomial. Held at their cohort
% means when drawing the trajectory; the omnibus age test and GOF use the
% real per-row values.
covs    = getopt(opts, 'continuous_covariates', {});
if ischar(covs) || isstring(covs), covs = cellstr(covs); end
resamp  = normalize_resample(getopt(opts, 'Resample', 'none'));
B       = getopt(opts, 'B', 60);
keep_fr = getopt(opts, 'KeepFrac', 0.8);
do_sel  = getopt(opts, 'Select', false);
max_ord = getopt(opts, 'MaxOrder', 2);
fix_ord = getopt(opts, 'Order', max_ord);
eval_ages = getopt(opts, 'eval_ages', (7:80)'); eval_ages = eval_ages(:);
% Prior on the subject-spread concentration kappa_phi, passed through to
% fitcirc_lme. Defaults match fitcirc_lme's own defaults.
kphi_prior = getopt(opts, 'KappaPhiPrior', 'halfcauchy');
kphi_scale = getopt(opts, 'KappaPhiPriorScale', 8);

% --- Restrict to used columns and drop NaN rows ---
% covs are included here so they survive the NaN-drop and reach the
% design; rows missing any predictor (e.g. the few NaN AHI values) are
% dropped so the design and response stay row-aligned.
keep_cols = unique([{x_col, feature, 'Subj_ID'}, cats(:)', covs(:)'], 'stable');
keep_cols = keep_cols(ismember(keep_cols, tbl.Properties.VariableNames));
sub = tbl(:, keep_cols);
good = true(height(sub),1);
for ii = 1:numel(keep_cols)
    col = sub.(keep_cols{ii});
    if isnumeric(col), good = good & ~isnan(col); end
end
sub = sub(good, :);
n   = height(sub);

% Reference value for each continuous covariate: its cohort mean. The
% trajectory is drawn holding the covariates here, so the curve reads as
% "x_col effect at the mean BMI/AHI/..." rather than at a covariate of 0.
cov_means = struct();
for ii = 1:numel(covs)
    if ismember(covs{ii}, sub.Properties.VariableNames)
        cov_means.(covs{ii}) = mean(sub.(covs{ii}), 'omitnan');
    end
end

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
warm = struct('Beta', [], 'Names', {{}}, 'Kappa', [], 'KappaPhi', []);
for i = 1:n_ord
    fml = [build_ortho_formula(orders(i), x_col, feature, cats, intx, covs) ' + (1|Subj_ID)'];
    m_cold = fit_one_order(sub, fml, theta_shift, resamp, B, keep_fr, ...
                           struct('Beta',[],'Names',{{}}), kphi_prior, kphi_scale);
    if i == 1 || isempty(warm.Beta)
        m = m_cold;
    else
        % Warm-start order i from order i-1's full converged state: Beta in
        % the expanded basis (the new higher-order columns start at 0), plus
        % Kappa and KappaPhi. The warm fit therefore begins essentially at
        % the lower order's solution, and because the EM is monotone
        % (fitcirc_lme's ascent guard) it can only climb from there -- so the
        % higher order's log-likelihood will not fall below the lower
        % order's. Keep whichever of {cold, warm} reaches the higher LL.
        m_warm = fit_one_order(sub, fml, theta_shift, resamp, B, keep_fr, warm, ...
                               kphi_prior, kphi_scale);
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
    % Defensive guard: with a monotone EM warm-started from the previous
    % order, LLs(i) >= LLs(i-1) should hold. If it is ever violated the
    % higher-order fit failed to converge; surface it loudly instead of
    % letting the LRT below read the drop as "this order adds nothing"
    % (which would silently drop a real effect).
    if i > 1 && LLs(i) < LLs(i-1) - 1e-6
        warning('circ_fit_fitcirc:nonMonotoneLL', ...
            ['Order %d log-likelihood (%.4f) fell below order %d (%.4f); ' ...
             'the EM did not converge. Consider a larger MaxIter.'], ...
            orders(i), LLs(i), orders(i-1), LLs(i-1));
    end
    % Seed the next order from this order's full converged state (Beta,
    % Kappa, KappaPhi), so the next warm fit starts at this solution.
    warm.Beta     = m.Beta;
    warm.Names    = m.Coefficients.Name;
    warm.Kappa    = m.Kappa;
    warm.KappaPhi = m.KappaPhi;
end

% --- LRT order selection (reuse get_LLR with residual-df-style df) ---
% Step up the polynomial order while each nested likelihood-ratio test
% comparing (order_i, order_{i-1}) is significant at p < 0.05; stop at
% the first non-significant step. This is the methodologically standard
% selector (matched to the lme/blme path in get_best_iterative_order.m)
% and what the paper's Methods will report.
%
% This relies on the per-order marginal log-likelihoods being finite and
% on a common footing. The subject-spread concentration kappa_phi can sit
% at a boundary (kappa_phi -> infinity when the fixed effects already
% capture the trend), where an unregularized estimate would send the
% log-likelihood to NaN and corrupt the comparison -- the classic
% variance-component-at-a-boundary problem (Stram & Lee 1994). fitcirc_lme
% applies the same weakly-informative kappa_phi prior at every order (see
% KappaPhiPrior / KappaPhiPriorScale above), which keeps each
% log-likelihood finite and smooth so the LRT compares like with like.
% The OrderTable still records R2_circ alongside LL as a cross-check.
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
Traj  = build_trajectory(names, chosen.Beta, chosen.cov_b, x_col, eval_ages, has_elec, has_sex, ortho_info, cov_means);

% --- GOF (marginal + conditional R2_circ, comparable to R backends) ---
% R2_circ_marginal is the fixed-effect-only variance fraction on the
% angle scale, computed by circ_gof from the population predictions
% yhat_pop = X*Beta. Comparable across backends.
%
% R2_circ_conditional lifts R2_circ_marginal by the per-subject random
% phase intercept's contribution using the Nakagawa-Schielzeth (2013)
% closed-form ICC adjustment, adapted to a von Mises GLMM. Each
% variance component is the circular variance V = 1 - I_1(kappa)/I_0(kappa)
% of the corresponding von Mises concentration parameter:
%     V_alpha = 1 - I_1(KappaPhi) / I_0(KappaPhi)   (subject phase RE)
%     V_eps   = 1 - I_1(Kappa)    / I_0(Kappa)      (residual)
%     ICC     = V_alpha / (V_alpha + V_eps)
%     R2_c    = R2_m + ICC * (1 - R2_m)
% KappaPhi -> Inf is the EM's signal that the data does not support a
% per-subject random intercept (the angular feature carries no
% persistent personal baseline -- e.g. mode tilt in some clusters).
% In that limit V_alpha = 0, ICC = 0, R2_c = R2_m, which is the
% correct conservative reporting.
yhat_pop = wrap_pi(chosen.X_design * chosen.Beta);
g        = circ_gof(sub.(feature), yhat_pop, chosen.NumCoefficients);
R2_marg  = g.R2_circ;
R2_cond  = R2_marg;
if isfinite(R2_marg) && isfinite(chosen.Kappa) && chosen.Kappa > 0
    if isfinite(chosen.KappaPhi) && chosen.KappaPhi > 0
        V_alpha = 1 - besseli(1, chosen.KappaPhi) / besseli(0, chosen.KappaPhi);
    else
        V_alpha = 0;
    end
    V_eps = 1 - besseli(1, chosen.Kappa) / besseli(0, chosen.Kappa);
    if (V_alpha + V_eps) > 0
        icc     = V_alpha / (V_alpha + V_eps);
        R2_cond = R2_marg + icc * (1 - R2_marg);
    end
end
GOF = struct( ...
    'R2_circ',             R2_marg, ...      % alias of R2_circ_marginal (back-compat)
    'R2_circ_marginal',    R2_marg, ...
    'R2_circ_conditional', R2_cond, ...
    'R2_adj',              g.R2_adj, ...
    'MAE_angular',         g.MAE_angular, ...
    'LogLikelihood',       chosen.LogLikelihood, ...
    'AIC',                 chosen.AIC, ...
    'BIC',                 chosen.BIC);

% --- Uniform age-effect test: joint Wald on the x_col block ---
% Omnibus age effect: joint Wald that ALL age terms (polynomial main
% effect + every age-interaction) are zero -> one "any age effect" p.
AgeEffect = struct('pValue', NaN, 'stat', NaN, 'df', NaN, 'Method', 'Wald-F-circ');
if chosen_order >= 1 && isfield(chosen.ContrastIndex, 'x_age') && ~isempty(chosen.ContrastIndex.x_age)
    try
        jt = chosen.coefTest('x_age');
        AgeEffect = struct('pValue', jt.pValue, 'stat', jt.Fstat, 'df', jt.df1, 'Method', 'Wald-F-circ');
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

function m = fit_one_order(sub, fml, theta_shift, resamp, B, keep_fr, warm, kphi_prior, kphi_scale)
% Fit one order with the given circular-mean shift; bag over subject
% resamples when resamp is 'cboot' / 'sub80'. `warm` (optional) carries
% the previous order's converged Beta/Names/Kappa/KappaPhi for warm-
% starting the EM (see the order loop in the caller). kphi_prior /
% kphi_scale set the kappa_phi prior and are forwarded to fitcirc_lme.
if nargin < 7 || isempty(warm), warm = struct('Beta',[],'Names',{{}},'Kappa',[],'KappaPhi',[]); end
if nargin < 8 || isempty(kphi_prior), kphi_prior = 'halfcauchy'; end
if nargin < 9 || isempty(kphi_scale), kphi_scale = 8; end
kphi_nv = {'KappaPhiPrior', kphi_prior, 'KappaPhiPriorScale', kphi_scale};
nv = [{'ThetaShift', theta_shift}, kphi_nv];
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
        mb = fitcirc_lme(Tb, fml, 'ThetaShift', theta_shift, kphi_nv{:});
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


function Traj = build_trajectory(names, beta, cov_b, x_col, ages, has_elec, has_sex, ortho_info, cov_means)
% Population trajectory + 95% CI on the eval grid, per electrode level.
% ortho_info carries the orthogonal polynomial basis transformation used
% during fitting; the design at eval_ages is built in the SAME basis so
% predictions are valid. Traj.Age stays raw (degrees-of-time axis).
% cov_means holds each continuous covariate at its cohort mean so the
% drawn curve is the x_col effect at average covariate values.
if nargin < 8, ortho_info = []; end
if nargin < 9, cov_means = struct(); end
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
sex_levels  = 0; if has_sex,  sex_levels  = [0 1]; end
rows = cell(numel(elec_levels) * numel(sex_levels), 1);
r = 0;
for e = 1:numel(elec_levels)
    for s = 1:numel(sex_levels)
        r = r + 1;
        cv = struct();
        if has_elec, cv.electrode = elec_levels(e); end
        if has_sex,  cv.sex       = sex_levels(s);  end
        % Hold each continuous covariate at its cohort mean.
        fn = fieldnames(cov_means);
        for c = 1:numel(fn), cv.(fn{c}) = cov_means.(fn{c}); end
        X  = design_from_names(names, x_col, P_eval, cv);
        eta = X * beta;                                   % continuous linear predictor
        se  = sqrt(max(diag(X * cov_b * X'), 0));
        eta = unwrap(eta);                                % no-op if already continuous
        lo  = eta - 1.96*se;
        hi  = eta + 1.96*se;
        elc = elec_levels(e) * ones(numel(ages),1);
        sxc = sex_levels(s)  * ones(numel(ages),1);
        rows{r} = table(ages, elc, sxc, eta, lo, hi, ...
            'VariableNames', {'Age','electrode','sex','mean','lo','hi'});
    end
end
Traj = vertcat(rows{:});
end


function fml = build_ortho_formula(order, x_col, feature, cats, intx, covs)
% Wilkinson formula in the orthogonal-polynomial basis. Uses explicit
% column names `<x_col>_op1`, `<x_col>_op2`, ..., `<x_col>_opK` (added to
% the table by circ_fit_fitcirc before this call) instead of the
% polynomial syntax `Age^k` which fitlme would expand to a correlated
% raw-power basis. cats are categorical main effects; covs are continuous
% adjustment covariates; both enter as main effects (covs are not
% interacted with x_col).
if nargin < 6, covs = {}; end
parts = {'1'};
for j = 1:order
    parts{end+1} = sprintf('%s_op%d', x_col, j); %#ok<AGROW>
end
for k = 1:numel(cats)
    parts{end+1} = cats{k}; %#ok<AGROW>
end
for k = 1:numel(covs)
    parts{end+1} = covs{k}; %#ok<AGROW>
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
            % Factor not supplied in catvals: default to 0. This is the
            % path for categorical level dummies (e.g. race_1), so the
            % trajectory is drawn at the reference level. Continuous
            % covariates are supplied through catvals (held at their
            % cohort means), so they do not reach this branch. Either way
            % this affects only the drawn curve, not the age-block omnibus
            % statistic, which is computed from beta and cov_b directly.
            col = col .* 0;
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
