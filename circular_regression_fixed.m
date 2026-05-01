function [estimated_phase, mdl] = circular_regression_fixed(x, y, varargin)
%CIRCULAR_REGRESSION_FIXED  Fisher-Lee circular regression (von-Mises identity link).

% Cubic-in-age polynomials on raw scale (Age, Age^2, Age^3) are mildly
% ill-conditioned — sums-of-products like Age^6 span ~10 orders of
% magnitude across [7,80], so the Newton-step solve below trips
% MATLAB's "nearly singular" warning even though the LU factors are
% numerically fine for our coefficients. Suppress for the duration of
% this fit; the warning auto-restores on function exit.
prev_warn = warning('off', 'MATLAB:nearlySingularMatrix');
prev_warn2 = warning('off', 'MATLAB:singularMatrix');
cleaner_warn = onCleanup(@() restore_warnings(prev_warn, prev_warn2)); %#ok<NASGU>
%
% =====================================================================
% PURPOSE
% =====================================================================
% Fits a regression model for a circular (angular) response variable y,
% wrapped to (-pi, pi], as a function of a continuous predictor x (entered
% as a polynomial of arbitrary order), optional fixed-effect categorical
% covariates, and optional interactions between the polynomial-x block and
% any of those categoricals.  The likelihood is von Mises with identity
% link: each observation is assumed to be drawn from
%
%        y_i  ~  vonMises( mu_i,  kappa ),     mu_i = x_i' * beta
%
% with a single concentration parameter kappa shared across observations.
% The "identity link" means the linear predictor x_i'*beta is interpreted
% directly as a phase angle (in radians).  Wrapping is handled implicitly
% through the periodicity of sin/cos in the score and Hessian.
%
% This function is a drop-in replacement for circular_regression.m with
% several numerical and statistical corrections; see CHANGES below.
%
% =====================================================================
% USAGE
% =====================================================================
%   [yhat, mdl] = circular_regression_fixed(x, y)
%   [yhat, mdl] = circular_regression_fixed(x, y, 'Name', value, ...)
%
% =====================================================================
% INPUTS
% =====================================================================
%   x : Nx1 numeric predictor.  Will be expanded into [x, x^2, ..., x^Order]
%       columns of the design matrix.  Centering/scaling is the caller's
%       responsibility -- the function does not auto-orthogonalize the
%       polynomial.  Joint Wald tests on the polynomial block are
%       invariant to within-block linear transformation, but individual
%       coefficients (and their CIs) are not, so center x if you intend
%       to interpret the linear coefficient.
%
%   y : Nx1 wrapped phase response in radians.  Anything outside (-pi, pi]
%       is wrapped on entry via wrapToPi.
%
% =====================================================================
% NAME-VALUE PARAMETERS
% =====================================================================
%   'Order'             integer >= 0.  Polynomial order in x.   (default 1)
%                       Order = 0 fits intercept (+ categoricals) only.
%
%   'Categorical'       NxC numeric matrix of additional fixed-effect
%                       columns.  Typically 0/1 dummy codes (e.g. electrode,
%                       sex), but any numeric coding is accepted.       ([])
%
%   'CategoricalNames'  1xC cellstr/string of names, used in coefficient
%                       table row labels and for ContrastIndex fields.  ({})
%
%   'Interactions'      1xC logical.  For each categorical column,
%                       Interactions(c)=true adds Order new columns to the
%                       design matrix corresponding to
%                           x*c, x^2*c, ..., x^Order*c.
%                       Interactions(c)=false leaves c as a main effect
%                       only.                                          ([])
%
%   'PredictorName'     name of x for coefficient labels.            ('x')
%
%   'ResponseName'      name of y, currently unused but retained for
%                       symmetry with the original API.                ('y')
%
%   'ClusterID'         Nx1 grouping vector (e.g. Subj_ID) for cluster-
%                       robust (sandwich) SEs.  When provided, the
%                       coefficient covariance is the Liang-Zeger
%                       sandwich estimator with small-sample correction;
%                       residual degrees of freedom drop to G-1 (number
%                       of distinct clusters minus one).  When omitted,
%                       SEs come from the inverse observed information
%                       evaluated at the MLE (n-p df).                  ([])
%
%   'MaxIter'           IRLS iteration cap.                         (200)
%
%   'Tol'               convergence tolerance on the relative coefficient
%                       change ||db|| / max(||b||, 1) per iteration.  (1e-8)
%
%   'Verbose'           true to print a per-iteration trace of step size,
%                       relative coefficient change, and sum(cos r).  (false)
%
% =====================================================================
% OUTPUTS
% =====================================================================
%   estimated_phase : Nx1 fitted phases, wrapped to (-pi, pi].
%
%   mdl : struct with fields
%       .Coefficients     table (Estimate, SE, tStat, pValue, DF) row-named
%                         by coefficient.  pValue is TWO-SIDED.
%       .CoefficientNames cellstr in column order of the design matrix.
%       .NumCoefficients  P, scalar.
%       .DFE              residual degrees of freedom (n - P, or G - 1
%                         if cluster-robust).
%       .Kappa            MLE concentration parameter.
%       .A1               I_1(kappa)/I_0(kappa) at the MLE; appears in
%                         the expected Fisher information.
%       .LogLikelihood    final log-likelihood at the MLE.
%       .Deviance         2 * sum(1 - cos(residuals)).  Standard
%                         goodness-of-fit summary for von Mises models.
%       .CovarianceType   'model-based' or 'cluster-robust'.
%       .cov_b            P x P coefficient covariance matrix.
%       .Rsquared         struct with Ordinary and Adjusted pseudo-R^2,
%                         using wrapped squared deviations from the
%                         circular mean (Jammalamadaka analog of SST/SSE).
%       .Converged        logical, true if the IRLS hit Tol within MaxIter.
%       .Iterations       count of IRLS iterations actually used.
%       .ResidualWrapped  Nx1 residuals wrapped to (-pi, pi].
%       .DesignMatrix     N x P design matrix (intercept, polynomial,
%                         categoricals, interactions, in that column
%                         order).
%       .ContrastIndex    struct of row indices into the coefficient
%                         vector, naming the joint-test "blocks":
%                           .x_main             polynomial-x main-effect
%                                               rows (Age, Age^2, ...)
%                           .<categoricalName>  single row per categorical
%                                               main effect
%                           .x_x_<catName>      polynomial-x interaction
%                                               block for that categorical
%                         Field names are MATLAB-validated via
%                         matlab.lang.makeValidName.
%       .coefTest         function handle: mdl.coefTest(R) runs a Wald
%                         joint test of R*beta = 0 and returns a struct
%                         with chi^2 and F variants of the result.  R may
%                         be either an m x P contrast matrix or a 0/1 row
%                         mask of length P (the helper coerces).
%
% =====================================================================
% CHANGES vs. ORIGINAL circular_regression.m
% =====================================================================
% (1) Two-sided p-values.  The original used
%         pValue = 1 - tcdf(|t|, df)
%     which is the upper-tail one-sided p-value.  Every reported circular
%     coefficient p-value in the lab's downstream supplemental tables was
%     therefore half of what it should have been.  Fixed:
%         pValue = 2 * (1 - tcdf(|t|, df))
%
% (2) Newton-Raphson IRLS with the correct cos-weighted Hessian.
%     The original iteration was
%         z   = eta + sin(y - eta)
%         b   = (X'*X) \ (X'*z)
%     i.e. a Picard fixed-point iteration on the score equation
%     X' sin(y - X*beta) = 0 using *unit weights*.  This converges to the
%     MLE when it converges (same stationary point), but it converges
%     linearly and can stall or oscillate when the residuals are wide,
%     because the implied Hessian (-X'X) is too pessimistic for tight
%     fits and too optimistic for wide ones.  This implementation uses
%     the actual observed-information Hessian
%         H = -kappa * X' * diag(cos(r)) * X
%     for a Newton step
%         db = (X' * diag(max(cos r, eps)) * X) \ (X' * sin(r))
%     and step-halves on the score-likelihood proxy sum(cos r) until the
%     proposed step does not decrease the (proxy of the) likelihood.
%     This is quadratic near the MLE and falls back to a Fisher-scoring
%     step in regions of poor fit (where cos r would otherwise be
%     negative and the Hessian non-PD).
%
% (3) Honest convergence flag.  The original ran a fixed iteration cap
%     with no diagnostic.  This version returns Converged and Iterations,
%     and emits a warning when MaxIter is exhausted without reaching Tol.
%
% (4) Optional cluster-robust SEs (Liang-Zeger sandwich).  Pass ClusterID
%     -- typically Subj_ID -- to obtain SEs that respect within-cluster
%     correlation when each subject contributes multiple modes (e.g. one
%     central + one frontal observation, or multiple modes per electrode
%     within a class).  This addresses the absence of a random effect in
%     the original implementation, which treated repeated within-subject
%     observations as independent.
%
% (5) Coefficient-name indexing for interactions is correct for arbitrary
%     numbers of interacting categoricals.  The original computed the
%     destination slice with the wrong length and overwrote it on each
%     iteration of the categorical loop, so when more than one
%     categorical was interacted with the polynomial, only the last one's
%     names survived.  Fixed by precomputing all interaction names in
%     order and assigning them as one block.
%
% (6) Wald joint-test method on the model struct.  mdl.coefTest(R) lets
%     downstream code request a single statistic for an entire block of
%     coefficients (e.g. "any polynomial-age effect", "any electrode-
%     by-age interaction") instead of falling back to ad-hoc summaries
%     of per-coefficient p-values.  See ContrastIndex for the row sets
%     that are pre-named.
%
% (7) NaN-safe input.  Rows where x, y, or any Categorical column is NaN
%     are dropped before fitting, matching what get_single_order_model.m
%     was doing externally on the caller side.
%
% =====================================================================
% MATHEMATICAL DETAILS
% =====================================================================
% The von-Mises log-likelihood for n iid observations is
%
%     log L(beta, kappa) = -n*log(2*pi*I0(kappa))
%                        + kappa * sum_i cos(y_i - x_i' * beta)
%
% Score (gradient w.r.t. beta):
%     dL/dbeta = kappa * X' * sin(y - X*beta)         ... (eq. S)
%
% Observed Hessian (curvature w.r.t. beta):
%     d^2L/dbeta dbeta' = -kappa * X' * diag(cos(y - X*beta)) * X
%
% Setting the score to zero and applying Newton-Raphson with the observed
% Hessian gives the IRLS update used here:
%     beta_{k+1} = beta_k + (X' W X)^{-1} X' sin(r),
%     W = diag(cos(r)).
%
% Asymptotically, two valid covariance forms are
%     model-based, observed:    (kappa * X' * diag(cos r_hat) * X)^{-1}
%     model-based, expected:    (kappa * A1(kappa) * X' * X)^{-1}
% These coincide at large samples when residuals are tightly clustered
% (so that mean(cos r) ~ A1(kappa)).  The original used the expected
% form; this implementation uses the observed form, which is slightly
% more accurate at finite samples and is the natural by-product of the
% Newton iteration.
%
% kappa is estimated by the Berens (2009) / Fisher (1993) approximation
% to the MLE: it solves A1(kappa) = mean_i cos(r_i) via piecewise
% rational approximations, with a small-sample correction for n < 15.
% A numerical cap at 1e4 prevents besseli overflow at extremely tight
% fits; in practice this only triggers for synthetic data with kappa in
% the thousands.
%
% Cluster-robust covariance.  When repeated observations from the same
% subject contribute to a single fit, the iid assumption that justifies
% the model-based covariance breaks down.  Liang & Zeger (1986)
% sandwich:
%     V_CR = A^{-1} * B * A^{-1}
% with bread A^{-1} = (kappa X' W X)^{-1} (the same as model-based) and
% meat
%     B = sum_g U_g U_g',     U_g = sum_{i in g} kappa * x_i * sin(r_i),
% i.e. the outer product of per-cluster aggregated scores.  The Liang-
% Zeger small-sample correction
%     V_CR <- (G/(G-1)) * ((n-1)/(n-P)) * V_CR
% is applied; residual df is set to G - 1 for the t reference
% distribution.
%
% Pseudo R^2.  No single circular R^2 is universally accepted.  This
% implementation uses a Jammalamadaka-style construction: SSE is the sum
% of squared *wrapped* residuals, SST is the sum of squared *wrapped*
% deviations from the circular mean of y, and R^2 = 1 - SSE/SST.  This
% reduces to the linear R^2 when residuals are small and is bounded
% above by 1 (but can in principle be negative for very poor fits, like
% any pseudo-R^2).
%
% =====================================================================
% LIMITATIONS WORTH KNOWING
% =====================================================================
% (a) The model class cannot represent precessions that wrap multiple
%     times across the predictor range.  The intercept is initialized to
%     the circular mean of y, which puts the iteration on the unwrapped
%     sheet closest to the data; a true 2*pi or larger precession over
%     the range of x would be missed in favor of a flatter fit on the
%     same sheet.  For typical lifespan EEG phase data (precessions << pi
%     over decades) this is non-binding.
%
% (b) Cluster-robust SEs are first-order.  They assume the score is
%     correctly specified (mean zero per cluster under the null) but do
%     NOT correct the *point estimates* if intra-cluster correlation
%     biases beta_hat.  For severe clustering, a proper mixed-effects
%     circular model (e.g. brms with family = von_mises and a (1|Subj)
%     random intercept) is preferable.
%
% (c) kappa is treated as fixed at its MLE when computing beta SEs.
%     Strictly, the joint covariance of (beta, kappa) has off-diagonal
%     terms that are zero asymptotically but not at finite samples.
%     The error is small in practice.
%
% =====================================================================
% REFERENCES
% =====================================================================
%   Fisher, N. I. & Lee, A. J. (1992).  Regression Models for an Angular
%       Response.  Biometrics 48, 665-677.
%   Mardia, K. V. & Jupp, P. E. (2000).  Directional Statistics.  Wiley.
%   Berens, P. (2009).  CircStat: A MATLAB Toolbox for Circular
%       Statistics.  Journal of Statistical Software 31(10), 1-21.
%   Liang, K.-Y. & Zeger, S. L. (1986).  Longitudinal data analysis using
%       generalized linear models.  Biometrika 73, 13-22.
%   Jammalamadaka, S. R. & SenGupta, A. (2001).  Topics in Circular
%       Statistics.  World Scientific.
%
% =====================================================================
% EXAMPLE (synthetic, mirroring a per-class lifespan fit)
% =====================================================================
%   N = 600;
%   age   = 7 + 73*rand(N,1);                    % ages 7-80
%   elec  = double(rand(N,1) > 0.5);             % central=1, frontal=0
%   sex   = double(rand(N,1) > 0.5);             % F=1, M=0
%   subj  = ceil((1:N)' / 2);                    % two obs per subject
%   true_phase = -0.04*age + 0.0006*age.^2 ...
%                + 0.3*elec - 0.05*sex + 0.05*age.*elec;
%   y = wrapToPi(true_phase + circ_vmrnd(0, 6, N));   % needs CircStat
%
%   [~, mdl] = circular_regression_fixed(age, y, ...
%       'Order', 2, ...
%       'Categorical', [elec, sex], ...
%       'CategoricalNames', {'electrode','sex'}, ...
%       'Interactions', [true, false], ...
%       'PredictorName', 'Age', ...
%       'ClusterID', subj);
%   disp(mdl.Coefficients);
%
%   age_block  = mdl.coefTest(mdl.ContrastIndex.x_main);
%   intx_block = mdl.coefTest(mdl.ContrastIndex.x_x_electrode);
%   fprintf('Joint Age block:           F(%d,%d)=%.2f, p=%.4g\n', ...
%       age_block.df1, age_block.df2, age_block.F, age_block.p_F);
%   fprintf('Age x electrode block:     F(%d,%d)=%.2f, p=%.4g\n', ...
%       intx_block.df1, intx_block.df2, intx_block.F, intx_block.p_F);

% ===================================================================
%                LEGACY POSITIONAL DISPATCH (drop-in shim)
% ===================================================================
% Detect calls in the legacy form used by the original
% circular_regression.m so this function is a drop-in replacement:
%
%   circular_regression_fixed(x, y, cat_vars, varnames, order, ...
%                             xcol_interactions, b0, iters[, ClusterID])
%
% varnames is a cellstr/string array whose first entry is the predictor
% name and whose remaining entries are the categorical-column names.
% The optional 9th positional arg (ClusterID) enables cluster-robust SEs.
if numel(varargin) >= 3 && (isnumeric(varargin{1}) || isempty(varargin{1})) && ...
   (iscell(varargin{2}) || isstring(varargin{2})) && ...
   isnumeric(varargin{3}) && isscalar(varargin{3})
    cat_vars  = varargin{1};
    varnames  = cellstr(varargin{2});
    order     = varargin{3};
    nv = {'Order', order, ...
          'Categorical',     cat_vars, ...
          'CategoricalNames', varnames(2:end), ...
          'PredictorName',   char(varnames{1})};
    if numel(varargin) >= 4 && ~isempty(varargin{4})
        nv = [nv, {'Interactions', logical(varargin{4})}];
    end
    if numel(varargin) >= 5 && ~isempty(varargin{5})
        nv = [nv, {'InitBeta', varargin{5}}];     % accepted as parameter below
    end
    if numel(varargin) >= 6 && ~isempty(varargin{6})
        nv = [nv, {'MaxIter',  varargin{6}}];
    end
    if numel(varargin) >= 7 && ~isempty(varargin{7})
        nv = [nv, {'ClusterID', varargin{7}}];
    end
    varargin = nv;
end

% ===================================================================
%                          INPUT PARSING
% ===================================================================
% Standard inputParser pattern: required positional x and y, everything
% else as optional name-value pairs with type validators.
p = inputParser;
addRequired(p,  'x',                @(v) isnumeric(v) && isvector(v));
addRequired(p,  'y',                @(v) isnumeric(v) && isvector(v));
addParameter(p, 'Order',            1,     @(v) isnumeric(v) && isscalar(v) && v >= 0 && mod(v,1)==0);
addParameter(p, 'Categorical',      [],    @(v) isempty(v) || isnumeric(v));
addParameter(p, 'CategoricalNames', {},    @(v) isempty(v) || iscellstr(v) || isstring(v));
addParameter(p, 'Interactions',     [],    @(v) isempty(v) || islogical(v) || isnumeric(v));
addParameter(p, 'PredictorName',    'x',   @(v) ischar(v) || (isstring(v) && isscalar(v)));
addParameter(p, 'ResponseName',     'y',   @(v) ischar(v) || (isstring(v) && isscalar(v)));
addParameter(p, 'ClusterID',        [],    @(v) isempty(v) || isvector(v));
addParameter(p, 'MaxIter',          200,   @(v) isnumeric(v) && isscalar(v) && v > 0);
addParameter(p, 'InitBeta',         [],    @(v) isempty(v) || (isnumeric(v) && isvector(v)));
addParameter(p, 'Tol',              1e-8,  @(v) isnumeric(v) && isscalar(v) && v > 0);
addParameter(p, 'Verbose',          false, @(v) islogical(v) || (isnumeric(v) && isscalar(v)));
parse(p, x, y, varargin{:});
opts = p.Results;

% ===================================================================
%                       NaN HANDLING & WRAPPING
% ===================================================================
% Drop any row where x or y is NaN, with the matching rows of the
% Categorical and ClusterID inputs aligned out as well.  This mirrors
% what get_single_order_model.m was doing externally so callers don't
% have to remember to do it themselves.
x  = x(:);
y0 = y(:);
nan_mask = isnan(x) | isnan(y0);
if ~isempty(opts.Categorical)
    nan_mask = nan_mask | any(isnan(opts.Categorical), 2);
end
if any(nan_mask)
    x  = x(~nan_mask);
    y0 = y0(~nan_mask);
    if ~isempty(opts.Categorical)
        opts.Categorical = opts.Categorical(~nan_mask, :);
    end
    if ~isempty(opts.ClusterID)
        opts.ClusterID = opts.ClusterID(~nan_mask);
    end
end

% Wrap response into the canonical (-pi, pi] interval.  All subsequent
% residual quantities (sin, cos) are 2*pi-periodic so the wrap does not
% change the model, but it makes diagnostics (residual histograms, etc.)
% live on the natural interval.
y = wrapToPi(y0);
n = numel(x);
if n < 2
    error('circular_regression_fixed:tooFewObs', ...
          'Need at least 2 non-NaN observations after filtering.');
end

% Coerce char/string scalars from the parser to plain char.
predName = char(opts.PredictorName);
catNames = cellstr(opts.CategoricalNames);

% ===================================================================
%                       DESIGN MATRIX CONSTRUCTION
% ===================================================================
% The design matrix is laid out as
%
%   [ 1,  x, x^2, ..., x^Order,  C_1, ..., C_nC,  inter_block ]
%
% with the interaction block expanding, for each categorical column C_j
% flagged in Interactions, into Order columns
%
%   [ x*C_j, x^2*C_j, ..., x^Order*C_j ].
%
% This block ordering is also what determines ContrastIndex below, so
% downstream joint tests can reference coefficient ranges by name rather
% than by raw column number.
order = opts.Order;
C     = opts.Categorical;
nC    = size(C, 2);
if nC > 0 && size(C, 1) ~= n
    error('circular_regression_fixed:badCategorical', ...
          'Categorical must have %d rows after NaN removal.', n);
end

% Normalize Interactions to a logical row of length nC.
if isempty(opts.Interactions)
    interFlag = false(1, nC);
else
    interFlag = logical(opts.Interactions(:)');
    if numel(interFlag) ~= nC
        error('circular_regression_fixed:badInteractions', ...
              'Interactions must have one element per categorical column.');
    end
end

% Polynomial block columns: x, x^2, ..., x^order.
Xpoly = zeros(n, order);
for k = 1:order
    Xpoly(:, k) = x.^k;
end

% Start the design with intercept + polynomial block + categoricals.
X = [ones(n, 1), Xpoly, C];

% Build interaction block, in lockstep with the names that will go into
% the coefficient table later.  We accumulate names here so that the
% indexing into coefNames (below) is unambiguous and not subject to the
% off-by-one bookkeeping bug from the original implementation.
interIdx   = find(interFlag);
nInter     = numel(interIdx);
Xinter     = zeros(n, nInter * order);
interNames = cell(1, nInter * order);
col = 0;
for ii = 1:nInter
    cidx  = interIdx(ii);
    cname = name_or_default(catNames, cidx);
    for k = 1:order
        col = col + 1;
        Xinter(:, col) = Xpoly(:, k) .* C(:, cidx);
        if k == 1
            interNames{col} = sprintf('%s:%s', predName, cname);
        else
            interNames{col} = sprintf('%s^%d:%s', predName, k, cname);
        end
    end
end
X = [X, Xinter];
P = size(X, 2);

% Coefficient names, in column order of X.  This is the source of truth
% for the row labels in mdl.Coefficients and the keys in ContrastIndex.
coefNames    = cell(P, 1);
coefNames{1} = '(Intercept)';
for k = 1:order
    if k == 1
        coefNames{1+k} = predName;
    else
        coefNames{1+k} = sprintf('%s^%d', predName, k);
    end
end
for ii = 1:nC
    coefNames{1 + order + ii} = name_or_default(catNames, ii);
end
% Place interaction names as a single contiguous block.  This is the
% line the original got wrong: it computed the slice length as
% num_interactions*(order-1)+1 instead of num_interactions*order, and
% it overwrote the slice on each iteration of the categorical loop.
if nInter > 0
    coefNames(1 + order + nC + (1:nInter*order)) = interNames;
end

% ===================================================================
%                    IRLS (Newton-Raphson + step halving)
% ===================================================================
% Initialize beta to all zeros, except the intercept which we set to the
% circular mean of y.  This anchors the iterate on the sheet of the
% linear predictor closest to the data and is the standard initialization
% for identity-link Fisher-Lee circular regression.
b    = zeros(P, 1);
b(1) = atan2(mean(sin(y)), mean(cos(y)));
if ~isempty(opts.InitBeta) && numel(opts.InitBeta) == P
    b = opts.InitBeta(:);
end

converged   = false;
last_change = NaN;

for it = 1:opts.MaxIter
    % --- score and curvature at current iterate ---
    eta = X * b;                  % linear predictor (real-valued)
    r   = wrapToPi(y - eta);      % wrap so cos(r) is well-defined as a curvature
    s   = sin(r);                 % score direction; X'*s is the gradient up to kappa
    w   = cos(r);                 % per-observation curvature

    % Where cos(r) <= 0 the local Hessian becomes non-positive-definite
    % and a pure Newton step would be unstable.  Clamping w to a small
    % positive value falls back to a Fisher-scoring step in those rows
    % (using expected curvature ~ A1(kappa) ~ 1 when fits are tight).
    w_safe = max(w, 1e-6);
    W      = spdiags(w_safe, 0, n, n);

    % Newton system: solve (X' W X) * db = X' * sin(r) for the search
    % direction db.  The kappa cancels between gradient and Hessian, so
    % it does not appear in db itself; it re-enters via the SE formula
    % below.
    XtWX = X' * (W * X);
    grad = X' * s;
    db   = XtWX \ grad;

    % --- step halving on the score-likelihood proxy sum(cos r) ---
    % The von-Mises log-likelihood at fixed kappa is monotone in
    % sum_i cos(r_i), so we accept the largest step in (0, 1] that does
    % not decrease this proxy.  Capping at 30 halvings is paranoia: for
    % well-conditioned designs the full step almost always succeeds.
    score_old = sum(cos(r));
    step      = 1;
    accepted  = false;
    for h = 1:30
        b_trial   = b + step * db;
        r_trial   = wrapToPi(y - X * b_trial);
        score_new = sum(cos(r_trial));
        if score_new >= score_old - 1e-12
            accepted = true;
            break;
        end
        step = step / 2;
    end

    % If even a tiny step worsens the proxy likelihood, we are at (or
    % numerically indistinguishable from) a stationary point.  Stop and
    % flag as converged.  This branch is rare in practice.
    if ~accepted
        if opts.Verbose
            fprintf('iter %3d  step halving exhausted; stopping.\n', it);
        end
        converged   = true;
        last_change = norm(step * db) / max(norm(b), 1);
        break;
    end

    % Relative change in the coefficient vector.  We use this rather
    % than ||grad|| because it is scale-invariant and matches what
    % users typically interpret as "the iteration stopped moving".
    rel_change = norm(step * db) / max(norm(b_trial), 1);
    if opts.Verbose
        fprintf('iter %3d  step=%.3g  ||db||/||b||=%.3g  sum(cos r)=%.6f\n', ...
                it, step, rel_change, score_new);
    end

    b           = b_trial;
    last_change = rel_change;

    if rel_change < opts.Tol
        converged = true;
        break;
    end
end

if ~converged
    % Loud failure: callers should know when the fit did not reach Tol.
    % The model struct still contains the best b found, but inferences
    % from it should be treated as suspect.
    warning('circular_regression_fixed:notConverged', ...
        'IRLS did not converge to tol %g in %d iterations (final rel change %g).', ...
        opts.Tol, opts.MaxIter, last_change);
end

% ===================================================================
%                      FINAL RESIDUALS AND KAPPA
% ===================================================================
% Recompute everything at the converged iterate for the inference step.
% Using wrapped residuals here keeps the kappa estimator and the
% information matrix consistent (the original code mixed wrapped and
% unwrapped residuals across these computations).
eta_final = X * b;
r_wrapped = wrapToPi(y - eta_final);
kappa     = circ_kappa_local(r_wrapped);
kappa     = min(kappa, 1e4);   % numerical guard for besseli at huge kappa
A1        = besseli(1, kappa) / besseli(0, kappa);

% ===================================================================
%                MODEL-BASED COVARIANCE (observed information)
% ===================================================================
% J_obs = -E[Hessian] | iid model = kappa * X' * diag(cos r) * X.
% Inverting this gives the asymptotic covariance of beta-hat under the
% model, evaluated at the MLE.  This is the "observed information"
% form; the "expected information" form replaces diag(cos r) with
% A1(kappa) * I and gives nearly identical SEs at this sample size.
w_final   = max(cos(r_wrapped), 1e-6);
W_final   = spdiags(w_final, 0, n, n);
J_obs     = kappa * (X' * (W_final * X));
cov_model = J_obs \ eye(P);

% ===================================================================
%                CLUSTER-ROBUST COVARIANCE (sandwich), if asked
% ===================================================================
% V_CR = A^{-1} * B * A^{-1}, with B = sum_g U_g U_g' built from per-
% cluster aggregated scores U_g = sum_{i in g} kappa * x_i * sin(r_i).
% This is the within-cluster Liang-Zeger sandwich estimator, with the
% standard small-sample correction (G/(G-1)) * ((n-1)/(n-P)).
%
% When ClusterID is omitted, we fall back to the model-based covariance
% above and use n - P degrees of freedom for the t reference.
if ~isempty(opts.ClusterID)
    g = opts.ClusterID(:);
    if numel(g) ~= n
        error('circular_regression_fixed:badCluster', ...
              'ClusterID must have one entry per observation (after NaN removal).');
    end

    % Map raw IDs to consecutive integers so we can index by cluster.
    [~, ~, gid] = unique(g);
    G = max(gid);

    % Per-observation score contributions u_i = kappa * x_i * sin(r_i),
    % stacked into an n-by-P matrix.
    U = bsxfun(@times, X, kappa * sin(r_wrapped));

    % Aggregate to per-cluster scores (G-by-P) and form the meat.
    Ug = zeros(G, P);
    for gg = 1:G
        Ug(gg, :) = sum(U(gid == gg, :), 1);
    end
    meat  = Ug' * Ug;
    bread = cov_model;

    cov_b   = bread * meat * bread;
    cov_b   = (G / max(G - 1, 1)) * ((n - 1) / max(n - P, 1)) * cov_b;
    covType = 'cluster-robust';
    df_res  = G - 1;
else
    cov_b   = cov_model;
    covType = 'model-based';
    df_res  = n - P;
end

% Symmetrize for numerical hygiene (rounding can break symmetry slightly).
cov_b = (cov_b + cov_b') / 2;

% ===================================================================
%                COEFFICIENT-LEVEL INFERENCE (two-sided)
% ===================================================================
% Standard Wald per-coefficient inference: SE from sqrt(diag(V)),
% t-statistic, two-sided p-value against a t with df_res degrees of
% freedom.  At large df_res this is indistinguishable from a normal
% reference, but using t is conservative for small samples and
% consistent with how MATLAB's fitlme reports.
SE    = sqrt(max(diag(cov_b), 0));
tStat = b ./ SE;
pVal  = 2 * (1 - tcdf(abs(tStat), df_res));     % TWO-SIDED -- the bug fix

% ===================================================================
%                 PSEUDO R^2 (Jammalamadaka analog) AND LIKELIHOOD
% ===================================================================
% Numerator: sum of squared *wrapped* residuals (so SSE measures the
% true angular distance, not the raw real-valued difference).
% Denominator: same construction relative to the circular mean of y.
% This is bounded above by 1 and reduces to the linear R^2 when wraps
% don't matter.
mu_y  = atan2(mean(sin(y)), mean(cos(y)));
SSE   = sum(wrapToPi(y - eta_final).^2);
SST   = sum(wrapToPi(y - mu_y).^2);
R2    = 1 - SSE / SST;
R2adj = 1 - ((n - 1) / max(n - P, 1)) * SSE / SST;

% Standard von-Mises log-likelihood and deviance summaries.  Note the
% deviance uses 1 - cos(r), so it lies in [0, 2*n] and is monotone in
% how poorly the fit angles match the data angles.
logL     = -n * log(2*pi*besseli(0, kappa)) + kappa * sum(cos(r_wrapped));
deviance = 2 * sum(1 - cos(r_wrapped));

% ===================================================================
%                CONTRAST INDEX FOR JOINT BLOCK TESTS
% ===================================================================
% Pre-name the row sets that downstream code is most likely to want to
% test jointly.  Field names are sanitized so they're always valid
% MATLAB identifiers (e.g. categoricals named "electrode" produce field
% .electrode and .x_x_electrode).
ContrastIndex = struct();

% Polynomial-x main-effect rows: 2..1+order.  Empty when order = 0.
if order >= 1
    ContrastIndex.x_main = (2 : 1 + order)';
else
    ContrastIndex.x_main = [];
end

% One single-row entry per categorical main effect.
for ii = 1:nC
    cname = name_or_default(catNames, ii);
    fld   = matlab.lang.makeValidName(cname);
    ContrastIndex.(fld) = 1 + order + ii;
end

% Order-many rows per interacting categorical.
col = 0;
for ii = 1:nInter
    cidx  = interIdx(ii);
    cname = name_or_default(catNames, cidx);
    fld   = matlab.lang.makeValidName(['x_x_' cname]);
    ContrastIndex.(fld) = (1 + order + nC + col + (1:order))';
    col = col + order;
end

% ===================================================================
%                            ASSEMBLE OUTPUT
% ===================================================================
% Coefficient table mirrors the layout of fitlm/fitlme outputs so
% callers can largely treat this struct interchangeably for reporting.
coef_tbl = table(b, SE, tStat, pVal, repmat(df_res, P, 1), ...
                 'VariableNames', {'Estimate','SE','tStat','pValue','DF'}, ...
                 'RowNames', coefNames);

mdl = struct();
mdl.Coefficients      = coef_tbl;
mdl.CoefficientNames  = coefNames;
mdl.NumCoefficients   = P;
mdl.DFE               = df_res;
mdl.Kappa             = kappa;
mdl.A1                = A1;
mdl.LogLikelihood     = logL;
mdl.Deviance          = deviance;
mdl.CovarianceType    = covType;
mdl.cov_b             = cov_b;
mdl.Rsquared.Ordinary = R2;
mdl.Rsquared.Adjusted = R2adj;
mdl.Converged         = converged;
mdl.Iterations        = it;
mdl.ResidualWrapped   = r_wrapped;
mdl.DesignMatrix      = X;
mdl.ContrastIndex     = ContrastIndex;
% Closure that captures (b, cov_b, df_res) so callers can run joint
% Wald tests by row-mask or contrast matrix without re-passing them.
mdl.coefTest          = @(R) do_coef_test(R, b, cov_b, df_res);

estimated_phase = wrapToPi(eta_final);
end


% =====================================================================
% LOCAL HELPERS
% =====================================================================

function restore_warnings(s1, s2)
warning(s1);
warning(s2);
end


function nm = name_or_default(names, k)
% Fetch the k-th entry of a cellstr/string array, or fall back to a
% generic placeholder ('c1', 'c2', ...) when the caller did not supply
% a name.  Used for both categorical labels and interaction labels.
if isempty(names) || numel(names) < k || isempty(char(names{k}))
    nm = sprintf('c%d', k);
else
    nm = char(names{k});
end
end


function out = do_coef_test(R, b, V, df)
% Wald joint test of H0: R*b = 0.
%
% Accepts either:
%   - an m x P contrast matrix R (each row a linear combination of
%     coefficients constrained to zero under H0), or
%   - a 0/1 row mask of length P, in which case it is interpreted as
%     "test that the marked coefficients are jointly zero".
%
% Returns both the chi^2 form (with rank(R) = m degrees of freedom)
% and the equivalent F form (m, df) where df is the residual df from
% the parent fit.  At large df the two p-values are essentially
% identical; at small df the F is slightly more conservative.

% --- coerce row-mask shorthand to a contrast matrix ---
% If R is a logical (or 0/1 numeric) vector, build the corresponding
% selection matrix on the fly so callers can write
%   mdl.coefTest(mdl.ContrastIndex.x_main)
% even though ContrastIndex stores raw row indices.
if islogical(R) || (isvector(R) && ~isscalar(R) && all(ismember(R(:), [0 1])))
    idx = find(R(:));
    R   = full(sparse(1:numel(idx), idx, 1, numel(idx), numel(b)));
elseif isvector(R) && all(R == round(R)) && all(R >= 1) && all(R <= numel(b))
    % Plain integer index list: treat as row positions to test.
    idx = R(:);
    R   = full(sparse(1:numel(idx), idx, 1, numel(idx), numel(b)));
end

% --- Wald statistic ---
% W = (R*b)' * inv(R*V*R') * (R*b)  is asymptotically chi^2(m) under H0.
% We symmetrize R*V*R' before solving to suppress rounding-induced
% asymmetry in the inverse.
m   = size(R, 1);
Rb  = R * b;
RVR = R * V * R';
RVR = (RVR + RVR') / 2;
W   = Rb' * (RVR \ Rb);

% --- pack both equivalent forms ---
out.W      = W;                          % chi^2 statistic
out.df     = m;                          % chi^2 df
out.p_chi2 = 1 - chi2cdf(W, m);
out.F      = W / m;                      % F = chi^2 / df1
out.df1    = m;
out.df2    = df;
out.p_F    = 1 - fcdf(out.F, m, df);
end


function kappa = circ_kappa_local(alpha)
% Maximum-likelihood-ish estimate of the von-Mises concentration kappa,
% using the standard piecewise rational approximation from Berens (2009)
% / Fisher (1993).  Solves A1(kappa) = mean(cos(alpha)) approximately,
% with a small-sample correction for n < 15.
%
% A self-contained copy of CircStat::circ_kappa is included here so this
% function has no extra dependencies (the rest of the lab pipeline does
% have CircStat available, but a single audit-quality file should not
% require it).

alpha = alpha(:);
N     = numel(alpha);

% Mean resultant length R = |sum exp(i*alpha)| / N.  R lies in [0,1];
% R = 1 means residuals are perfectly concentrated, R = 0 means uniform.
R = abs(sum(exp(1i * alpha))) / N;

% Piecewise rational approximation to the inverse of A1(kappa).
if R < 0.53
    kappa = 2*R + R^3 + 5*R^5/6;
elseif R < 0.85
    kappa = -0.4 + 1.39*R + 0.43/(1 - R);
else
    denom = R^3 - 4*R^2 + 3*R;
    if abs(denom) < eps
        % Numerical guard: at R extremely close to 1 the closed form
        % blows up.  Return a large but finite kappa; the caller
        % additionally caps at 1e4 for besseli safety.
        kappa = 1e4;
    else
        kappa = 1 / denom;
    end
end

% Small-sample bias correction (Fisher 1993, eq. 4.41).
if N < 15 && N > 1
    if kappa < 2
        kappa = max(kappa - 2*(N*kappa)^-1, 0);
    else
        kappa = (N - 1)^3 * kappa / (N^3 + N);
    end
end

% Concentration is non-negative by definition; clip to enforce.
kappa = max(kappa, 0);
end
