classdef fitcirc_lme
%FITCIRC_LME  Mixed-effects regression for circular data, von Mises GLMM.
%
% The outcome here is an ANGLE (for example the phase of an oscillation),
% so ordinary linear regression does not apply: an angle near +pi and one
% near -pi are close together, not far apart. Each observation belongs to
% a subject, and every subject is allowed its own baseline angular offset
% (a random intercept on the circle). The model is
%
%   y_ij | beta, phi_i  ~  vonMises( X_ij*beta + phi_i,  kappa )
%   phi_i               ~  vonMises( 0,  kappa_phi )
%
% Reading the symbols:
%   y_ij       angle for observation j of subject i (radians)
%   X_ij*beta  the fixed-effect prediction -- the population-level trend
%   phi_i      subject i's own angular offset -- the random intercept
%   kappa      how tightly observations cluster around their prediction
%              (large kappa = little noise). This is the von Mises
%              "concentration", the circular cousin of 1/variance.
%   kappa_phi  how tightly the subject offsets cluster around 0 (large
%              kappa_phi = subjects are alike). It plays the role that the
%              random-intercept variance plays in an ordinary mixed model,
%              but measured on the circle.
%
% The von Mises distribution is the circular analogue of the Normal. A
% useful one-number summary of any cluster of angles is the mean
% resultant length
%   A(k) = I_1(k) / I_0(k),   between 0 and 1
% (I_0, I_1 are modified Bessel functions). It is the average of
% cos(angle - mean direction): near 0 when angles are spread all around
% the circle, near 1 when they are tightly bunched.
%
% Why both pieces are von Mises: it keeps the per-subject math exact.
% Given the data and the current parameters, each offset phi_i again
% follows a von Mises distribution that we can write down directly -- no
% Gaussian (Laplace) approximation, and no need to "unwrap" angles onto
% the real line.
%
% --------------------------------------------------------------------
% HOW IT IS FIT (EM algorithm)
% --------------------------------------------------------------------
% The fit alternates two steps until the log-likelihood stops changing.
%
% E-STEP -- summarize each subject's offset.
%   For subject i, remove the fixed-effect prediction and add up the
%   leftover angles as unit vectors:
%     alpha_ij = y_ij - X_ij*beta
%     C_i = sum_j cos(alpha_ij),   S_i = sum_j sin(alpha_ij)
%   Combining subject i's data with the prior vonMises(0, kappa_phi)
%   yields the offset's distribution -- again von Mises -- with
%     K_post_i  = sqrt( (kappa*C_i + kappa_phi)^2 + (kappa*S_i)^2 )
%     mu_post_i = atan2( kappa*S_i,  kappa*C_i + kappa_phi )
%   mu_post_i is the best estimate of subject i's offset, and
%     rho_i = A(K_post_i)
%   says how sure we are of it (near 1 = very sure). rho_i is all the
%   M-step needs from this step.
%
% M-STEP -- update the parameters, carrying the offset uncertainty
% through rho_i.
%   1) beta (the population trend). Solve the circular regression
%        sum_ij rho_i * X_ij' * sin(y_ij - X_ij*beta - mu_post_i) = 0
%      by iteratively reweighted least squares. Each observation is
%      weighted by its subject's rho_i, so subjects whose offset we are
%      sure of count fully and uncertain ones count less.
%   2) kappa (response concentration). Form the residual angles
%      y_ij - X_ij*beta - mu_post_i, average their unit vectors with the
%      same rho_i weights, and turn that mean resultant length into a
%      concentration (local_kappa_from_R). Weighting by rho_i accounts for
%      the offsets being known only approximately.
%   3) kappa_phi (how alike the subjects are). See the next block.
%
% --------------------------------------------------------------------
% ESTIMATING kappa_phi, AND WHY IT NEEDS A PRIOR
% --------------------------------------------------------------------
% With the offset prior centered at 0, the natural estimate solves
%   A(kappa_phi) = R_phi,   R_phi = mean_i rho_i * cos(mu_post_i),
% i.e. it measures how tightly the estimated offsets bunch around 0. It is
% found by maximizing the exact one-dimensional objective
%   kappa_phi = argmax_k  n_subj*( R_phi*k - log I_0(k) ) + log prior(k).
%
% The "log prior(k)" term is there for a specific failure mode. When the
% fixed effects already capture the trend well, every subject's estimated
% offset sits almost exactly at 0, so R_phi -> 1. The plain estimate of
% kappa_phi then runs off toward infinity (equivalently the subject-to-
% subject spread sigma_phi -> 0). Numerically that overflows the Bessel
% functions and sends the log-likelihood to NaN, which breaks comparison
% across nested models. A gentle, weakly-informative prior on kappa_phi
% keeps the estimate finite and the log-likelihood smooth, so likelihood-
% ratio tests and AIC across models stay well behaved. When the data
% genuinely pin kappa_phi down the prior barely moves it; it only tames
% the runaway case. See the KappaPhiPrior / KappaPhiPriorScale options.
% (When R_phi <= 0 the estimate is kappa_phi = 0: the offsets do not bunch
% around 0 at all, so the prior is effectively uniform on the circle.)
%
% A note on which boundary. Gelman (2006) recommends a half-t / half-
% Cauchy prior on a group-level standard deviation for the case of too FEW
% groups, where the spread can come out implausibly LARGE; that prior has
% its mode at zero spread. Here the troublesome boundary is the opposite
% one -- zero spread, kappa_phi -> infinity -- so the same weakly-
% informative idea is placed on kappa_phi itself, where it pushes back
% against the runaway.
%
% --------------------------------------------------------------------
% LOG-LIKELIHOOD
% --------------------------------------------------------------------
% Because each subject's offset can be integrated out exactly, the
% marginal log-likelihood is available in closed form:
%   log p(y_i) = log I_0(K_post_i)
%              - n_i * log( 2*pi * I_0(kappa) )
%              -        log( I_0(kappa_phi) )
% summed over subjects, with n_i the number of observations for subject i.
% This (unpenalized) value is what LogLikelihood reports and what model
% comparisons should use. The EM stops when its relative change drops
% below Tol, or after MaxIter iterations.
%
% The M-step updates are only approximate maximizers, so a full step can
% occasionally overshoot and lower the likelihood. To prevent that, each
% step is backtracked (step-halved) until it does not decrease the
% objective. The EM is therefore monotone: the log-likelihood never goes
% down from one iteration to the next, which also guarantees that a larger
% model warm-started from a smaller nested one cannot end up with a lower
% likelihood -- the property model-selection (LRT/AIC) relies on.
%
% --------------------------------------------------------------------
% USAGE
% --------------------------------------------------------------------
%   mdl = fitcirc_lme(tbl, 'phase ~ 1 + Age + (1|Subj_ID)')
%   mdl = fitcirc_lme(tbl, formula, Name, Value, ...)
%
% NAME-VALUE OPTIONS
%   'MaxIter'      (default 500)   max EM iterations
%   'Tol'          (default 1e-5)  relative LL tolerance for convergence
%   'Verbose'      (default false) print per-iteration progress
%   'InitKappa'    (default 4)     starting kappa
%   'InitKappaPhi' (default 4)     starting kappa_phi (prior concentration)
%   'InitSigma'    (deprecated)    if given, converted to InitKappaPhi via
%                                  A(kappa_phi) = exp(-sigma^2/2)
%   'KappaPhiPrior' (default 'halfcauchy') weakly-informative prior on the
%                                  subject-spread concentration kappa_phi,
%                                  which keeps its estimate finite (see the
%                                  kappa_phi notes above). One of
%                                  'halfcauchy', 'halfnormal', or 'none'.
%   'KappaPhiPriorScale' (default 8) scale of that prior, on the kappa_phi
%                                  axis. Larger = weaker pull = more
%                                  subject-to-subject spread allowed.
%   'KappaPhiMax'  (default Inf)   optional hard upper limit on kappa_phi,
%                                  applied after the prior. Inf leaves the
%                                  prior in charge; a finite value clamps
%                                  kappa_phi at that ceiling.
%
% OUTPUT FIELDS
%   Formula, ResponseName, GroupingVar
%   Coefficients          - table: Name, Estimate, SE, tStat, pValue
%   Beta                  - p-vector of fixed effects
%   Kappa                 - scalar concentration of the response noise
%   KappaPhi              - scalar concentration of the subject-phase prior
%   SigmaPhi              - equivalent circular SD,
%                           sqrt( -2 * log( A(KappaPhi) ) )
%   PhiHat                - n_subj-by-1 posterior mean directions
%   PhiRho                - n_subj-by-1 posterior mean resultant lengths
%   PhiKappaPost          - n_subj-by-1 posterior concentrations
%   LogLikelihood         - exact marginal LL at the converged params
%                           (UNpenalized; comparable across nested models)
%   LogPrior              - log kappa_phi prior at the converged params;
%                           penalized objective = LogLikelihood + LogPrior
%   AIC, BIC              - based on (p + 2) parameters
%   NumObservations, NumSubjects, NumCoefficients
%   ConvergedIn, ConvergenceHistory
%
% METHODS
%   predict(newdata, ...) - predicted angle X*beta wrapped to [-pi, pi]
%                           ('Conditional', false) or X*beta + phi_hat
%                           with subject lookup ('Conditional', true)
%   coefTest(R)           - Wald joint test on a contrast of beta
%
% STANDARD ERRORS
% Standard errors on beta use a cluster-robust ("sandwich") estimator with
% each subject treated as one cluster, so they remain valid even though a
% subject's repeated observations are correlated. The sandwich is built
% from the same rho-weighted circular score the M-step solves, and its
% "bread" is the expected (Fisher) information
%   kappa * A(kappa) * sum_ij rho_i * X_ij' * X_ij,
% which is always positive-definite. A small-sample correction factor
%   m/(m-1) * (n-1)/(n-p)
% is applied (m = number of subjects, n = number of observations,
% p = number of fixed effects), and tests use a t-distribution with
% (m - 1) degrees of freedom. With only a couple of observations per
% subject these SEs can be mildly optimistic; for primary inference a
% cluster-robust von Mises fit without the random intercept is a
% conservative alternative.
%
% LIMITATIONS
%   - Single (1|group) random-intercept term only.
%   - Cluster-robust SEs need a reasonable number of subjects.
%
% REFERENCES
%   Stram, D.O. & Lee, J.W. (1994). Variance components testing in the
%     longitudinal mixed effects model. Biometrics 50(4):1171-1177.
%     When a variance component is zero it lies on the boundary of the
%     parameter space, so the usual chi-squared reference distribution for
%     the likelihood-ratio statistic does not apply. This is why the
%     kappa_phi -> infinity boundary above needs care.
%   Gelman, A. (2006). Prior distributions for variance parameters in
%     hierarchical models. Bayesian Analysis 1(3):515-533.
%     Recommends a weakly-informative half-t / half-Cauchy prior for a
%     group-level standard deviation. The same idea is used here, placed
%     on kappa_phi (see "a note on which boundary", above).

    properties
        Formula
        ResponseName
        GroupingVar
        Coefficients
        Beta
        Kappa
        KappaPhi
        SigmaPhi
        PhiHat
        PhiRho
        PhiKappaPost
        LogLikelihood
        LogPrior
        AIC
        BIC
        NumObservations
        NumSubjects
        NumCoefficients
        ConvergedIn
        ConvergenceHistory
        DesignNames
        SubjectIDs
        X_design
        TrainingData
        cov_b
        DFE
        ContrastIndex
        Rsquared
        % Variance-minimizing circular shift applied to the response
        % before fitting. Predictions add it back and wrap to (-pi, pi].
        % Default 0 means "fit on the data as given".
        ThetaShift = 0
    end

    methods
        function obj = fitcirc_lme(tbl, formula, varargin)
            % --- Parse name-value options ---
            p = inputParser;
            p.addParameter('MaxIter',  500);
            p.addParameter('Tol',      1e-5);
            p.addParameter('Verbose',  false);
            p.addParameter('InitKappa',     4);
            p.addParameter('InitKappaPhi',  4);
            p.addParameter('InitSigma',    []);  % deprecated; converted if given
            % Weakly-informative prior on the subject-spread concentration
            % kappa_phi. It guards against a specific failure: when the
            % fixed effects absorb the population trend, each subject's
            % estimated offset collapses near 0, R_phi -> 1, and the plain
            % estimate of kappa_phi runs to infinity (the sigma_phi -> 0
            % boundary; see Stram & Lee 1994 on variance components at a
            % boundary). At that point I0(kappa_phi) overflows and the
            % marginal log-likelihood becomes NaN, which would make
            % nested-model comparisons (e.g. the order-selection LRT)
            % unreliable. The prior is a smooth half-Cauchy (by default) on
            % kappa_phi, folded into the exact 1-D M-step; it keeps the
            % estimate finite and the log-likelihood smooth. The reported
            % LogLikelihood is the unpenalized marginal value at the fitted
            % kappa_phi, so it stays comparable across models. The scale
            % defaults to 8 (prior median kappa_phi ~ 8, i.e. a subject
            % spread sigma_phi ~ 0.36 rad / 21 deg).
            p.addParameter('KappaPhiPrior',      'halfcauchy');
            p.addParameter('KappaPhiPriorScale', 8);
            % Optional hard upper limit on kappa_phi, applied after the
            % prior. Inf (default) leaves the prior in charge; a finite
            % value simply clamps via kappa_phi = min(estimate, KappaPhiMax).
            p.addParameter('KappaPhiMax', inf);
            % AutoShift = true: rotate y by the variance-minimizing shift
            % so wrapping doesn't split the response near +/- pi. The shift
            % is recorded in ThetaShift and added back during predict().
            p.addParameter('AutoShift', false);
            p.addParameter('ThetaShift', []);    % override AutoShift with explicit value
            % Warm-start support: callers (e.g. circular_regression's order
            % loop) can pass a previous order's converged Beta + coefficient
            % names. We copy estimates by NAME into the new column basis
            % (columns absent from the prior fit start at 0), giving the
            % EM a much better initial guess for higher orders. Without
            % this, IRLS cold-starts at all-zeros and higher-order fits
            % routinely converge to LL worse than the intercept-only fit.
            p.addParameter('InitBeta',      []);  % numeric vector (matching InitBetaNames)
            p.addParameter('InitBetaNames', {});  % cellstr of coefficient names
            p.parse(varargin{:});
            opt = p.Results;
            if ~isempty(opt.InitSigma)
                opt.InitKappaPhi = local_invA( exp(-opt.InitSigma^2/2) );
            end

            % --- Parse formula: response, fixed-effects RHS, grouping var ---
            formula  = char(formula);
            tildeIdx = strfind(formula, '~');
            assert(~isempty(tildeIdx), 'formula must contain ~');
            respName = strtrim(formula(1:tildeIdx(1)-1));
            assert(ismember(respName, tbl.Properties.VariableNames), ...
                'response variable "%s" not found', respName);

            grpTok = regexp(formula, '\(\s*1\s*\|\s*([A-Za-z]\w*)\s*\)', ...
                            'tokens', 'once');
            assert(~isempty(grpTok), ...
                'fitcirc_lme:noGroup', ...
                'formula must contain a (1|group) random-intercept term');
            groupVar = grpTok{1};
            assert(ismember(groupVar, tbl.Properties.VariableNames), ...
                'grouping variable "%s" not found', groupVar);

            % Variance-minimizing circular shift, computed on the
            % NaN-stripped response. Applied to a working copy of the
            % table (so design matrix and grouping match), and recorded
            % on the model so predict() can invert it.
            theta_shift = 0;
            if ~isempty(opt.ThetaShift)
                theta_shift = double(opt.ThetaShift);
            elseif opt.AutoShift
                theta_shift = circ_shift_min_var(tbl.(respName));
            end
            if theta_shift ~= 0
                tbl_fit = tbl;
                tbl_fit.(respName) = wrapToPi(tbl.(respName) - theta_shift);
            else
                tbl_fit = tbl;
            end

            tmp = fitlme(tbl_fit, formula);
            X        = designMatrix(tmp, 'Fixed');
            colNames = tmp.CoefficientNames(:);
            y        = tbl_fit.(respName);

            grp = tbl.(groupVar);
            [g_idx, g_levels] = grp2idx(grp);
            n      = length(y);
            n_subj = length(g_levels);
            n_par  = size(X, 2);

            % --- Initialize ---
            kappa     = opt.InitKappa;
            kappa_phi = opt.InitKappaPhi;
            % Warm-start beta from caller-supplied InitBeta if given. Names
            % are matched against the new column names; absent columns
            % start at 0. IRLS then refines from this seed instead of
            % starting cold at all-zeros.
            beta_init = zeros(n_par, 1);
            if ~isempty(opt.InitBeta) && ~isempty(opt.InitBetaNames)
                init_names = cellstr(opt.InitBetaNames);
                init_b     = opt.InitBeta(:);
                if numel(init_names) == numel(init_b)
                    for k = 1:n_par
                        m_idx = find(strcmp(init_names, colNames{k}), 1);
                        if ~isempty(m_idx)
                            beta_init(k) = init_b(m_idx);
                        end
                    end
                end
            end
            beta = local_irls_circ_offset(X, y, zeros(n,1), beta_init, [], 50);

            mu_post = zeros(n_subj, 1);
            K_post  = zeros(n_subj, 1);
            rho     = zeros(n_subj, 1);

            ll_trace = nan(opt.MaxIter, 1);
            it = 0;
            n_bt = 20;   % max step-halvings for the ascent guard

            % Penalized marginal log-likelihood at the starting parameters.
            % The EM is kept MONOTONE in this objective (see the ascent
            % guard below), so it can never wander downhill into a bad
            % optimum or report a likelihood below a smaller nested model.
            [ll_data, ll_prev] = local_marginal_ll(X, y, g_idx, n_subj, n, ...
                beta, kappa, kappa_phi, opt.KappaPhiPrior, opt.KappaPhiPriorScale);
            ll_pen = ll_prev;

            for it = 1:opt.MaxIter
                % --- E-STEP (exact: posterior is vonMises) ---
                eta = X * beta;
                for i = 1:n_subj
                    idx = (g_idx == i);
                    a = y(idx) - eta(idx);
                    Sg = sum(sin(a));
                    Cg = sum(cos(a));
                    Cx = kappa*Cg + kappa_phi;
                    Sx = kappa*Sg;
                    K_post(i)  = sqrt(Cx*Cx + Sx*Sx);
                    mu_post(i) = atan2(Sx, Cx);
                    rho(i)     = local_A(K_post(i));
                end
                phi_hat = mu_post;
                offset  = mu_post(g_idx);

                % --- M-STEP (propose a candidate parameter update) ---
                % beta: weighted IRLS, weight rho_i per observation
                w = rho(g_idx);
                beta_new = local_irls_circ_offset(X, y, offset, beta, w, 50);

                % kappa: mean resultant length of the residual angles,
                % weighting each observation by rho_i, then converted to a
                % concentration. The rho_i weights account for the offsets
                % being known only approximately.
                resid = wrapToPi(y - X*beta_new - offset);
                Sc = sum(sin(resid) .* w);
                Cc = sum(cos(resid) .* w);
                R_corr = sqrt(Sc*Sc + Cc*Cc) / n;
                kappa_new = local_kappa_from_R(R_corr, n);

                % kappa_phi: penalized 1-D MLE (prior mean fixed at 0).
                % Maximizes n_subj*(R_phi*k - logI0(k)) + logprior(k) with
                % R_phi = mean_i rho_i cos(mu_post_i); the prior keeps the
                % sigma_phi -> 0 (kappa_phi -> Inf) boundary finite. The
                % optional hard ceiling (Inf by default) is applied last.
                R_phi         = mean(rho .* cos(mu_post));
                kappa_phi_new = local_kappaphi_update(R_phi, n_subj, ...
                                opt.KappaPhiPrior, opt.KappaPhiPriorScale);
                kappa_phi_new = min(kappa_phi_new, opt.KappaPhiMax);

                % --- ASCENT GUARD ---
                % The three coordinate updates above are only approximate
                % maximizers, so the full step can OVERSHOOT and lower the
                % marginal likelihood. To keep EM monotone (a generalized
                % EM), backtrack along the segment from the current
                % parameters to the proposed ones, accepting the first step
                % that does not decrease the penalized objective. At t -> 0
                % the step vanishes (no change), so a non-improving
                % direction simply stops the algorithm at the current --
                % best so far -- point rather than diverging.
                accepted = false;
                t = 1;
                for bt = 1:n_bt
                    b_t  = beta      + t*(beta_new      - beta);
                    k_t  = kappa     + t*(kappa_new     - kappa);
                    kp_t = kappa_phi + t*(kappa_phi_new - kappa_phi);
                    [lld_t, llp_t] = local_marginal_ll(X, y, g_idx, n_subj, n, ...
                        b_t, k_t, kp_t, opt.KappaPhiPrior, opt.KappaPhiPriorScale);
                    if llp_t >= ll_prev - 1e-9
                        beta = b_t; kappa = k_t; kappa_phi = kp_t;
                        ll_data = lld_t; ll_pen = llp_t;
                        accepted = true;
                        break
                    end
                    t = t / 2;
                end

                if ~accepted
                    % No step improves the objective: already at a (local)
                    % optimum. Keep the current parameters and stop.
                    ll_trace(it) = ll_data;
                    break
                end

                ll_trace(it) = ll_data;
                if opt.Verbose
                    fprintf(['  iter %3d  ll=%.4f  (pen=%.4f)  ' ...
                             'kappa=%.3f  kappa_phi=%.3f  step=%.3g\n'], ...
                            it, ll_data, ll_pen, kappa, kappa_phi, t);
                end
                if abs(ll_pen - ll_prev) / max(abs(ll_prev), 1) < opt.Tol
                    ll_prev = ll_pen;
                    break
                end
                ll_prev = ll_pen;
            end
            lp_kphi = local_kphi_logprior(kappa_phi, ...
                            opt.KappaPhiPrior, opt.KappaPhiPriorScale);

            % --- Final E-step (refresh posterior at converged params) ---
            eta = X * beta;
            for i = 1:n_subj
                idx = (g_idx == i);
                a = y(idx) - eta(idx);
                Sg = sum(sin(a));
                Cg = sum(cos(a));
                Cx = kappa*Cg + kappa_phi;
                Sx = kappa*Sg;
                K_post(i)  = sqrt(Cx*Cx + Sx*Sx);
                mu_post(i) = atan2(Sx, Cx);
                rho(i)     = local_A(K_post(i));
            end
            phi_hat = mu_post;
            offset  = phi_hat(g_idx);

            % --- Cluster-robust SEs on beta from the MARGINAL score ---
            % An earlier estimator built the sandwich "bread" from the
            % CONDITIONAL Fisher information kappa*A(kappa)*sum_i rho_i X'X,
            % i.e. the information about beta GIVEN each subject's offset.
            % Conditioning on the plugged-in random effects badly overstates
            % how tightly the data pin beta: as beta moves, every subject's
            % offset moves with it and absorbs part of the change, which the
            % conditional information ignores. That made Var(beta) far too
            % small -- anti-conservative p-values, the more so the stronger
            % the subject random intercept (it vanished as kappa_phi grew).
            %
            % Fix: the Huber-White sandwich for the MARGINAL MLE, each
            % subject one cluster. Both halves use the marginal
            % log-likelihood (random intercept integrated out exactly, the
            % same closed form the EM maximizes), over theta = (beta, kappa,
            % kappa_phi):
            %   meat   B = sum_i s_i s_i',  s_i = d/dtheta log p(y_i | theta)
            %   bread  A = -d^2/dtheta^2 sum_i log p(y_i)  + prior curvature
            % The beta block of A^{-1} B A^{-1} is the variance with kappa
            % and kappa_phi co-estimated. (Aside: the marginal beta-score
            % s_i(beta) = kappa*rho_i*sum_j X_ij*sin(y_ij - X_ij*beta -
            % mu_post_i) is exactly the old "meat" term -- the meat was
            % already correct; only the bread was wrong.) Derivatives are
            % central finite differences of the closed-form LL, O(q^2) cheap
            % evaluations.
            theta = [beta; kappa; kappa_phi];
            q     = numel(theta);
            f_subj = @(th) local_marginal_ll_subj(X, y, g_idx, n_subj, ...
                             th(1:n_par), max(th(n_par+1), 1e-8), max(th(n_par+2), 1e-8));
            hstep = 1e-5 * max(abs(theta), 1);
            % Per-subject score matrix S (n_subj x q): S(i,j) = d ll_i/d th_j.
            S = zeros(n_subj, q);
            for j = 1:q
                tp = theta; tp(j) = tp(j) + hstep(j);
                tm = theta; tm(j) = tm(j) - hstep(j);
                S(:, j) = (f_subj(tp) - f_subj(tm)) / (2 * hstep(j));
            end
            B_meat = S' * S;
            % Bread: observed information of the PENALIZED total LL. The
            % kappa_phi prior curvature regularizes the sigma_phi -> 0
            % boundary so A stays well-conditioned; the prior is a single
            % non-cluster term, so it enters the bread only, not the meat.
            f_tot_pen = @(th) sum(f_subj(th)) + ...
                local_kphi_logprior(max(th(n_par+2), 1e-8), ...
                                    opt.KappaPhiPrior, opt.KappaPhiPriorScale);
            A_info = -local_num_hessian(f_tot_pen, theta, hstep);
            A_inv  = pinv(A_info);
            V_full = A_inv * B_meat * A_inv;
            V_rob  = V_full(1:n_par, 1:n_par);
            % Cameron-Miller finite-sample cluster correction, symmetrized.
            corr_factor = (n_subj / (n_subj - 1)) * ((n - 1) / (n - n_par));
            V_rob = corr_factor * (V_rob + V_rob') / 2;

            se_b   = sqrt(max(diag(V_rob), 0));
            t_b    = beta ./ se_b;
            df_t   = n_subj - 1;
            p_b    = 2 * (1 - tcdf(abs(t_b), df_t));

            obj.Coefficients = table(colNames, beta, se_b, t_b, p_b, ...
                'VariableNames', {'Name','Estimate','SE','tStat','pValue'});

            obj.Formula            = formula;
            obj.ResponseName       = respName;
            obj.GroupingVar        = groupVar;
            obj.Beta               = beta;
            obj.Kappa              = kappa;
            obj.KappaPhi           = kappa_phi;
            % Equivalent circular SD: A(kappa_phi) = exp(-sigma^2/2)
            % (the wrapped-Normal -> vM matching identity).
            if kappa_phi <= 0
                obj.SigmaPhi = inf;
            else
                A_kp = local_A(kappa_phi);
                obj.SigmaPhi = sqrt(max(0, -2*log(max(A_kp, eps))));
            end
            obj.PhiHat             = phi_hat;
            obj.PhiRho             = rho;
            obj.PhiKappaPost       = K_post;
            obj.LogLikelihood      = ll_data;
            obj.LogPrior           = lp_kphi;
            obj.NumObservations    = n;
            obj.NumSubjects        = n_subj;
            obj.NumCoefficients    = n_par;
            obj.ConvergedIn        = it;
            obj.ConvergenceHistory = ll_trace(1:it);
            obj.DesignNames        = colNames;
            obj.SubjectIDs         = g_levels;
            obj.X_design           = X;
            obj.TrainingData       = tbl_fit;
            obj.ThetaShift         = theta_shift;

            % Bake the shift back into the (Intercept) coefficient so the
            % linear predictor X*Beta is on the original (unshifted) angle
            % scale. Variance/covariance is invariant under translation, so
            % cov_b, joint Wald tests, and CI half-widths are unchanged.
            % After this, predict() and external CI code do NOT need to
            % know about ThetaShift; it is recorded only as metadata.
            if theta_shift ~= 0 && ~isempty(obj.Beta)
                int_idx = find(strcmp(colNames, '(Intercept)'), 1);
                if ~isempty(int_idx)
                    obj.Beta(int_idx) = wrapToPi(obj.Beta(int_idx) + theta_shift);
                    obj.Coefficients.Estimate(int_idx) = obj.Beta(int_idx);
                    % The intercept's t/p were already not interpretable
                    % on a circular scale; leave them as-is rather than
                    % falsely advertising a meaningful test.
                end
            end

            % Free parameters: p (beta) + kappa + kappa_phi
            k_eff = n_par + 2;
            obj.AIC = -2*ll_data + 2*k_eff;
            obj.BIC = -2*ll_data + log(n)*k_eff;

            obj.cov_b = V_rob;
            obj.DFE   = df_t;

            % --- Build ContrastIndex by parsing colNames for predictor blocks. ---
            % Find the polynomial base predictor by scanning ALL columns
            % (not just colNames{2}, which is fragile to fitlme's
            % coefficient reordering when a categorical and a continuous
            % predictor are both in the formula). We accept TWO
            % naming conventions:
            %   * orthogonal polynomial columns: `<base>_op<k>` with k
            %     in 1..K (used by circular_regression when the basis was
            %     pre-orthogonalized via ortho_poly_basis); these are
            %     ALL polynomial main effects of `<base>` -- no
            %     auto-expansion needed.
            %   * raw-power: `<base>`, `<base>^k`; auto-expanded
            %     by fitlme from a `^k` formula term.
            % The first convention wins if any column matches it.
            ci = struct();
            if n_par >= 2
                op_tok = regexp(char(colNames{2}), '^(.+)_op\d+$', 'tokens', 'once');
                base = '';
                is_op_basis = false;
                % Scan all columns for an orthogonal-polynomial pattern.
                for kk_scan = 2:n_par
                    tok = regexp(char(colNames{kk_scan}), '^([A-Za-z_]\w*)_op\d+$', 'tokens', 'once');
                    if ~isempty(tok)
                        base = tok{1};
                        is_op_basis = true;
                        break;
                    end
                end
                if ~is_op_basis
                    base_tok = regexp(char(colNames{2}), '^([A-Za-z_]+?)\d*$', 'tokens', 'once');
                    if isempty(base_tok)
                        base_tok = regexp(char(colNames{2}), '^([A-Za-z_]\w*)', 'tokens', 'once');
                    end
                    if ~isempty(base_tok), base = base_tok{1}; end
                end
                if ~isempty(base)
                    if is_op_basis
                        is_poly_term = @(nm) ~isempty(regexp(nm, ...
                            ['^' regexptranslate('escape',base) '_op\d+$'], 'once'));
                    else
                        is_poly_term = @(nm) strcmp(nm, base) || ...
                            ~isempty(regexp(nm, ['^' regexptranslate('escape',base) '\^?\d+$'], 'once'));
                    end
                    main_idx = [];
                    for kk = 2:n_par
                        nm = char(colNames{kk});
                        if is_poly_term(nm)
                            main_idx(end+1,1) = kk; %#ok<AGROW>
                        end
                    end
                    ci.x_main = main_idx;
                    intx_groups = containers.Map('KeyType','char','ValueType','any');
                    for kk = 2:n_par
                        nm   = char(colNames{kk});
                        toks = strsplit(nm, ':');
                        if numel(toks) >= 2
                            is_poly = cellfun(is_poly_term, toks);
                            if any(is_poly)
                                cat_tokens = toks(~is_poly);
                                key = strjoin(cat_tokens, '_');
                                if ~isKey(intx_groups, key)
                                    intx_groups(key) = [];
                                end
                                v = intx_groups(key);
                                v(end+1,1) = kk; %#ok<AGROW>
                                intx_groups(key) = v;
                            end
                        end
                    end
                    keys_ = intx_groups.keys;
                    age_idx = ci.x_main;
                    for kk = 1:numel(keys_)
                        fld = matlab.lang.makeValidName(['x_x_' keys_{kk}]);
                        ci.(fld) = intx_groups(keys_{kk});
                        age_idx = [age_idx; ci.(fld)]; %#ok<AGROW>
                    end
                    % Omnibus age block: the polynomial main effect plus every
                    % age-interaction term -> a single "any age effect" test.
                    ci.x_age = unique(age_idx);
                end
            end
            obj.ContrastIndex = ci;

            % Angle-scale R^2 via 1 - SSE_circ / SST_circ
            mu_y    = atan2(mean(sin(y)), mean(cos(y)));
            sse     = sum(1 - cos(wrapToPi(y - X*beta - phi_hat(g_idx))));
            sst     = sum(1 - cos(wrapToPi(y - mu_y)));
            R2      = 1 - sse/max(sst, eps);
            R2adj   = 1 - (1 - R2) * (n - 1) / max(n - n_par, 1);
            obj.Rsquared.Ordinary = R2;
            obj.Rsquared.Adjusted = R2adj;
        end

        function out = coefTest(obj, R)
            % Wald joint test of H0: R*beta = 0, using cluster-robust cov.
            if ischar(R) || (isstring(R) && isscalar(R))
                R = obj.ContrastIndex.(char(R));
            end
            if isvector(R) && ~all(ismember(R(:), [0,1]))
                idx = R(:);
                R   = zeros(numel(idx), obj.NumCoefficients);
                for k = 1:numel(idx), R(k, idx(k)) = 1; end
            end
            Rb  = R * obj.Beta;
            V   = R * obj.cov_b * R';
            chi = Rb' * (V \ Rb);
            df  = size(R, 1);
            % Cluster-robust small-sample reference: the Wald quadratic form
            % scaled by its rank, W/df1, is referenced to F(df1, m-1) with
            % m = number of subject clusters (obj.DFE = m-1), NOT to a
            % chi-square. For df1 = 1 this is identically the two-sided
            % t(m-1) used by the per-coefficient table, so the omnibus and
            % single-coefficient tests agree; for df1 > 1 it supplies the
            % finite-cluster correction a chi-square reference omits (the
            % chi-square form is anti-conservative when m is small, the
            % regime circular GLMMs are usually fit in). 'upper' evaluates
            % the tail directly so extreme statistics do not underflow to a
            % literal p = 0.
            df2   = obj.DFE;                 % = n_subj - 1 (cluster df)
            Fstat = chi / df;
            out = struct('Fstat',  Fstat, ...
                         'pValue', fcdf(Fstat, df, df2, 'upper'), ...
                         'df1',    df, ...
                         'df2',    df2);
        end


        function ang = predict(obj, nd, varargin)
            % PREDICT  Predicted angle in [-pi, pi].
            pp = inputParser;
            pp.addParameter('Conditional', false);
            pp.parse(varargin{:});
            cond = pp.Results.Conditional;

            if nargin < 2 || isempty(nd)
                X_new = obj.X_design;
                if cond
                    g = obj.SubjectIDs;
                    offset = obj.PhiHat(grp2idx(g));
                else
                    offset = zeros(size(X_new,1), 1);
                end
            else
                base = obj.TrainingData;
                nd2  = nd;
                missing_cols = setdiff(base.Properties.VariableNames, ...
                                       nd2.Properties.VariableNames);
                for c = missing_cols
                    cn = c{1};
                    if iscategorical(base.(cn))
                        nd2.(cn) = repmat(base.(cn)(1), height(nd2), 1);
                    elseif iscell(base.(cn))
                        nd2.(cn) = repmat(base.(cn)(1), height(nd2), 1);
                    else
                        nd2.(cn) = zeros(height(nd2), 1, 'like', base.(cn));
                    end
                end
                nd2 = nd2(:, base.Properties.VariableNames);
                combined = [base; nd2];
                tmp = fitlme(combined, obj.Formula);
                X_all = designMatrix(tmp, 'Fixed');
                X_new = X_all(height(base)+1 : end, :);
                offset = zeros(size(X_new,1), 1);
                if cond
                    grp = nd.(obj.GroupingVar);
                    [gi, names] = grp2idx(grp);
                    [~, locb] = ismember(names, obj.SubjectIDs);
                    assert(all(locb > 0), ...
                        'fitcirc_lme:unknownSubject', ...
                        'newdata contains subjects not seen in the fitted model');
                    offset = obj.PhiHat(locb(gi));
                end
            end
            ang = wrapToPi(X_new * obj.Beta + offset);
        end


        function disp(obj)
            fprintf('  Circular linear mixed-effects model (von Mises GLMM, exact EM)\n');
            fprintf('    Formula:        %s\n', obj.Formula);
            fprintf('    Response:       %s (radians)\n', obj.ResponseName);
            fprintf('    Grouping:       %s (%d levels)\n', ...
                obj.GroupingVar, obj.NumSubjects);
            fprintf('    Observations:   %d   Coefficients: %d\n', ...
                obj.NumObservations, obj.NumCoefficients);
            fprintf('    Kappa:          %.3f   KappaPhi: %.3f   (SigmaPhi=%.3f)\n', ...
                obj.Kappa, obj.KappaPhi, obj.SigmaPhi);
            fprintf('    LogLik (exact): %.2f   AIC: %.2f   BIC: %.2f\n', ...
                obj.LogLikelihood, obj.AIC, obj.BIC);
            fprintf('    Converged in %d EM iterations.\n\n', obj.ConvergedIn);
            fprintf('  Fixed-effect coefficients (cluster-robust SEs):\n');
            disp(obj.Coefficients);
        end
    end
end


% --------------------------------------------------------------------
% IRLS for von Mises identity-link regression with an offset and
% optional per-observation weights.  Solves the weighted score equation
%   sum_ij w_ij * X_ij' * sin(y_ij - X_ij*beta - offset_ij) = 0
% by iterating
%   z      = X*beta + sin(y - X*beta - offset)
%   beta+  = (X' W X)^{-1} X' W z
% (cos-weighted Hessian approximated by W; OLS when w = []).
% --------------------------------------------------------------------
function beta = local_irls_circ_offset(X, y, offset, beta0, w, n_iter)
    beta = beta0;
    use_w = ~isempty(w);
    for it = 1:n_iter
        eta = X * beta + offset;
        r   = y - eta;
        z   = X * beta + sin(r);
        if use_w
            XtW = X' .* w(:)';        % p x n
            beta_new = (XtW * X) \ (XtW * z);
        else
            beta_new = (X' * X) \ (X' * z);
        end
        if max(abs(beta_new - beta)) < 1e-9, return, end
        beta = beta_new;
    end
end


% --------------------------------------------------------------------
% A(k) = I_1(k)/I_0(k), the mean resultant length of vonMises(0, k).
% Numerically stable for small and large k.
% --------------------------------------------------------------------
function out = local_A(k)
    if k <= 0
        out = 0;
    elseif k < 1e-6
        out = k/2;                       % Taylor at zero
    elseif k > 700
        out = 1 - 1/(2*k) - 1/(8*k^2);   % asymptotic for large k
    else
        % Scaled Bessels: besseli(0,k,1) = I_0(k)*exp(-k); the exp(-k)
        % factors cancel in the ratio, so this is safe up to k ~ 1e6.
        out = besseli(1, k, 1) / besseli(0, k, 1);
    end
end


% --------------------------------------------------------------------
% log( I_0(k) ), numerically stable for large k.
% --------------------------------------------------------------------
function out = local_logI0(k)
    if k <= 0
        out = 0;
    else
        out = log(besseli(0, k, 1)) + k;
    end
end


% --------------------------------------------------------------------
% Consistent marginal log-likelihood at a single parameter triple
% (beta, kappa, kappa_phi). Every piece -- each subject's posterior
% concentration K_post, the response term, and the prior term -- is
% evaluated at the SAME parameters, so the returned value is the true
% marginal LL at that point. The EM uses this both to test convergence
% and, in the ascent guard, to score candidate steps. Returns the
% unpenalized LL and the penalized objective (LL + log kappa_phi prior).
% --------------------------------------------------------------------
function [ll_data, ll_pen] = local_marginal_ll(X, y, g_idx, n_subj, n, ...
                                  beta, kappa, kappa_phi, prior, scale)
    eta = X * beta;
    acc = 0;
    for i = 1:n_subj
        idx = (g_idx == i);
        a   = y(idx) - eta(idx);
        Cg  = sum(cos(a));  Sg = sum(sin(a));
        Cx  = kappa*Cg + kappa_phi;  Sx = kappa*Sg;
        acc = acc + local_logI0(sqrt(Cx*Cx + Sx*Sx));
    end
    ll_data = acc - n*(log(2*pi) + local_logI0(kappa)) - n_subj*local_logI0(kappa_phi);
    ll_pen  = ll_data + local_kphi_logprior(kappa_phi, prior, scale);
end


% --------------------------------------------------------------------
% Log of the weakly-informative prior placed ON kappa_phi to regularize
% the sigma_phi -> 0 (kappa_phi -> Inf) boundary. Normalizing constants
% (which do not depend on k and are identical across nested models with
% the same scale) are dropped; only the k-dependent part is kept, so the
% argmax and any nested-model differences are unaffected.
%   'halfcauchy' : log p ~ -log(1 + (k/s)^2)   (heavy tail; gentle)
%   'halfnormal' : log p ~ -k^2/(2 s^2)        (light tail; firmer)
%   'none'       : 0
% --------------------------------------------------------------------
function lp = local_kphi_logprior(k, prior, scale)
    s = scale;
    switch lower(char(prior))
        case 'none'
            lp = 0;
        case 'halfcauchy'
            lp = -log1p((k./s).^2);
        case 'halfnormal'
            lp = -(k.^2) ./ (2*s^2);
        otherwise
            error('fitcirc_lme:badPrior', ...
                  'KappaPhiPrior must be ''halfcauchy'', ''halfnormal'', or ''none''');
    end
end


% --------------------------------------------------------------------
% One M-step update for kappa_phi: maximize the exact 1-D objective
%   g(k) = n_subj*(R*k - logI0(k)) + logprior(k),   k >= 0,
% where R = mean_i rho_i cos(mu_post_i) summarizes how tightly the
% estimated subject offsets bunch around 0 (R can be negative). The prior
% keeps the maximizer finite, and the search is capped at k_hi so I0(k)
% never overflows. Because the half-Cauchy term can make g non-concave, a
% coarse grid first brackets the peak and fminbnd then refines it. When
% R <= 0 the maximizer is k = 0.
% --------------------------------------------------------------------
function kphi = local_kappaphi_update(R, n_subj, prior, scale)
    k_hi  = 1e3;                       % I0 stays finite well past here
    % neg_g is always evaluated at a SCALAR k (grid sweep + fminbnd).
    neg_g = @(k) -(n_subj*(R*k - local_logI0(k)) ...
                   + local_kphi_logprior(k, prior, scale));
    grid  = [0, logspace(-2, log10(k_hi), 60)];
    gv    = -arrayfun(neg_g, grid);
    [gbest, gi] = max(gv);
    lo = grid(max(gi-1, 1));
    hi = grid(min(gi+1, numel(grid)));
    kphi = fminbnd(neg_g, lo, hi, optimset('TolX', 1e-6));
    if -neg_g(kphi) < gbest        % fall back to best grid point
        kphi = grid(gi);
    end
    if ~isfinite(kphi) || kphi < 0
        kphi = 0;
    end
end


% --------------------------------------------------------------------
% Turn a target mean resultant length R back into a concentration k, i.e.
% solve A(k) = R, using the standard piecewise approximation to the
% inverse of A(k) = I_1(k)/I_0(k). Used, e.g., to convert an InitSigma
% option into InitKappaPhi.
% --------------------------------------------------------------------
function k = local_invA(R)
    if R <= 0
        k = 0;
    elseif R >= 1
        k = 1e6;
    elseif R < 0.53
        k = 2*R + R^3 + 5*R^5/6;
    elseif R < 0.85
        k = -0.4 + 1.39*R + 0.43/(1-R);
    else
        k = 1/(R^3 - 4*R^2 + 3*R);
    end
end


% --------------------------------------------------------------------
% Concentration kappa from a mean resultant length R, using the standard
% piecewise approximation that inverts A(k) = I_1(k)/I_0(k). For small
% samples (N < 15) the maximum-likelihood kappa is biased upward, so a
% standard small-sample correction is applied. Matches the formula used
% in fitlme_circ.m so the two stay comparable.
% --------------------------------------------------------------------
function kappa = local_kappa_from_R(R, N)
    if R <= 0
        kappa = 0;
        return
    end
    if R < 0.53
        kappa = 2*R + R^3 + 5*R^5/6;
    elseif R < 0.85
        kappa = -0.4 + 1.39*R + 0.43/(1-R);
    else
        kappa = 1/(R^3 - 4*R^2 + 3*R);
    end
    if N < 15 && N > 1
        if kappa < 2
            kappa = max(kappa - 2/(N*kappa), 0);
        else
            kappa = (N-1)^3 * kappa / (N^3 + N);
        end
    end
end


% --------------------------------------------------------------------
% Per-subject UNPENALIZED marginal log-likelihood (random intercept
% integrated out exactly), returned as a column vector. sum(llv) equals
% the scalar ll_data from local_marginal_ll at the same parameters, so the
% two stay consistent. Used to build the marginal-MLE sandwich SEs.
% --------------------------------------------------------------------
function llv = local_marginal_ll_subj(X, y, g_idx, n_subj, beta, kappa, kappa_phi)
    eta = X * beta;
    llv = zeros(n_subj, 1);
    for i = 1:n_subj
        idx = (g_idx == i);
        a   = y(idx) - eta(idx);
        Cg  = sum(cos(a));  Sg = sum(sin(a));  ni = sum(idx);
        Cx  = kappa*Cg + kappa_phi;  Sx = kappa*Sg;
        llv(i) = local_logI0(sqrt(Cx*Cx + Sx*Sx)) ...
                 - ni*(log(2*pi) + local_logI0(kappa)) ...
                 - local_logI0(kappa_phi);
    end
end


% --------------------------------------------------------------------
% Symmetric numerical Hessian of a scalar function f at x, via central
% differences with per-coordinate step h. Used for the marginal observed
% information that forms the sandwich bread.
% --------------------------------------------------------------------
function H = local_num_hessian(f, x, h)
    q  = numel(x);
    H  = zeros(q);
    f0 = f(x);
    for i = 1:q
        for j = i:q
            if i == j
                xp = x; xp(i) = xp(i) + h(i);
                xm = x; xm(i) = xm(i) - h(i);
                H(i,i) = (f(xp) - 2*f0 + f(xm)) / (h(i)^2);
            else
                xpp = x; xpp(i) = xpp(i)+h(i); xpp(j) = xpp(j)+h(j);
                xpm = x; xpm(i) = xpm(i)+h(i); xpm(j) = xpm(j)-h(j);
                xmp = x; xmp(i) = xmp(i)-h(i); xmp(j) = xmp(j)+h(j);
                xmm = x; xmm(i) = xmm(i)-h(i); xmm(j) = xmm(j)-h(j);
                val = (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4*h(i)*h(j));
                H(i,j) = val;  H(j,i) = val;
            end
        end
    end
end
