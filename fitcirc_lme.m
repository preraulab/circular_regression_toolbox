classdef fitcirc_lme
%FITCIRC_LME  Mixed-effects regression for circular data, von Mises GLMM.
%
% True circular GLMM with a circular prior on the random subject phase:
%
%   y_ij | beta, phi_i, kappa  ~  vonMises( X_ij*beta + phi_i,  kappa )
%   phi_i                      ~  vonMises( 0,  kappa_phi )
%
% Both the response noise and the random subject phase are von Mises.
% This makes the conditional posterior of phi_i CLOSED FORM (also von
% Mises) rather than requiring a Laplace approximation, and bounds the
% random-effect concentration on the natural circular scale instead of
% letting an unwrapped-Normal prior run sigma_phi to infinity.
%
% --------------------------------------------------------------------
% MATH AT EACH STAGE
% --------------------------------------------------------------------
% E-step (per subject i, fixed beta, kappa, kappa_phi):
%   alpha_ij = y_ij - X_ij*beta            (residual of fixed effects)
%   S_i = sum_j sin(alpha_ij)
%   C_i = sum_j cos(alpha_ij)
%
% Combining likelihood ~ vonMises( mu_i = atan2(S_i, C_i), kappa*R_i )
% with prior vonMises(0, kappa_phi) gives an exact von-Mises posterior
% phi_i | y_i ~ vonMises( mu_post_i, K_post_i ) where:
%   K_post_i * cos(mu_post_i) = kappa * C_i + kappa_phi
%   K_post_i * sin(mu_post_i) = kappa * S_i
%   K_post_i  = sqrt( (kappa C_i + kappa_phi)^2 + (kappa S_i)^2 )
%   mu_post_i = atan2( kappa S_i,  kappa C_i + kappa_phi )
%
% Posterior moments:
%   phi_i_hat   := mu_post_i               (mean direction)
%   rho_i       := A(K_post_i) = I_1/I_0   (mean resultant length)
%   E[cos(phi_i - mu_post_i)] = rho_i      (used by the M-step)
%   E[exp(i*phi_i)]           = rho_i * exp(i*mu_post_i)
%
% No Newton iteration, no Laplace approximation, no unwrapped phi.
%
% M-step:
%   1) beta:      weighted IRLS on the circular score with offset
%        beta solves   sum_ij rho_i * X_ij' * sin(y_ij - X_ij*beta - mu_post_i) = 0
%      i.e. each observation gets weight rho_i (its subject's posterior
%      mean resultant length).  When the posterior is very concentrated
%      (rho_i ~ 1), the obs counts fully; when the posterior is diffuse,
%      it down-weights itself.
%
%   2) kappa:     Banerjee MLE on residuals deflated by rho_i.
%      For phi_i ~ vonMises(mu_post_i, K_i), the characteristic function
%      gives  E[cos(epsilon_ij)] = cos(e_ij) * rho_i  where
%      epsilon_ij = e_ij - (phi_i - mu_post_i) is the true residual.
%      So multiply per-obs sin/cos contributions by rho_i before
%      computing the resultant length.
%
%   3) kappa_phi: constrained MLE for the prior concentration.
%      The prior mean direction is fixed at 0, so the score equation is
%        A(kappa_phi) = max(0, (1/n_subj) * sum_i rho_i * cos(mu_post_i))
%      i.e. the projection of the posterior phase cluster onto the prior
%      mean direction.  When that average is non-positive (subjects'
%      posterior phases scatter away from 0), kappa_phi = 0 and the
%      prior becomes uniform on the circle.
%
% Marginal log-likelihood (EXACT, no Laplace correction):
%   log p(y_i) = log I_0(K_post_i)
%              - n_i * log( 2*pi * I_0(kappa) )
%              -      log(        I_0(kappa_phi) )
%   (the integral over phi_i is closed form, not a Laplace approximation)
%
% Convergence: monitor the marginal log-likelihood; stop when relative
% change is below tol or after MaxIter EM steps.
%
% --------------------------------------------------------------------
% USAGE
% --------------------------------------------------------------------
%   mdl = fitcirc_lme(tbl, 'phase ~ 1 + Age + (1|Subj_ID)')
%   mdl = fitcirc_lme(tbl, formula, Name, Value, ...)
%
% NAME-VALUE OPTIONS
%   'MaxIter'      (default 100)   max EM iterations
%   'Tol'          (default 1e-5)  relative LL tolerance for convergence
%   'Verbose'      (default false) print per-iteration progress
%   'InitKappa'    (default 4)     starting kappa
%   'InitKappaPhi' (default 4)     starting kappa_phi (prior concentration)
%   'InitSigma'    (deprecated)    if given, converted to InitKappaPhi via
%                                  A(kappa_phi) = exp(-sigma^2/2)
%
% OUTPUT FIELDS
%   Formula, ResponseName, GroupingVar
%   Coefficients          - table: Name, Estimate, SE, tStat, pValue
%   Beta                  - p-vector of fixed effects
%   Kappa                 - scalar concentration of the response noise
%   KappaPhi              - scalar concentration of the subject-phase prior
%   SigmaPhi              - equivalent circular SD,
%                           sqrt( -2 * log( A(KappaPhi) ) )  (for back-
%                           compat with code that read the old field name)
%   PhiHat                - n_subj-by-1 posterior mean directions
%   PhiRho                - n_subj-by-1 posterior mean resultant lengths
%   PhiKappaPost          - n_subj-by-1 posterior concentrations
%   LogLikelihood         - exact marginal LL at the converged params
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
% Standard errors on beta come from the cluster-robust sandwich
% (Liang & Zeger 1986) with subject as the cluster, rescaled by the
% small-sample factor m/(m-1) * (n-1)/(n-p).  Inference uses the
% t-distribution with (n_subjects - 1) d.o.f.  As with any plug-in EM
% sandwich, the n_j=2 design can under-cover; the cluster-robust vM
% MLE without random-effect machinery is recommended for primary
% inference.
%
% LIMITATIONS
%   - Single (1|group) random-intercept term only.
%   - Cluster-robust SEs need a reasonable number of subjects.

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
        % Default 0 means "fit on the data as given" (legacy behavior).
        ThetaShift = 0
    end

    methods
        function obj = fitcirc_lme(tbl, formula, varargin)
            % --- Parse name-value options ---
            p = inputParser;
            p.addParameter('MaxIter',  100);
            p.addParameter('Tol',      1e-5);
            p.addParameter('Verbose',  false);
            p.addParameter('InitKappa',     4);
            p.addParameter('InitKappaPhi',  4);
            p.addParameter('InitSigma',    []);  % deprecated; converted if given
            % AutoShift = true: rotate y by the variance-minimizing shift
            % so wrapping doesn't split the response near +/- pi. The shift
            % is recorded in ThetaShift and added back during predict().
            p.addParameter('AutoShift', false);
            p.addParameter('ThetaShift', []);    % override AutoShift with explicit value
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
            beta      = local_irls_circ_offset(X, y, zeros(n,1), zeros(n_par,1), [], 50);

            mu_post = zeros(n_subj, 1);
            K_post  = zeros(n_subj, 1);
            rho     = zeros(n_subj, 1);

            ll_trace = nan(opt.MaxIter, 1);
            ll_prev  = -inf;
            it = 0;

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

                % --- M-STEP ---
                % beta: weighted IRLS, weight rho_i per observation
                w = rho(g_idx);
                beta = local_irls_circ_offset(X, y, offset, beta, w, 50);

                % kappa: Banerjee on residuals deflated by rho_i
                resid = wrapToPi(y - X*beta - offset);

                Sc = sum(sin(resid) .* w);
                Cc = sum(cos(resid) .* w);
                R_corr = sqrt(Sc*Sc + Cc*Cc) / n;
                kappa  = local_kappa_from_R(R_corr, n);

                % kappa_phi: constrained MLE (prior mean fixed at 0).
                % A(kappa_phi) = max(0, mean_i rho_i * cos(mu_post_i))
                R_phi = max(0, mean(rho .* cos(mu_post)));
                kappa_phi = local_kappa_from_R(R_phi, n_subj);

                % Marginal LL (exact)
                ll_data = sum(local_logI0(K_post)) ...
                        - n      * (log(2*pi) + local_logI0(kappa)) ...
                        - n_subj * local_logI0(kappa_phi);

                ll_trace(it) = ll_data;
                if opt.Verbose
                    fprintf('  iter %3d  ll=%.4f  kappa=%.3f  kappa_phi=%.3f\n', ...
                            it, ll_data, kappa, kappa_phi);
                end
                if abs(ll_data - ll_prev) / max(abs(ll_prev), 1) < opt.Tol
                    break
                end
                ll_prev = ll_data;
            end

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

            % --- Cluster-robust (Liang-Zeger) sandwich SEs on beta ---
            r = wrapToPi(y - X*beta - offset);
            W = diag(cos(r));
            A_bread = kappa * X' * W * X;          % bread
            B_meat  = zeros(n_par);
            for i = 1:n_subj
                idx_i = (g_idx == i);
                u_i   = kappa * (X(idx_i,:)' * sin(r(idx_i)));
                B_meat = B_meat + u_i * u_i';
            end

            A_inv  = A_bread \ eye(n_par);
            V_rob  = A_inv * B_meat * A_inv;
            corr_factor = (n_subj / (n_subj - 1)) * ((n - 1) / (n - n_par));
            V_rob  = corr_factor * V_rob;

            se_b   = sqrt(diag(V_rob));
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
            ci = struct();
            if n_par >= 2
                base_tok = regexp(char(colNames{2}), '^([A-Za-z_]+?)\d*$', 'tokens', 'once');
                if isempty(base_tok)
                    base_tok = regexp(char(colNames{2}), '^([A-Za-z_]\w*)', 'tokens', 'once');
                end
                if ~isempty(base_tok)
                    base = base_tok{1};
                    is_poly_term = @(nm) strcmp(nm, base) || ...
                        ~isempty(regexp(nm, ['^' regexptranslate('escape',base) '\^?\d+$'], 'once'));
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
            out = struct('Fstat',  chi/df, ...
                         'pValue', 1 - chi2cdf(chi, df), ...
                         'df1',    df, ...
                         'df2',    obj.DFE);
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
% Invert A(k) = R using the Banerjee large-N approximation.  Used to
% translate a target mean resultant length back to a concentration
% (e.g. for converting an InitSigma option to InitKappaPhi).
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
% Banerjee kappa as a function of mean resultant length R and N.
% Same formula as in fitlme_circ.m (kept for cross-comparability).
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
