classdef fitlme_circ
%FITLME_CIRC  Mixed-effects regression with a circular response.
%
% Drop-in analog of fitlme for outcomes measured in radians.  Internally
% fits two LinearMixedModel objects (one on sin(response), one on
% cos(response)) using the same formula with the response symbol
% substituted, and combines their results into a single object that
% mimics the fitlme API.
%
% USAGE
%   mdl = fitlme_circ(tbl, 'phase ~ 1 + Age + (1|Subj_ID)')
%   mdl = fitlme_circ(tbl, formula, Name, Value, ...)        % NV pairs
%                                                            % forwarded to fitlme
%   mdl.Coefficients                  % combined table
%   yhat = mdl.predict(newdata)       % predicted ANGLE (radians)
%   [p, info] = mdl.coefTest(R)       % Wald joint, Bonferroni'd
%   cmp  = compare(mdl_red, mdl_full) % LRT, Bonferroni'd, fitlme-shaped table
%
% --------------------------------------------------------------------
% MATHEMATICAL BACKGROUND
% --------------------------------------------------------------------
% A circular variable theta in [-pi, pi] is naturally identified with
% the point (cos theta, sin theta) on the unit circle in R^2.  This
% function fits two independent linear mixed models — one with sin(y)
% as the response, one with cos(y) — using the same fixed- and random-
% effects design.  Predicted angles are then reconstructed from the two
% component predictions via atan2(s_hat, c_hat).  This is the
% "projected-Gaussian" approximation: we model the (sin, cos) bivariate
% as Gaussian on R^2 instead of doing a true von Mises fit.  The
% approximation is excellent at moderate-to-high concentration (kappa
% above ~5) and degrades smoothly at low concentration where it biases
% predicted angles toward the polar origin (see CAVEATS).
%
% Why this is useful: every fitlme tool — Wald tests on coefficients
% and contrast matrices, likelihood-ratio comparisons of nested models,
% conditional / marginal prediction, REML/ML model criteria — works on
% sin(y) and cos(y) the same way it does on any continuous outcome.
% We get all of that machinery for circular data at the cost of a small
% projection bias and Bonferroni-conservative joint inference.
%
% Joint inference.  For any test fitlme would do on a single LME, we
% run that test independently on the sin and cos components and report
%
%     p_joint = min(1, 2 * min(p_sin, p_cos))
%
% This is the Bonferroni union over the two component tests.  The
% intuition: the null hypothesis "the predictor has no effect on the
% angle" implies BOTH the sin and cos coefficients are zero, so we can
% reject if EITHER of the per-component tests rejects.  Bonferroni
% controls the family-wise error at twice the smaller component p,
% which is conservative when the two components are correlated (almost
% always positively, since sin and cos of the same latent angle are
% functionally related).  In practice the conservatism is mild — about
% a factor of 1.5 in p compared to a Hotelling-T^2-style joint test —
% and it has the advantage of needing nothing beyond the per-component
% machinery.
%
% --------------------------------------------------------------------
% INPUTS
% --------------------------------------------------------------------
%   tbl     - table with response (in radians) and predictors
%   formula - Wilkinson formula string, e.g. 'y ~ 1 + x + (1|g)'.
%             Response must be a bare variable name in the table (no
%             function transforms like log(y) on the LHS).
%   ...     - any additional name-value pairs forwarded verbatim to
%             fitlme (e.g. 'FitMethod', 'REML' or 'CovariancePattern',
%             'Diagonal').
%
% --------------------------------------------------------------------
% OUTPUT FIELDS
% --------------------------------------------------------------------
%   Formula             - original formula string
%   ResponseName        - response variable name parsed from the formula
%   Sin, Cos            - the two underlying LinearMixedModel objects;
%                         use these to access any fitlme functionality
%                         not exposed at the top level (e.g. anova,
%                         residuals, fixedEffects, randomEffects)
%   Coefficients        - table: Name, EstSin, SESin, pSin, EstCos,
%                         SECos, pCos, pJointBonf
%   CoefficientNames    - cell array of fixed-effect term names
%   NumObservations     - sample size
%   NumCoefficients     - number of fixed-effect coefficients (per
%                         component; identical in sin and cos because
%                         they share the design matrix)
%   LogLikelihood       - sum of two component LLs.  Useful for AIC /
%                         BIC and for chi-square LRT against another
%                         fitlme_circ model with the same response.
%                         Not interpretable as a circular likelihood.
%   DFE                 - residual degrees of freedom (same in each
%                         component)
%   AIC, BIC            - sums of the two component model criteria.
%                         Lower is better.  Comparable across
%                         fitlme_circ models on the same response.
%   ResidualKappa       - von Mises concentration estimated from the
%                         circular residuals e_i = wrap(y_i - yhat_i),
%                         where yhat_i = atan2(s_hat_i, c_hat_i).
%                         Higher = tighter fit on the angle scale.
%   RandomEffectSD_Sin  - per-grouping random-effect SDs for the sin
%                         component LME (cell array; one entry per
%                         grouping variable)
%   RandomEffectSD_Cos  - same for the cos component LME
%
% --------------------------------------------------------------------
% METHODS
% --------------------------------------------------------------------
%   predict(newdata, ...) - predicted angle via atan2(s_hat, c_hat).
%   the                         Forwards Name-Value pairs (e.g. 'Conditional',
%                           false to marginalize over random effects)
%                           to the underlying predict() calls.
%   coefTest(R)           - joint Wald F-test on contrast matrix R,
%                           combined via Bonferroni over the two
%                           components.  Returns p_joint and an info
%                           struct with the per-component p and F.
%   compare(red, full)    - LRT comparing two nested fitlme_circ
%                           models.  Returns a 2-row table shaped like
%                           fitlme's compare output, where pValue(2)
%                           is the Bonferroni-joint p.
%
% --------------------------------------------------------------------
% CAVEATS
% --------------------------------------------------------------------
%   1. Projected-Gaussian likelihood, not von Mises.  Predicted angles
%      are biased toward the polar origin (0 rad) at low to moderate
%      concentration.  The mechanism: at low kappa, both sin and cos
%      are noisy, and atan2 of a noisy point near the unit circle
%      pulls toward zero in expectation.  Empirically the bias is
%      sub-0.05 rad for kappa >= 20 and grows to ~0.2 rad at kappa = 4.
%      See test_kappa_sweep.m for a calibration on synthetic data with
%      known truth.
%
%   2. Component LMEs assume Gaussian errors on sin / cos.  When the
%      response is concentrated near +/- pi (heavy wrapping in the
%      data), this assumption breaks: a small change in latent angle
%      across the +/- pi boundary maps to a large jump in the
%      response, so the residuals look heavy-tailed and inference
%      becomes anti-conservative.  Recenter the response away from
%      +/- pi before fitting if your data straddles the boundary,
%      e.g. by adding a constant offset and wrapping back to [-pi, pi].
%
%   3. There is no single "circular random-intercept variance".  The
%      two component random-effect SDs together describe the per-group
%      offset on the (sin, cos) plane, which is the natural object for
%      a projected-Gaussian model.  If you want a single number on the
%      circular scale, ResidualKappa across subjects gives a rough
%      summary; the cleaner answer is to fit a true mixed-effects von
%      Mises model (e.g. brms with family=von_mises and (1|group)) and
%      read the kappa posterior.

    properties
        Formula              % original formula string
        ResponseName         % response variable name
        Sin                  % LinearMixedModel on sin(response)
        Cos                  % LinearMixedModel on cos(response)
        Coefficients         % combined table with Bonferroni joint p
        CoefficientNames     % fixed-effect term names
        NumObservations      % sample size
        NumCoefficients      % number of fixed-effect coefs (per component)
        LogLikelihood        % LL_sin + LL_cos (sum, not circular LL)
        DFE                  % residual degrees of freedom
        AIC                  % AIC_sin + AIC_cos
        BIC                  % BIC_sin + BIC_cos
        ResidualKappa        % von Mises kappa of circular residuals
        RandomEffectSD_Sin   % per-group RE SDs for sin LME
        RandomEffectSD_Cos   % per-group RE SDs for cos LME
    end

    methods
        function obj = fitlme_circ(tbl, formula, varargin)
            %FITLME_CIRC  Constructor.  Fits two LMEs and combines.
            %
            % Steps:
            %   1. Parse the response variable name from the LHS of ~.
            %   2. Build sin/cos columns of the response in a working
            %      copy of the table (originals are not mutated).
            %   3. Substitute the response symbol with sin_<name> and
            %      cos_<name> in the formula, using a word-boundary
            %      regex so we do not accidentally rename predictors
            %      whose names happen to contain the response substring.
            %   4. Run fitlme on each component formula, forwarding any
            %      additional name-value pairs.
            %   5. Build the combined coefficient table with Bonferroni
            %      joint p-values.
            %   6. Pull random-effect covariance parameters from each
            %      component LME via covarianceParameters().
            %   7. Compute residual kappa: project each (s_hat, c_hat)
            %      back to an angle via atan2, take the wrapped circular
            %      residual y_i - yhat_i, and fit a von Mises kappa to
            %      that residual distribution as a calibration of fit.

            % --- 1. Parse response name ---
            formula  = char(formula);
            tildeIdx = strfind(formula, '~');
            assert(~isempty(tildeIdx), ...
                'fitlme_circ:badFormula', 'formula must contain ~');
            respName = strtrim(formula(1:tildeIdx(1)-1));
            assert(ismember(respName, tbl.Properties.VariableNames), ...
                'fitlme_circ:badResponse', ...
                'response variable "%s" not found in table', respName);

            % --- 2. Build sin/cos response columns ---
            % These live alongside the original response in a working
            % copy.  fitlme will only touch the names referenced in the
            % substituted formulas, so the originals are inert.
            y = tbl.(respName);
            sinName = ['sin_' respName];
            cosName = ['cos_' respName];
            tbl_sc  = tbl;
            tbl_sc.(sinName) = sin(y);
            tbl_sc.(cosName) = cos(y);

            % --- 3. Substitute response symbol in the formula ---
            % Word-boundary lookarounds: (?<![A-Za-z0-9_])X(?![A-Za-z0-9_])
            % matches X only when it is not part of a longer identifier.
            % This is more reliable than \b for typical MATLAB variable
            % names, which can include underscores.
            tok = regexptranslate('escape', respName);
            fml_sin = regexprep(formula, ...
                ['(?<![A-Za-z0-9_])' tok '(?![A-Za-z0-9_])'], sinName);
            fml_cos = regexprep(formula, ...
                ['(?<![A-Za-z0-9_])' tok '(?![A-Za-z0-9_])'], cosName);

            % --- 4. Fit each component LME ---
            % Identical fixed- and random-effects design; only the
            % response differs.  Any extra Name-Value args (FitMethod,
            % CovariancePattern, etc.) are forwarded to both fits so
            % they remain comparable.
            obj.Sin = fitlme(tbl_sc, fml_sin, varargin{:});
            obj.Cos = fitlme(tbl_sc, fml_cos, varargin{:});

            % --- 5. Combined coefficient table with Bonferroni p ---
            % Each component LME tests H0: beta_j = 0 separately on its
            % own response.  The joint null is "beta_j is zero in BOTH
            % sin and cos coefficients", which is rejected if either
            % component test rejects.  Bonferroni-adjusted p across the
            % two components is the standard conservative joint test.
            cs = obj.Sin.Coefficients;
            cc = obj.Cos.Coefficients;
            names = obj.Sin.CoefficientNames(:);
            pJ = min(1, 2 * min(cs.pValue, cc.pValue));
            obj.Coefficients = table( ...
                names, cs.Estimate, cs.SE, cs.pValue, ...
                       cc.Estimate, cc.SE, cc.pValue, pJ, ...
                'VariableNames', {'Name', ...
                                  'EstSin','SESin','pSin', ...
                                  'EstCos','SECos','pCos', ...
                                  'pJointBonf'});

            % --- Top-level scalar fields (mirrors fitlme API where it makes sense) ---
            obj.Formula          = formula;
            obj.ResponseName     = respName;
            obj.CoefficientNames = names;
            obj.NumObservations  = obj.Sin.NumObservations;
            obj.NumCoefficients  = obj.Sin.NumCoefficients;
            % LL is the sum of the two independent component LLs.  This
            % is the right quantity for AIC/BIC across nested
            % fitlme_circ models, but it is NOT a circular log-
            % likelihood (no von Mises kernel).  Don't compare it to
            % the LL from circular_regression_fixed.m or to brms LOO.
            obj.LogLikelihood    = obj.Sin.LogLikelihood + obj.Cos.LogLikelihood;
            obj.DFE              = obj.Sin.DFE;   % identical across components
            obj.AIC              = obj.Sin.ModelCriterion.AIC + obj.Cos.ModelCriterion.AIC;
            obj.BIC              = obj.Sin.ModelCriterion.BIC + obj.Cos.ModelCriterion.BIC;

            % --- 6. Per-grouping random-effect SDs ---
            % covarianceParameters returns a cell array of psi matrices,
            % one per grouping variable (e.g. {Subj_ID, Site}).  Each
            % psi is the covariance of the random effects within that
            % grouping; its diagonal is the per-RE-term variance.
            % Square-root the diagonals to get SDs on the response scale.
            [psi_s, ~, ~] = covarianceParameters(obj.Sin);
            [psi_c, ~, ~] = covarianceParameters(obj.Cos);
            obj.RandomEffectSD_Sin = cellfun(@(M) sqrt(diag(M))', psi_s, ...
                                             'UniformOutput', false);
            obj.RandomEffectSD_Cos = cellfun(@(M) sqrt(diag(M))', psi_c, ...
                                             'UniformOutput', false);

            % --- 7. Concentration of circular residuals ---
            % Reconstruct the predicted angle from the two component
            % predictions, then take the wrapped residual on the
            % circle.  Fitting a von Mises kappa to those residuals
            % gives a calibration of fit quality on the circular scale,
            % comparable to kappa-hat from circular_regression_fixed.m
            % even though our likelihood is Gaussian-on-(sin,cos).
            ang_hat = atan2(predict(obj.Sin), predict(obj.Cos));
            resid   = wrapToPi(y - ang_hat);
            obj.ResidualKappa = local_circ_kappa(resid);
        end


        function ang = predict(obj, nd, varargin)
            %PREDICT  Predicted angle in [-pi, pi] for new data.
            %
            % Predicts sin and cos separately on the new design then
            % reconstructs the angle via atan2(s, c).  Note this is the
            % geometrically correct projection: atan2 reads quadrant
            % from the SIGN of both arguments, so it returns angles in
            % the full circle [-pi, pi], not just [-pi/2, pi/2] like
            % asin would.
            %
            % Any additional Name-Value pairs are forwarded to the
            % underlying predict() calls.  The most useful one for
            % marginal trajectory plots is 'Conditional', false, which
            % marginalizes over (i.e. zeros out) random effects so the
            % predicted angle reflects population-level structure
            % rather than a specific subject's offset.
            %
            % If called without newdata, predicts on the training rows.
            %
            % Limitation: there is no closed-form prediction interval
            % on the angle scale, because atan2 is a nonlinear function
            % of two correlated normal predictions.  If you need
            % uncertainty bands, simulate from the two predictive
            % distributions and apply atan2 to each draw.
            if nargin < 2 || isempty(nd)
                s = predict(obj.Sin);
                c = predict(obj.Cos);
            else
                s = predict(obj.Sin, nd, varargin{:});
                c = predict(obj.Cos, nd, varargin{:});
            end
            ang = atan2(s, c);
        end


        function [pJoint, info] = coefTest(obj, R)
            %COEFTEST  Joint Wald F-test on contrast matrix R.
            %
            % R is an r-by-p matrix of linear contrasts on the fixed-
            % effect coefficients.  Each component LME computes the
            % standard Wald F:
            %
            %     F = (R*beta - 0)' * (R * Sigma * R')^{-1} * (R*beta - 0) / r
            %
            % where Sigma is the estimated coefficient covariance.  The
            % p-value is the upper-tail of F_{r, dfe}.  We run this
            % independently on sin and cos and combine via Bonferroni.
            %
            % Use this to test "all polynomial age terms together" or
            % any other multi-coefficient block effect, exactly as you
            % would call coefTest on a single fitlme model.
            [p_s, F_s] = coefTest(obj.Sin, R);
            [p_c, F_c] = coefTest(obj.Cos, R);
            pJoint = min(1, 2 * min(p_s, p_c));
            if nargout > 1
                info = struct('pSin', p_s, 'pCos', p_c, ...
                              'FSin', F_s, 'FCos', F_c);
            end
        end


        function out = compare(obj_red, obj_full, varargin)
            %COMPARE  Likelihood-ratio test against a nested model.
            %
            % Each component LME runs the standard chi-squared LRT:
            %
            %     LR_chi^2 = -2 * (LL_red - LL_full)  ~  chi^2_{dDF}
            %
            % under the null that the extra parameters in the full
            % model are zero, where dDF is the difference in fixed-
            % effect parameter counts.  We Bonferroni-combine the two
            % component p-values for the joint test "the extra terms
            % matter for the circular response".
            %
            % Returns a 2-row table with the same column schema as
            % fitlme's compare() output (Model, DF, AIC, BIC, LogLik,
            % LRStat, deltaDF, pValue), so existing code paths that
            % read comp.pValue(2) work unchanged.  The reported LRStat
            % is the larger of the two component LR statistics, which
            % is consistent with the Bonferroni joint p (whichever
            % component is more significant determines both the
            % reported chi^2 and the joint p).
            %
            % Extra NV pairs (e.g. 'CheckNesting', true) are forwarded
            % to the underlying compare() calls.
            assert(isa(obj_full, 'fitlme_circ'), ...
                'fitlme_circ:badCompare', ...
                'second argument must be a fitlme_circ object');
            cmp_s = compare(obj_red.Sin, obj_full.Sin, varargin{:});
            cmp_c = compare(obj_red.Cos, obj_full.Cos, varargin{:});
            p_s = cmp_s.pValue(2);
            p_c = cmp_c.pValue(2);
            pJoint = min(1, 2 * min(p_s, p_c));

            % Build a 2-row LRT table that mirrors fitlme's compare
            % shape so callers can use comp.pValue(2) drop-in.
            out = table( ...
                {'reduced'; 'full'}, ...
                [obj_red.NumCoefficients;  obj_full.NumCoefficients], ...
                [obj_red.AIC;              obj_full.AIC], ...
                [obj_red.BIC;              obj_full.BIC], ...
                [obj_red.LogLikelihood;    obj_full.LogLikelihood], ...
                [NaN; max(cmp_s.LRStat(2), cmp_c.LRStat(2))], ...
                [NaN; cmp_s.deltaDF(2)], ...
                [NaN; pJoint], ...
                'VariableNames', ...
                {'Model','DF','AIC','BIC','LogLik','LRStat','deltaDF','pValue'});
            out.Properties.UserData = struct( ...
                'pSin',       p_s, ...
                'pCos',       p_c, ...
                'pJointBonf', pJoint, ...
                'method',     'Bonferroni on sin/cos LRT');
        end


        function disp(obj)
            %DISP  Pretty-print summary, called by MATLAB's display engine.
            fprintf('  Circular linear mixed-effects model (sin/cos pair)\n');
            fprintf('    Formula:       %s\n', obj.Formula);
            fprintf('    Response:      %s (radians)\n', obj.ResponseName);
            fprintf('    Observations:  %d   Coefficients: %d\n', ...
                obj.NumObservations, obj.NumCoefficients);
            fprintf('    LogLik (sin+cos): %.2f   AIC: %.2f   BIC: %.2f\n', ...
                obj.LogLikelihood, obj.AIC, obj.BIC);
            fprintf('    Residual kappa: %.2f\n\n', obj.ResidualKappa);
            fprintf('  Coefficients (Bonferroni joint p):\n');
            disp(obj.Coefficients);
        end
    end
end


% --------------------------------------------------------------------
% LOCAL HELPER: maximum-likelihood von Mises concentration estimator.
% --------------------------------------------------------------------
% Given a sample of angles alpha_1, ..., alpha_N, compute the mean
% resultant length R = |sum(exp(i*alpha))| / N.  The MLE of kappa
% satisfies A_1(kappa) = R, where A_1(k) = I_1(k) / I_0(k) is the ratio
% of modified Bessel functions of orders 1 and 0.
%
% This function uses the three-piece polynomial approximation due to
% Banerjee et al. (2005), accurate to ~0.005 across kappa in [0, 700],
% plus Fisher's small-sample correction (1993) for N < 15.  It mirrors
% the kappa estimator used in circular_regression_fixed.m and the
% R `circular` package, so values are directly comparable.
%
% References
%   Banerjee, A. et al. (2005).  Clustering on the unit hypersphere
%     using von Mises-Fisher distributions.  JMLR 6: 1345-1382.
%   Fisher, N. I. (1993).  Statistical Analysis of Circular Data.
%     Cambridge University Press.

function kappa = local_circ_kappa(alpha)
    N = length(alpha);
    % Mean resultant length.  R is in [0, 1]: R = 1 means all angles
    % identical (kappa -> infinity), R = 0 means uniform on the circle
    % (kappa -> 0).
    R = sqrt(sum(cos(alpha))^2 + sum(sin(alpha))^2) / N;
    if R < 0.53
        kappa = 2*R + R^3 + 5*R^5/6;
    elseif R < 0.85
        kappa = -0.4 + 1.39*R + 0.43/(1-R);
    else
        kappa = 1/(R^3 - 4*R^2 + 3*R);
    end
    % Fisher (1993) small-sample correction; only meaningful below
    % about 15 observations.
    if N < 15 && N > 1
        if kappa < 2
            kappa = max(kappa - 2/(N*kappa), 0);
        else
            kappa = (N-1)^3 * kappa / (N^3 + N);
        end
    end
end
