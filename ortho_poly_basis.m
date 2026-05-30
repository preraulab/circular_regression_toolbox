function [P, info] = ortho_poly_basis(x, k, info)
%ORTHO_POLY_BASIS  Orthogonal polynomial basis of degree k over a sample.
%
%   [P, info] = ortho_poly_basis(x, k)              build new basis from x
%    P        = ortho_poly_basis(x, k, info)        apply basis to new x
%
% Mirrors R's stats::poly(x, k) (with the centered, orthonormalized
% convention). `x` is an n-vector; `k` is the maximum polynomial degree.
% The returned P is n x k, with columns orthogonal to each other and to
% the constant column 1 (over the sample used to BUILD the basis):
%   * P(:,j) is the j-th orthogonal polynomial evaluated at x.
%   * P(:,1) is monotone in x (the "linear" component, centered + scaled).
%   * P(:,j+1) is the residual of x^(j+1) regressed on {1, P(:,1)..P(:,j)}.
%
% Why use this instead of `[x, x.^2, x.^3]` ("raw power basis"):
%   1. NUMERICAL CONDITIONING. Raw columns span O(1)..O(x_max^k) and have
%      cross-correlation > 0.99 for typical lifespan ranges. IRLS/EM on
%      such designs jitter or settle in wrong basins. P has orthonormal
%      columns by construction, so X'X is the identity.
%   2. INTERPRETABLE PER-COEFFICIENT INFERENCE. With raw columns, the
%      Wald p-value on beta_x is "is there a linear trend AFTER beta_x^2
%      absorbed as much as it could?" -- not "is there a linear trend?".
%      With orthogonal columns the cross terms are zero in the sample
%      design, so each beta_P(:,j) has a standalone marginal interpretation.
%   3. NESTED ORDER COMPATIBILITY. QR is greedy: cols 1..j of QR
%      ([x, x.^2, ..., x.^K]) match QR([x, ..., x.^j]) exactly for any
%      K >= j. So fitting orders 0, 1, ..., K all use a *consistent*
%      column basis -- adding higher orders doesn't reshuffle the lower
%      coefficients. Warm-starting EM by name therefore actually
%      transfers structure, instead of just seeding zeros into a
%      reparameterized space.
%
% What is *not* changed by using P instead of raw power columns:
%   * The fitted curve (polynomial subspace is identical).
%   * Joint Wald tests for "any age effect" (Wald is invariant under
%     nonsingular linear reparameterization of the tested block).
%   * R^2, residuals, predicted intervals.
%
% Implementation:
%   Build:   M = [x, x.^2, ..., x.^k];  Mc = M - mean(M); [Q, R] = qr(Mc);
%            P = Q;  info = struct('means', mean(M), 'R', R, 'k', k).
%   Apply:   M_e = [x_e, x_e.^2, ..., x_e.^k];  Mc_e = M_e - info.means;
%            P_e = Mc_e / info.R.
% Both paths produce columns in the same orthogonal basis, so a model
% fitted on P generalizes to predictions at any x via the same
% transformation. info is the only side-channel needed to apply the
% basis at new x; no symbolic algebra required.
%
% Reference convention: matches R's `stats::poly` (which itself cites the
% two references below in its help page).
%
% REFERENCES
%   Chambers, J.M. & Hastie, T.J. (eds.) (1992). Statistical Models in S.
%     Wadsworth & Brooks/Cole Advanced Books & Software, Pacific Grove,
%     CA. ISBN 0-534-16765-9. Source for the S/R `poly` orthogonal
%     polynomial convention and treatment of polynomial regression terms.
%   Kennedy, W.J. Jr. & Gentle, J.E. (1980). Statistical Computing.
%     Marcel Dekker, New York. The three-term recursion for evaluating
%     orthogonal polynomials given coefficient summaries (used by R's
%     internal `predict.poly` -- pp. 343-344).

x = x(:);
if nargin < 3 || isempty(info)
    M     = x .^ (1:k);
    means = mean(M, 1);
    Mc    = M - means;
    [Q, R] = qr(Mc, 0);
    P     = Q;
    info  = struct('means', means, 'R', R, 'k', k);
else
    if isfield(info, 'k') && info.k ~= k
        error('ortho_poly_basis:DegreeMismatch', ...
              'info was built at k=%d; cannot apply at k=%d.', info.k, k);
    end
    M  = x .^ (1:info.k);
    Mc = M - info.means;
    P  = Mc / info.R;
end
end
