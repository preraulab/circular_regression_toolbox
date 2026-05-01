function [theta_star, y_shifted] = circ_shift_min_var(y)
%CIRC_SHIFT_MIN_VAR  Find a circular shift that minimizes the linear
% variance of the wrapped angles, then apply it.
%
%   [theta, y_shifted] = circ_shift_min_var(y)
%
% INPUT
%   y          Vector of angles in radians (any range; will be wrapped).
%
% OUTPUT
%   theta_star Shift in radians on (-pi, pi].
%   y_shifted  wrap(y - theta_star), in (-pi, pi].
%
% USE
%   The shift is chosen so that the shifted data minimizes its linear
%   variance — i.e. the seam (+-pi) lands in the largest empirical gap
%   on the circle. Equivalent to the antipode of the *circular mean*
%   for unimodal data; for multi-modal data it picks the geometrically
%   best split.
%
% After fitting any model on y_shifted, prediction unshifting is:
%   y_pred = wrap(y_pred_shifted + theta_star)
% The non-intercept regression coefficients are invariant under the
% shift; the intercept absorbs theta_star.

w = @(x) ((x + pi) - 2*pi*floor((x + pi) / (2*pi))) - pi;
thetas = linspace(-pi, pi, 721);                  % 0.5-degree grid
y = y(~isnan(y));
v = arrayfun(@(th) var(w(y - th)), thetas);
[~, k] = min(v);
theta_star = thetas(k);
if nargout > 1
    y_shifted = w(y(:) - theta_star);
end
end
