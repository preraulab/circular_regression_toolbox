function [theta_shift, y_shifted] = circ_center(y)
%CIRC_CENTER  Center angles by subtracting their circular mean.
%
%   [theta_shift, y_shifted] = circ_center(y)
%
% INPUT
%   y           Vector of angles in radians (any range; will be wrapped).
%
% OUTPUT
%   theta_shift Circular mean of y, atan2(mean(sin y), mean(cos y)), on (-pi, pi].
%   y_shifted   wrap(y - theta_shift), in (-pi, pi].
%
% This is the canonical preprocessing for every circular-regression
% backend (circ_fit_*). Centering the data at 0 pushes the +-pi seam to
% the far side of the circle from the data mass, which minimizes wrap
% discontinuities in both the fit and the plotted trajectory.
%
% After fitting on y_shifted, predictions are unshifted by adding
% theta_shift back and re-wrapping:  y_pred = wrap(y_pred_shifted + theta_shift).
% Non-intercept regression coefficients are invariant under the shift; the
% intercept absorbs theta_shift.
%
% Compare circ_shift_min_var (variance-minimizing seam placement), which
% is retained for the legacy comparison harness.

y = y(:);
good = ~isnan(y);
theta_shift = atan2(mean(sin(y(good))), mean(cos(y(good))));
if nargout > 1
    y_shifted = wrap_pi(y - theta_shift);
end
end

function w = wrap_pi(x)
w = ((x + pi) - 2*pi*floor((x + pi) / (2*pi))) - pi;
end
