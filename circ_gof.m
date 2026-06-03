function gof = circ_gof(y, yhat, n_par)
%CIRC_GOF  Circular goodness-of-fit metrics shared by all backends.
%
%   gof = circ_gof(y, yhat)
%   gof = circ_gof(y, yhat, n_par)
%
% INPUTS
%   y      observed angles (radians)
%   yhat   fitted/predicted angles at the same rows (radians)
%   n_par  number of fixed-effect parameters (for adjusted R2); optional
%
% OUTPUT struct fields
%   R2_circ      1 - SSE_circ / SST_circ, using circular dispersion
%                sum(1 - cos(resid)). The same formula the R backends use,
%                so R2_circ is directly comparable across fitcirc_lme / brms
%                / bpnreg.
%   R2_adj       1 - (1-R2)*(n-1)/(n-n_par)   (NaN if n_par not given)
%   MAE_angular  mean(|wrap(y - yhat)|)
%
% These are the only cross-backend-comparable fit metrics (LL/AIC/BIC are
% on different likelihood scales between model families).

y    = y(:);
yhat = yhat(:);
good = ~isnan(y) & ~isnan(yhat);
y    = y(good);
yhat = yhat(good);
n    = numel(y);

resid = wrap_pi(y - yhat);
mu_y  = atan2(mean(sin(y)), mean(cos(y)));
sse   = sum(1 - cos(resid));
sst   = sum(1 - cos(wrap_pi(y - mu_y)));

R2 = 1 - sse / max(sst, eps);
if nargin >= 3 && ~isempty(n_par)
    R2_adj = 1 - (1 - R2) * (n - 1) / max(n - n_par, 1);
else
    R2_adj = NaN;
end

gof = struct('R2_circ', R2, 'R2_adj', R2_adj, 'MAE_angular', mean(abs(resid)));
end

function w = wrap_pi(x)
w = ((x + pi) - 2*pi*floor((x + pi) / (2*pi))) - pi;
end
