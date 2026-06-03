function r = circ_vmrnd(mu, kappa, sz)
%CIRC_VMRND  Random samples from the von Mises distribution VM(mu, kappa).
%
%   r = circ_vmrnd(mu, kappa, sz)
%
% INPUTS
%   mu     scalar mean direction, radians, on (-pi, pi].
%   kappa  non-negative concentration parameter. kappa = 0 is the uniform
%          distribution on the circle; large kappa concentrates the
%          samples near mu. As a rough rule of thumb:
%               kappa = 1      -> half-width of distribution ~ 1 rad
%               kappa = 5      -> ~26 degrees
%               kappa = 50     -> ~8 degrees
%               kappa -> Inf   -> point mass at mu
%   sz     size of the requested sample, as a row vector (e.g. [n 1] or
%          [n m]). Output has the same shape.
%
% OUTPUT
%   r      angles in (-pi, pi], same shape as sz.
%
% METHOD
%   When kappa = 0, draw uniformly on (-pi, pi]. Otherwise use the
%   Best-Fisher (1979) rejection sampler from a wrapped-Cauchy envelope.
%   This is the standard, widely-implemented algorithm and runs in O(n)
%   expected time with no setup. The envelope acceptance rate stays high
%   for the kappa range typically used in mixed-effects circular models
%   (kappa < 100 ish).
%
% USE
%   The toolbox's parameter-recovery tests and tutorial both draw
%   simulated angles with circ_vmrnd. To reproduce a draw exactly, seed
%   the RNG with rng(seed) before the call.
%
% REFERENCE
%   Best, D. J., & Fisher, N. I. (1979). Efficient simulation of the
%   von Mises distribution. Applied Statistics, 28, 152-157.
%
% SEE ALSO  fitcirc_lme, circ_center, circ_gof.

if nargin < 3, sz = [1 1]; end
n = prod(sz);

if kappa == 0
    % Uniform on the circle: rejection method below would divide by zero.
    r = reshape(-pi + 2*pi*rand(n,1), sz);
    return;
end

% Best-Fisher 1979 constants (precomputed once per call).
a   = 1 + sqrt(1 + 4*kappa^2);
b   = (a - sqrt(2*a)) / (2*kappa);
r0  = (1 + b^2) / (2*b);

out = nan(n, 1);
k_acc = 0;
while k_acc < n
    u1 = rand;  u2 = rand;  u3 = rand;
    z  = cos(pi*u1);
    f  = (1 + r0*z) / (r0 + z);
    c  = kappa * (r0 - f);
    % Best-Fisher acceptance test (a two-stage shortcut that avoids the
    % expensive log call on most iterations).
    if c*(2 - c) - u2 > 0 || log(c/u2) + 1 - c >= 0
        k_acc = k_acc + 1;
        out(k_acc) = mu + sign(u3 - 0.5) * acos(f);
    end
end

% Wrap to the canonical (-pi, pi] interval.
r = reshape(((out + pi) - 2*pi*floor((out + pi)/(2*pi))) - pi, sz);
end
