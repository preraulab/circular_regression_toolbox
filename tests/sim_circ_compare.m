function sim_circ_compare(backends, outpng)
%SIM_CIRC_COMPARE  Fit + overlay all circ_fit backends on noisy, wrapping
% simulated phase data.
%
%   sim_circ_compare                                   % all four backends
%   sim_circ_compare({'fitcirc_lme','lme4'})           % a subset
%
% The true mean trajectory sweeps ~6 rad across the lifespan so it crosses
% the +pi seam once (~age 55) — i.e. the data genuinely wraps around — and
% von-Mises-ish observation noise plus subject phase offsets make it noisy.
% Saves and opens an overlay PNG; prints a per-backend comparison table.

if nargin < 1 || isempty(backends)
    backends = {'fitcirc_lme','brms','lme4','bpnreg'};
end
here = fileparts(mfilename('fullpath'));
root = fullfile(here, '..', '..', '..');
addpath(genpath(fullfile(root, 'stats')));
if nargin < 2 || isempty(outpng)
    outpng = fullfile(here, 'sim_circ_compare.png');
end

% ---- Simulate ----
rng(7);
n_subj = 60; n_per = 10; N = n_subj*n_per;
Subj = repelem((1:n_subj)', n_per);
Age  = 7 + 73*rand(N,1);
u    = (Age - 7) / 73;
eta_true = -1 + 5*u + 2*u.^2;            % unwrapped truth; crosses +pi ~age55

sigma_phi = 0.5;                         % subject phase spread (rad)
sigma_eps = 0.8;                         % observation noise (rad) -> noisy
phi = sigma_phi * randn(n_subj,1);
y   = wrapToPi(eta_true + phi(Subj) + sigma_eps*randn(N,1));

tbl = table();
tbl.Phase   = y;
tbl.Age     = Age;
tbl.Subj_ID = Subj;

% ---- Fit every backend through the unified dispatcher ----
circ_fit_config('reset');
opts = struct();
opts.Select   = true;
opts.MaxOrder = 2;
opts.x_col    = 'Age';
opts.feature  = 'Phase';
opts.Band     = true;
opts.Chains   = 4;
opts.Iter     = 1500;        % brms: keep the demo quick
opts.Warmup   = 750;
opts.categorical_varnames = {};
opts.xcol_categorical_interactions = [];
fml = [build_model_formula(2, 'Age', 'Phase', {}, []) ' + (1|Subj_ID)'];

results = {};
fprintf('\n%-14s %6s %9s %9s %12s %10s\n','backend','order','R2_circ','MAE','ageP','crit');
for b = 1:numel(backends)
    r = circ_fit(tbl, fml, backends{b}, opts);
    results{end+1} = r; %#ok<AGROW>
    fprintf('%-14s %6d %9.3f %9.3f %12.2e %10s\n', backends{b}, r.SelectedOrder, ...
        r.GOF.R2_circ, r.GOF.MAE_angular, r.AgeEffect.pValue, r.SelectCriterion);
end

% ---- Overlay plot ----
fig = figure('Color','w','Position',[80 80 1000 620],'Visible','off');
ax  = axes(fig);
plot_circ_fit(results, tbl, struct('ax', ax, 'feature', 'Phase', 'plot_CI', true));

% Add the true generating trajectory (black dashed, triple-plotted).
ag = (7:80)'; ug = (ag-7)/73; eg = -1 + 5*ug + 2*ug.^2;
for off = [0 2*pi -2*pi]
    plot(ax, ag, eg + off, 'k--', 'LineWidth', 1.6, 'HandleVisibility', 'off');
end
ylim(ax, [-pi pi]);
title(ax, sprintf('Noisy wrapping phase ~ Age: %s  (black dashed = truth)', ...
    strjoin(backends, ', ')), 'Interpreter','none');

exportgraphics(fig, outpng, 'Resolution', 140);
close(fig);
fprintf('\nwrote %s\n', outpng);
end
