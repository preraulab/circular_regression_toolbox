function test_circ_fit_schema(backends)
%TEST_CIRC_FIT_SCHEMA  Schema-parity + wiring smoke test for circ_fit.
%
%   test_circ_fit_schema                       % fitcirc_lme (fast)
%   test_circ_fit_schema({'fitcirc_lme','brms','bpnreg'})  % all
%
% Drives circ_fit on a real phase slice for each backend and asserts:
%   - make_circ_result validates (required-tier fields present)
%   - required-tier fieldnames are identical across backends
%   - AgeEffect.pValue / SelectedOrder are populated
%   - Trajectory is unwrapped (no |Δmean|>pi within an electrode)
%   - get_model_fit returns the right shapes through the circ path
% Renders an overlay PNG via plot_circ_fit.

if nargin < 1, backends = {'fitcirc_lme'}; end

here = fileparts(mfilename('fullpath'));
root = fullfile(here, '..', '..', '..');
addpath(genpath(fullfile(root, 'stats')));

% --- Build a test table from an existing slice's data.csv ---
slice = fullfile(here, '..', 'results', 'phase_fit_Phase_o2_007', 'data.csv');
D = readtable(slice);
tbl = table();
tbl.Phase     = D.y;          % treat the (already-wrapped) angle as the response
tbl.Age       = D.Age;
tbl.electrode = D.electrode;
tbl.sex       = D.sex;
tbl.Subj_ID   = D.Subj_ID;

feature = 'Phase';
cats    = {'electrode','sex'};
intx    = [true false];       % electrode interacts with Age block; sex main only
fml     = [build_model_formula(2, 'Age', feature, cats, intx) ' + (1|Subj_ID)'];

opts = struct();
opts.Select  = true;
opts.MaxOrder= 2;
opts.x_col   = 'Age';
opts.feature = feature;
opts.categorical_varnames = cats;
opts.xcol_categorical_interactions = intx;

required = {'Backend','Formula','ResponseName','Order','ThetaShift','Trajectory', ...
            'GOF','AgeEffect','OrderTable','SelectedOrder','SelectCriterion', ...
            'Diagnostics','Converged'};

results = cell(1, numel(backends));
fprintf('\n%-14s %6s %9s %9s %12s %10s\n','backend','order','R2_circ','MAE','ageP','crit');
for b = 1:numel(backends)
    r = circ_fit(tbl, fml, backends{b}, opts);
    results{b} = r;

    % required-tier present
    for f = required
        assert(isfield(r, f{1}), 'missing field %s for %s', f{1}, backends{b});
    end
    assert(istable(r.Trajectory) && height(r.Trajectory) > 0, 'empty Trajectory (%s)', backends{b});
    assert(isfinite(r.SelectedOrder), 'no SelectedOrder (%s)', backends{b});

    % trajectory unwrapped: no within-electrode wrap jumps
    for e = unique(r.Trajectory.electrode)'
        m = r.Trajectory.mean(r.Trajectory.electrode == e);
        assert(all(abs(diff(m)) < pi), 'trajectory not unwrapped (%s, elec %g)', backends{b}, e);
    end

    % circ-path trajectory wiring
    [yh, ci] = get_model_fit((7:80), 'Age', r, 'circ', cats, [1 0]);
    assert(numel(yh) == 74 && isequal(size(ci), [74 3]), 'get_model_fit shape (%s)', backends{b});

    fprintf('%-14s %6d %9.3f %9.3f %12.2e %10s\n', backends{b}, r.SelectedOrder, ...
        r.GOF.R2_circ, r.GOF.MAE_angular, r.AgeEffect.pValue, r.SelectCriterion);
end

% --- schema parity across backends ---
f0 = sort(required);
for b = 1:numel(results)
    fb = fieldnames(results{b});
    assert(all(ismember(required, fb)), 'backend %s missing required fields', backends{b});
end

% --- overlay plot ---
fig = figure('Visible','off','Color','w','Position',[100 100 900 500]);
ax = axes(fig);
plot_circ_fit(results, tbl, struct('ax', ax, 'feature', feature, 'plot_CI', false));
title(ax, sprintf('circ\\_fit overlay: %s', strjoin(backends, ', ')), 'Interpreter','tex');
out = fullfile(here, 'test_circ_fit_schema_overlay.png');
exportgraphics(fig, out, 'Resolution', 120);
close(fig);
fprintf('\nwrote %s\n', out);
fprintf('PASS: schema parity + wiring across {%s}\n', strjoin(backends, ', '));
end
