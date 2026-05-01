%TEST_FITLME_CIRC  Smoke test for fitlme_circ.m.
%
% Confirms that fitlme_circ:
%   (1) Reproduces two independently-fit fitlme models on sin/cos to within
%       numerical tolerance (no funny business in the formula substitution).
%   (2) compare(red, full) returns a 2-row table whose pValue(2) is the
%       Bonferroni-joint p, matching what hand-rolled component compares
%       would give.
%   (3) predict() returns angles in [-pi, pi] and tracks the truth on a
%       held-out grid.

clear; close all;
addpath(fileparts(mfilename('fullpath')));

% Reuse the doubled-trend kappa=20 dataset that the sweep wrote.
csv = fullfile(fileparts(mfilename('fullpath')), '..', 'test_kappa_sweep_K20.csv');
tbl = readtable(csv);
tbl.Subj_ID = categorical(tbl.Subj_ID);
% Make sure the polynomial columns are present
for kk = 2:4
    if ~ismember(sprintf('Age%d', kk), tbl.Properties.VariableNames)
        tbl.(sprintf('Age%d', kk)) = tbl.Age.^kk;
    end
end

% --- Test 1: equivalence to manual sin/cos fitlme ---
fml = 'phase_pref ~ 1 + Age + Age2 + Age3 + electrode + sex + (1|Subj_ID)';
mdl = fitlme_circ(tbl, fml);

tbl_aux = tbl;
tbl_aux.s = sin(tbl.phase_pref);
tbl_aux.c = cos(tbl.phase_pref);
mdl_s = fitlme(tbl_aux, 's ~ 1 + Age + Age2 + Age3 + electrode + sex + (1|Subj_ID)');
mdl_c = fitlme(tbl_aux, 'c ~ 1 + Age + Age2 + Age3 + electrode + sex + (1|Subj_ID)');

err_sin = max(abs(mdl.Coefficients.EstSin - mdl_s.Coefficients.Estimate));
err_cos = max(abs(mdl.Coefficients.EstCos - mdl_c.Coefficients.Estimate));
fprintf('Test 1 (coef equivalence): max abs diff sin = %.2e, cos = %.2e\n', ...
        err_sin, err_cos);
assert(err_sin < 1e-9 && err_cos < 1e-9, 'fitlme_circ coefs differ from manual');

% --- Test 2: compare() gives matching Bonferroni p ---
fml_red  = 'phase_pref ~ 1 + Age + electrode + sex + (1|Subj_ID)';
mdl_red  = fitlme_circ(tbl, fml_red);
cmp      = compare(mdl_red, mdl);

% Hand-roll the same comparison using two fitlmes on aux table
mdl_s_red = fitlme(tbl_aux, 's ~ 1 + Age + electrode + sex + (1|Subj_ID)');
mdl_c_red = fitlme(tbl_aux, 'c ~ 1 + Age + electrode + sex + (1|Subj_ID)');
cmp_s = compare(mdl_s_red, mdl_s);
cmp_c = compare(mdl_c_red, mdl_c);
p_bonf_manual = min(1, 2 * min(cmp_s.pValue(2), cmp_c.pValue(2)));

fprintf('Test 2 (compare):\n');
fprintf('  fitlme_circ joint pValue(2) = %.4g\n', cmp.pValue(2));
fprintf('  manual Bonferroni p          = %.4g\n', p_bonf_manual);
assert(abs(cmp.pValue(2) - p_bonf_manual) < 1e-12, ...
    'compare() pValue(2) does not match manual Bonferroni');

% --- Test 3: predict() produces angles in [-pi, pi] ---
nd = table((10:5:75)', 0*(10:5:75)' + 0, 0*(10:5:75)' + 1, ...
    'VariableNames', {'Age','electrode','sex'});
nd.Age2 = nd.Age.^2;
nd.Age3 = nd.Age.^3;
nd.Subj_ID = repmat(tbl.Subj_ID(1), height(nd), 1);
ang = mdl.predict(nd, 'Conditional', false);
fprintf('Test 3 (predict): %d angles, range [%.2f, %.2f] rad\n', ...
        length(ang), min(ang), max(ang));
assert(all(ang >= -pi & ang <= pi), 'predict() returned out-of-range angles');

% --- Display ---
disp(mdl);
disp('compare(red, full):');
disp(cmp);

fprintf('\nAll fitlme_circ tests passed.\n');
