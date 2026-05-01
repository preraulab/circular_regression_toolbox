%TEST_JOINT_TEST  Wald block-test sanity check.
%
% Builds a fitcirc_lme on synthetic data where one covariate block has
% a strong true effect and another is shuffled noise. Asserts:
%   - x_main joint Wald p < 1e-3 (true-positive)
%   - shuffled-covariate single coef p > 0.1 (true-null)
%   - ContrastIndex parser populates x_main and an interaction block
%
% Deterministic via rng(123). Runs in a few seconds.

clear; close all;
this_dir = fileparts(mfilename('fullpath'));
repo_root = fileparts(fileparts(fileparts(this_dir)));
addpath(genpath(fullfile(repo_root,'stats')));

rng(123);

n_subj = 60;
n_per  = 8;
ages   = repmat(linspace(20,70,n_per).', n_subj, 1);
sid    = repelem((1:n_subj).', n_per);
sex    = double(rand(n_subj,1) > 0.5);
sex_obs = sex(sid);
% Shuffled noise covariate: same sex column but permuted at the obs level
noise  = sex_obs(randperm(numel(sex_obs)));

beta0 = 0.10; beta_age = 0.03; beta_age2 = -0.0008;
phi  = circ_vmrnd(0, 6, [n_subj, 1]);
eps  = circ_vmrnd(0, 8, [numel(ages), 1]);
eta  = beta0 + beta_age*(ages - 45) + beta_age2*(ages - 45).^2 ...
     + phi(sid) + eps;
wrap = @(x) ((x + pi) - 2*pi*floor((x + pi)/(2*pi))) - pi;
y    = wrap(eta);

T = table(ages, sid, noise, y, 'VariableNames', {'Age','Subj_ID','noise','y'});

mdl = fitcirc_lme(T, 'y ~ Age^2 + noise + (1|Subj_ID)');
nm  = string(mdl.Coefficients.Name);
fprintf('Coefficient names: %s\n', strjoin(nm, ', '));

ci = mdl.ContrastIndex;
assert(isfield(ci,'x_main'), 'ContrastIndex.x_main missing');
assert(numel(ci.x_main) == 2, ...
    'x_main should hold 2 indices (Age, Age^2); got %d', numel(ci.x_main));

%% True-positive: joint Wald on Age block
p_age = mdl.coefTest('x_main').pValue;
fprintf('p_Age (joint, x_main) = %.2e\n', p_age);
assert(p_age < 1e-3, ...
    'True-positive Age block not detected by joint Wald: p=%.3g', p_age);

%% True-null: shuffled noise covariate
i_noise = find(nm == "noise");
assert(~isempty(i_noise), 'noise coefficient not in table');
p_noise = mdl.Coefficients.pValue(i_noise);
fprintf('p_noise (single)      = %.3f\n', p_noise);
assert(p_noise > 0.1, ...
    'True-null noise covariate spuriously significant: p=%.3g', p_noise);

%% Joint Wald via numeric index list (alternate API)
res = mdl.coefTest(ci.x_main);
assert(abs(res.pValue - p_age) < 1e-12, ...
    'Joint test via index list disagrees with named contrast: %.3g vs %.3g', ...
    res.pValue, p_age);

%% Block parser: refit with an interaction so x_x_<cat> populates
T2 = T;
T2.cat = double(T.Age > median(T.Age));
mdl2 = fitcirc_lme(T2, 'y ~ Age * cat + (1|Subj_ID)');
ci2 = mdl2.ContrastIndex;
assert(isfield(ci2,'x_main'), 'ContrastIndex.x_main missing in interaction model');
assert(isfield(ci2,'x_x_cat'), ...
    'ContrastIndex.x_x_cat missing for Age*cat interaction');
fprintf('Interaction model: x_main=%s, x_x_cat=%s\n', ...
    mat2str(ci2.x_main'), mat2str(ci2.x_x_cat'));

fprintf('\n*** test_joint_test passed. ***\n');


function r = circ_vmrnd(mu, kappa, sz)
    n = prod(sz);
    if kappa == 0
        r = -pi + 2*pi*rand(sz); return;
    end
    a = 1 + sqrt(1 + 4*kappa^2);
    b = (a - sqrt(2*a)) / (2*kappa);
    r0 = (1 + b^2) / (2*b);
    out = nan(n,1); k = 0;
    while k < n
        u1 = rand; u2 = rand; u3 = rand;
        z  = cos(pi*u1);
        f  = (1 + r0*z) / (r0 + z);
        c  = kappa * (r0 - f);
        if c*(2 - c) - u2 > 0 || log(c/u2) + 1 - c >= 0
            k = k + 1;
            out(k) = mu + sign(u3 - 0.5) * acos(f);
        end
    end
    r = reshape(((out + pi) - 2*pi*floor((out + pi)/(2*pi))) - pi, sz);
end
