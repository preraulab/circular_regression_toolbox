function obj = fit_circ_method(tbl, formula, varargin)
%FIT_CIRC_METHOD  Dispatch wrapper for circular regression fits.
%
%   obj = fit_circ_method(tbl, formula, ...)
%
% Name-value options
%   Method     : 'legacy' (default), 'cboot', 'sub80'
%                  legacy : single fitcirc_lme call
%                  cboot  : cluster (subject-level) bootstrap, B fits, β bagged
%                           by componentwise median; cov_b replaced by the
%                           empirical covariance across bootstrap β's.
%                  sub80  : subsample-without-replacement of subjects
%                           (KeepFrac fraction), B fits, same bagging.
%   B          : number of resamples for cboot/sub80 (default 60)
%   KeepFrac   : fraction of subjects kept for sub80 (default 0.8)
%   AutoShift  : passed to fitcirc_lme (default true). When bagging, the
%                shift is computed once on the full data and frozen across
%                all bootstrap fits via ThetaShift, so β is comparable.
%   Verbose    : print progress (default false)
%
% The returned object is a fitcirc_lme model. For 'cboot'/'sub80', Beta,
% Coefficients.Estimate, cov_b, SE, tStat, pValue are replaced with bagged
% values. LogLikelihood, DFE, ConvergenceHistory, etc. remain from the base
% fit on the full data — used by LRT-based order selection upstream.

p = inputParser;
p.addParameter('Method',    'legacy');
p.addParameter('B',         60);
p.addParameter('KeepFrac',  0.8);
p.addParameter('AutoShift', true);
p.addParameter('Verbose',   false);
p.parse(varargin{:});
opt = p.Results;

method = lower(string(opt.Method));

if method == "legacy"
    obj = fitcirc_lme(tbl, formula, 'AutoShift', opt.AutoShift);
    return;
end
if method ~= "cboot" && method ~= "sub80"
    error('fit_circ_method:UnknownMethod', ...
          'Method must be ''legacy'', ''cboot'', or ''sub80''. Got "%s".', opt.Method);
end

% --- Compute a single shared ThetaShift on the full data so all bootstrap
% fits live in the same coordinate frame and β is comparable across them. ---
respName = strtrim(extractBefore(string(formula), '~'));
if opt.AutoShift
    theta_shift = circ_shift_min_var(tbl.(char(respName)));
else
    theta_shift = 0;
end

% --- Base fit on the full data, frozen ThetaShift ---
obj = fitcirc_lme(tbl, formula, 'ThetaShift', theta_shift);
beta_base = obj.Beta;
P = numel(beta_base);

% --- Resample loop ---
grpTok = regexp(char(formula), '\(\s*1\s*\|\s*([A-Za-z]\w*)\s*\)', 'tokens', 'once');
groupVar = grpTok{1};
subj   = unique(tbl.(groupVar));
n_subj = numel(subj);
B = opt.B;

Bm = zeros(P, B);
ok = false(B, 1);
for b = 1:B
    if method == "cboot"
        pick = subj(randi(n_subj, n_subj, 1));
        T_b  = resample_subjects(tbl, pick, groupVar);
    else  % sub80
        n_keep = max(2, round(opt.KeepFrac * n_subj));
        pick = subj(randperm(n_subj, n_keep));
        T_b  = tbl(ismember(tbl.(groupVar), pick), :);
    end
    try
        m = fitcirc_lme(T_b, formula, 'ThetaShift', theta_shift);
        if numel(m.Beta) == P
            Bm(:, b) = m.Beta;
            ok(b) = true;
        end
    catch ME
        if opt.Verbose
            fprintf('  fit_circ_method: bootstrap %d/%d failed: %s\n', b, B, ME.message);
        end
    end
end
if ~any(ok)
    warning('fit_circ_method:AllFailed', ...
            'All %d bootstrap fits failed; returning legacy fit.', B);
    return;
end
Bm = Bm(:, ok);

% --- Bagged β: componentwise median across resamples ---
beta_bag = median(Bm, 2);

% --- Bootstrap covariance of β (cluster-robust by construction for cboot) ---
cov_bag = cov(Bm', 1);  % MLE-style; rows are samples after transpose

% --- Substitute into the base model ---
obj.Beta = beta_bag;
obj.cov_b = cov_bag;

se_b = sqrt(diag(cov_bag));
t_b  = beta_bag ./ se_b;
df_t = max(obj.NumSubjects - 1, 1);
p_b  = 2 * (1 - tcdf(abs(t_b), df_t));

C = obj.Coefficients;
C.Estimate = beta_bag;
C.SE       = se_b;
C.tStat    = t_b;
C.pValue   = p_b;
obj.Coefficients = C;

if opt.Verbose
    fprintf('  fit_circ_method: %s bagged %d/%d successful (P=%d)\n', ...
        char(method), nnz(ok), B, P);
end
end


function T_out = resample_subjects(T, pick, groupVar)
% With-replacement bootstrap: each pick gets a fresh unique group ID so
% duplicates count as independent clusters in the random-intercept model.
parts = cell(numel(pick), 1);
next_id = 0;
for k = 1:numel(pick)
    rows = T(T.(groupVar) == pick(k), :);
    next_id = next_id + 1;
    if isnumeric(rows.(groupVar))
        rows.(groupVar) = repmat(next_id, height(rows), 1);
    else
        rows.(groupVar) = repmat(string(next_id), height(rows), 1);
    end
    parts{k} = rows;
end
T_out = vertcat(parts{:});
end
