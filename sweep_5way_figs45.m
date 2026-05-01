function sweep_5way_figs45(dump_dir, results_root, target_order)
%SWEEP_5WAY_FIGS45  Run the 5-way comparison on every (feature, cluster)
% combo used in figures 4 and 5 of the paper, at a single polynomial
% order. Aggregates per-method trajectories so per_method_summary can
% draw one figure per method showing all combos.
%
%   sweep_5way_figs45()                    % defaults below
%   sweep_5way_figs45(dump_dir, results_root, target_order)
%
% Defaults:
%   dump_dir     = /Users/Mike/Desktop/phase_fit_dumps
%   results_root = stats/circular_regression/results
%   target_order = 4 if order-4 dumps exist for every combo, else
%                  the highest order that does.
%
% Combos for figs 4 & 5 (the circ-typed features):
%   SOPH  pref_phase, Theta              x clusters 1..4   ->  8
%   SOPhH Phase, STDphase, Theta         x clusters 1..2   ->  6
%                                                      total: 14

if nargin < 1 || isempty(dump_dir)
    dump_dir = '/Users/Mike/Desktop/phase_fit_dumps';
end
if nargin < 2 || isempty(results_root)
    results_root = fileparts(mfilename('fullpath'));
    results_root = fullfile(results_root, 'results');
end
if nargin < 3 || isempty(target_order)
    target_order = [];                                   % auto-detect
end

% --- Catalog every dump file: feature, order, cluster ---
files = dir(fullfile(dump_dir, 'phase_fit_*.mat'));
catalog = struct('name',{},'feat',{},'order',{},'cluster',{},'path',{});
for ii = 1:numel(files)
    f = files(ii);
    S = load(fullfile(f.folder, f.name));
    if ~isfield(S,'tbl_full_save'), continue; end
    T = S.tbl_full_save;
    if ~ismember('mode_cluster', T.Properties.VariableNames), continue; end
    uc = unique(T.mode_cluster);
    if numel(uc) ~= 1, continue; end
    catalog(end+1) = struct('name', f.name, 'feat', S.meta.feature, ...
        'order', S.meta.order, 'cluster', double(uc), ...
        'path', fullfile(f.folder, f.name)); %#ok<AGROW>
end
fprintf('Cataloged %d dumps from %s\n', numel(catalog), dump_dir);

% --- Pick target order automatically if not given ---
if isempty(target_order)
    target_order = max([catalog.order]);
    fprintf('No target_order specified; using highest available = %d\n', target_order);
end

% --- The 14 fig-4/5 combos ---
combos = [ ...
    cellrow('pref_phase', 1); cellrow('pref_phase', 2);
    cellrow('pref_phase', 3); cellrow('pref_phase', 4);
    cellrow('Theta',      1); cellrow('Theta',      2);
    cellrow('Theta',      3); cellrow('Theta',      4);
    cellrow('Phase',      1); cellrow('Phase',      2);
    cellrow('STDphase',   1); cellrow('STDphase',   2);
    cellrow('Theta',      1); cellrow('Theta',      2)];      % SOPhH Theta is grouped 1:2 too
% The SOPH Theta and SOPhH Theta are different dump names — for fig 4/5
% the SOPH "Theta" dumps came from SOPH (4 clusters); SOPhH "Theta"
% dumps came from SOPhH (2 clusters). Same feature column, different
% mode_cluster column. We disambiguate by which dump file matches the
% combo's cluster id; the SOPH Theta dumps use clusters 1..4, the SOPhH
% Theta dumps clusters 1..2 (collide with the first two SOPH ones —
% pick whichever exists and let the caller cross-check via n_subj).

% De-dupe combos by (feat, cluster); duplicates resolve to whichever
% dump matches first.
keys = arrayfun(@(k) sprintf('%s_%d', combos{k,1}, combos{k,2}), ...
    1:size(combos,1), 'UniformOutput', false);
[~, idx] = unique(keys, 'stable');
combos = combos(idx, :);
n_combo = size(combos, 1);
fprintf('Running %d (feature, cluster) combos at order=%d\n', n_combo, target_order);

% --- For each combo, find matching dump and run the 5-way comparison ---
runs = struct('combo',{},'results_dir',{},'success',{},'stats',{});
for ii = 1:n_combo
    feat = combos{ii,1}; cluster = combos{ii,2};
    candidates = catalog(strcmp({catalog.feat}, feat) & ...
                         [catalog.cluster] == cluster & ...
                         [catalog.order]   == target_order);
    if isempty(candidates)
        fprintf('  [%2d/%2d] %-12s cluster %d  -> no dump at order %d (skipping)\n', ...
            ii, n_combo, feat, cluster, target_order);
        continue
    end
    dump_path = candidates(1).path;
    [~, base, ~] = fileparts(dump_path);
    rd = fullfile(results_root, base);
    have_all = exist(fullfile(rd, 'fit_stats.csv'), 'file') && ...
               exist(fullfile(rd, 'fitcirc_lme_predictions.csv'), 'file') && ...
               exist(fullfile(rd, 'brms_predictions.csv'), 'file');
    if have_all
        fprintf('  [%2d/%2d] %-12s cluster %d  -> %s   (cached)\n', ii, n_combo, feat, cluster, base);
        ok = true;
        fit_stats = readtable(fullfile(rd, 'fit_stats.csv'));
    else
        fprintf('  [%2d/%2d] %-12s cluster %d  -> %s\n', ii, n_combo, feat, cluster, base);
        try
            run_compare_one(dump_path, rd);
            ok = true;
            fit_stats = readtable(fullfile(rd, 'fit_stats.csv'));
        catch ME
            fprintf('         FAILED: %s\n', ME.message);
            ok = false; fit_stats = [];
        end
    end
    runs(end+1) = struct('combo', {combos(ii,:)}, 'results_dir', rd, ...
                         'success', ok, 'stats', {fit_stats}); %#ok<AGROW>
end

% --- Save the run manifest ---
mfile = fullfile(results_root, sprintf('sweep_manifest_o%d.mat', target_order));
save(mfile, 'runs', 'target_order');
fprintf('Saved sweep manifest -> %s\n', mfile);
end


function c = cellrow(feat, cluster)
c = {feat, cluster};
end


function run_compare_one(dump_path, rd)
% Drive a single 5-way run from inside MATLAB. Calls export_for_compare,
% then shells out to the three R scripts, then plot_5way_compare.
this_dir = fileparts(mfilename('fullpath'));
if ~exist(rd,'dir'), mkdir(rd); end

% 1) Export shifted CSVs + meta.json for this dump
export_for_compare(dump_path, rd);

% 2) R comparison scripts (each writes *_predictions.csv and *_stats.json)
for script = {'compare_brms.R','compare_lme4.R','compare_bpnreg.R'}
    cmd = sprintf('Rscript %s %s', fullfile(this_dir, script{1}), rd);
    [status, out] = system(cmd);
    if status ~= 0
        warning('%s failed (status=%d):\n%s', script{1}, status, out);
    end
end

% 3) Render the 5-way figure + fit_stats.csv
plot_5way_compare(dump_path, rd);
end
