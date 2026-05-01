function export_for_compare(dump_file, results_dir)
%EXPORT_FOR_COMPARE  Export a dump's tbl_full_save to CSVs that R/Stan
% comparison scripts can read. Builds the FULL electrode * Age + sex
% model regardless of what categorical_varnames the active fit used —
% relies on tbl_full_save (saved by the updated get_single_order_model)
% having electrode, sex, Age, Subj_ID columns.
%
% Writes to <results_dir>:
%   data.csv          - long-format data: y, Age, electrode, sex, Subj_ID
%   eval_grid.csv     - prediction grid: Age sweep × electrode {0,1} × sex 0
%   meta.json         - feature, x_col, order, full formula

if nargin < 2, results_dir = fullfile(fileparts(dump_file), 'results'); end
if ~exist(results_dir,'dir'), mkdir(results_dir); end

S = load(dump_file);
T = S.tbl_full_save;
meta = S.meta;
feat = meta.feature;

% Required columns
need = {'Age', feat, 'Subj_ID'};
for cc = need
    assert(ismember(cc{1}, T.Properties.VariableNames), ...
        'export_for_compare:missingColumn', '%s missing from tbl_full_save', cc{1});
end
has_elec = ismember('electrode', T.Properties.VariableNames);
has_sex  = ismember('sex',       T.Properties.VariableNames);

% Drop NaN rows in any required numeric column
keep = ~isnan(T.Age) & ~isnan(T.(feat));
if has_elec, keep = keep & ~isnan(T.electrode); end
if has_sex,  keep = keep & ~isnan(T.sex); end
T = T(keep, :);

% Fall back if only one electrode level is present (single-cluster slot)
if has_elec && numel(unique(T.electrode)) < 2
    fprintf('export_for_compare: only one electrode level (%d) in this slot — dropping electrode from model\n', unique(T.electrode));
    has_elec = false;
end
if has_sex && numel(unique(T.sex)) < 2
    fprintf('export_for_compare: only one sex level — dropping sex from model\n');
    has_sex = false;
end

% Rename response to 'y' for downstream tools
out = table();
out.y       = T.(feat);
out.Age     = T.Age;
if has_elec, out.electrode = T.electrode; end
if has_sex,  out.sex       = T.sex; end
out.Subj_ID = T.Subj_ID;
writetable(out, fullfile(results_dir, 'data.csv'));

% Eval grid: Age sweep × electrode levels × sex=0
x_eval = (7:80)';
grid = table();
elec_levels = 0; if has_elec, elec_levels = [0 1]; end
for elec = elec_levels
    G = table();
    G.Age = x_eval;
    if has_elec, G.electrode = elec * ones(numel(x_eval),1); end
    if has_sex,  G.sex       = zeros(numel(x_eval),1); end
    G.Subj_ID = repmat(T.Subj_ID(1), numel(x_eval), 1);
    grid = [grid; G]; %#ok<AGROW>
end
writetable(grid, fullfile(results_dir, 'eval_grid.csv'));

% Build the FULL formula: y ~ poly(Age, order) * electrode + sex + (1|Subj_ID)
% (overrides whatever was in the dump's meta.formula — the dump may have
%  been from a sex-only fit, but here we want the full model.)
ord = meta.order;
if ord == 0
    rhs = '1';
else
    rhs = sprintf('Age^%d', ord);
end
if has_elec, rhs = [rhs ' * electrode']; end
if has_sex,  rhs = [rhs ' + sex']; end
formula_full = sprintf('y ~ %s + (1|Subj_ID)', rhs);

fid = fopen(fullfile(results_dir, 'meta.json'),'w');
fprintf(fid, '{"formula":"%s","x_col":"Age","feature":"%s","order":%d,"has_electrode":%d,"has_sex":%d,"dump":"%s"}\n', ...
    formula_full, feat, ord, has_elec, has_sex, dump_file);
fclose(fid);

fprintf('Exported %s (n=%d, has_elec=%d) -> %s\n  full formula: %s\n', ...
    dump_file, height(out), has_elec, results_dir, formula_full);
end
