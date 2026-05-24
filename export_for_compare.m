function export_for_compare(dump_file, results_dir)
%EXPORT_FOR_COMPARE  Export a dump's tbl_full_save to the CSV/JSON contract
% the R/Stan comparison scripts read. Thin wrapper over write_circ_contract.
%
% Retained for the legacy 5-way comparison harness (run_compare.sh,
% sweep_5way_figs45.m, compare_*.R). Uses the variance-minimizing shift
% ('minvar') for byte-compatibility with previously cached results; the
% newer circ_fit path uses circular-mean centering instead.
%
% Builds the FULL electrode * Age + sex model regardless of what
% categorical_varnames the active fit used — relies on tbl_full_save
% having electrode, sex, Age, Subj_ID columns.
%
% Writes to <results_dir>: data.csv, eval_grid.csv, meta.json.

if nargin < 2, results_dir = fullfile(fileparts(dump_file), 'results'); end

S    = load(dump_file);
T    = S.tbl_full_save;
meta = S.meta;

opts = struct('Shift', 'minvar', 'x_col', 'Age', 'dump', dump_file);
write_circ_contract(T, meta.feature, meta.order, results_dir, opts);
end
