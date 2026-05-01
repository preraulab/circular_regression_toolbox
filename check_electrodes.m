cd('/Users/Mike/Desktop/phase_fit_dumps');
files = dir('phase_fit_*_o2_*.mat');
fprintf('Slot                                          n   nElec  uniqueElec\n');
for ii = 1:numel(files)
    info = whos('-file', files(ii).name);
    var_names = {info.name};
    if ~any(strcmp(var_names,'tbl_full_save')), continue; end
    S = load(files(ii).name, 'tbl_full_save');
    T = S.tbl_full_save;
    if ~ismember('electrode', T.Properties.VariableNames), continue; end
    ue = unique(T.electrode(~isnan(T.electrode)));
    fprintf('%-50s n=%d  nElec=%d  vals=%s\n', files(ii).name, height(T), numel(ue), mat2str(ue'));
end
