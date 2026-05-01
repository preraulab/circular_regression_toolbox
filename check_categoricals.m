S = load('/Users/Mike/Desktop/phase_fit_dumps/phase_fit_Phase_o2_001.mat');
T = S.tbl_full_save;
fprintf('vars: %s\n', strjoin(T.Properties.VariableNames, ', '));
fprintf('rows: %d\n', height(T));
fprintf('Subj_ID class: %s, n unique: %d\n', class(T.Subj_ID), numel(unique(T.Subj_ID)));
for cc = {'electrode','sex','race','mode_cluster','ahi'}
    nm = cc{1};
    if ismember(nm, T.Properties.VariableNames)
        ue = unique(T.(nm));
        if isnumeric(ue), ue_str = mat2str(ue'); else, ue_str = strjoin(string(ue),','); end
        fprintf('%-15s class=%-12s n_unique=%d   vals=%s\n', nm, class(T.(nm)), numel(ue), ue_str);
    end
end
