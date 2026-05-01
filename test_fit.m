addpath(genpath('/Users/Mike/code/projects/trends_v_individual/stats'));
S = load('/Users/Mike/Desktop/phase_fit_dumps/phase_fit_pref_phase_o2_003.mat');
T = S.tbl_full_save;
feat = S.meta.feature;
keep = ~isnan(T.Age) & ~isnan(T.(feat)) & ~isnan(T.electrode) & ~isnan(T.sex);
T = T(keep,:);
fprintf('class Age=%s electrode=%s sex=%s Subj_ID=%s\n', class(T.Age), class(T.electrode), class(T.sex), class(T.Subj_ID));
disp(unique(T.electrode));
disp(unique(T.sex));

formula = 'pref_phase ~ Age^2 * electrode + sex + (1|Subj_ID)';
fprintf('Trying fitlme...\n');
try
    tmp = fitlme(T, formula);
    disp(tmp.CoefficientNames);
catch ME
    fprintf('FAILED: %s\n', ME.message);
end
