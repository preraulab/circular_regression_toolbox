function model = build_model_formula(order, x_col, feature, categorical_varnames, xcol_categorical_interactions)
%BUILD_MODEL_FORMULA  Build the Wilkinson model string (without random term).
%
%   model = build_model_formula(order, x_col, feature, ...
%                               categorical_varnames, xcol_categorical_interactions)
%
% Returns a string like:
%   'feature ~ 1'                                   (order 0, no categoricals)
%   'feature ~ 1 + Age^2 + electrode + sex'         (no interaction)
%   'feature ~ 1 electrode*Age^2 + sex'             (electrode interacts with the Age block)
%
% The random-intercept term (' + (1|Subj_ID)') is appended by the caller.
% Factored out of get_single_order_model.m so the circular dispatcher
% (circ_fit) and the LME/GLM path build identical formulas.
%
% INPUTS
%   order                          polynomial degree in x_col (0 => intercept only)
%   x_col                          predictor name, e.g. 'Age'
%   feature                        response name
%   categorical_varnames           cellstr of categorical main effects ({} for none)
%   xcol_categorical_interactions  1xC logical: which categoricals interact with x_col^order

if isempty(categorical_varnames)
    model_categorical = '';
else
    model_categorical = expand_varnames(categorical_varnames);
end

if order == 0
    model = [feature ' ~ 1', model_categorical];
elseif isempty(xcol_categorical_interactions) || all(~xcol_categorical_interactions)
    model = [feature ' ~ 1 +' x_col '^', num2str(order), model_categorical];
else
    model_interactions = categorical_varnames(xcol_categorical_interactions);
    model_interactions = cellfun(@(s) [s '*' x_col '^', num2str(order)], ...
                                 model_interactions, 'UniformOutput', false);
    model_interactions = expand_varnames(model_interactions);
    model = [feature ' ~ 1 ' model_interactions model_categorical];
end
end


function model_varnames = expand_varnames(varnames)
% Build the trailing ' + a + b + c' fragment for a list of names. Empty
% input yields an empty string.
if isempty(varnames)
    model_varnames = '';
    return
end
parts = strjoin(varnames, ' + ');
model_varnames = [' + ', parts];
end
