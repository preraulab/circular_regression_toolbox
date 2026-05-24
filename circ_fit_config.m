function out = circ_fit_config(varargin)
%CIRC_FIT_CONFIG  Process-global config for the circular-regression dispatcher.
%
%   cfg = circ_fit_config()                       % get current config
%   cfg = circ_fit_config('get')                  % same
%   circ_fit_config('set', struct('Method','cboot','B',60,'KeepFrac',0.8))
%   circ_fit_config('reset')                      % restore defaults
%
% Read by stats/get_single_order_model.m for the 'circ' branch.
% Set by run_lifespan_pipeline.m before the stats stage so the chosen
% method propagates into every fit without threading params through every
% intermediate signature.

persistent CFG
if isempty(CFG)
    CFG = default_cfg();
end

if nargin == 0 || (ischar(varargin{1}) && strcmpi(varargin{1}, 'get'))
    out = CFG;
    return;
end

action = lower(string(varargin{1}));
switch action
    case "set"
        s = varargin{2};
        if isstruct(s)
            fns = fieldnames(s);
            for k = 1:numel(fns)
                CFG.(fns{k}) = s.(fns{k});
            end
        else
            error('circ_fit_config:BadArg','set requires a struct.');
        end
        out = CFG;
    case "reset"
        CFG = default_cfg();
        out = CFG;
    otherwise
        error('circ_fit_config:BadAction','Unknown action "%s".', varargin{1});
end
end


function cfg = default_cfg()
cfg = struct( ...
    'Backend',     'fitcirc_lme', ... % 'fitcirc_lme' | 'brms' | 'lme4' | 'bpnreg'
    'MaxOrder',    2, ...
    'Select',      true, ...
    'Resample',    'none', ...        % 'none' | 'cboot' | 'sub80'  (fitcirc_lme only)
    'Method',      'legacy', ...      % legacy alias for Resample ('legacy' -> 'none')
    'B',           60, ...
    'KeepFrac',    0.8, ...
    'Chains',      4, ...             % brms
    'Iter',        2000, ...
    'Warmup',      1000, ...
    'Seed',        1, ...
    'AdaptDelta',  0.95, ...
    'Band',        true, ...          % lme4 CI band via bootMer
    'BrmsFallback',true);             % fall back to fitcirc_lme if an R backend fails
end
