function [estimated_phase,mdl] = circular_regression(x, y,categorical_vars,varnames, order,xcol_categorical_interactions,b0, iterations, plot_on, verbose)
%CIRCULAR_REGRESSION  Perform circular regression on phase data
%
%   Usage:
%       stats = circular_regression(x, y, b0, order, iterations, plot_on, verbose)
%
%   Input:
%       x: vector - Nx1 predictor variable, required
%       y: vector - Nx1 response variable (wrapped phase data), required
%       categorical_vars: Nxc categorical variables (default:[])
%       varnames: cell array of name of x and of categorical_vars (default
%       x,x^1,c1,c2 etc)
%       b0: vector - initial parameter estimates (default: zeros(order+1,1))
%       order: integer - polynomial order for the regression (default: 2)
%       iterations: integer - number of iterations for IRLS (default: 100)
%       plot_on: boolean - flag to plot the results (default: false)
%       verbose: boolean - flag to display the results (default: false)
%
%   Output:
%       stats: structure containing the following fields:
%           deviance: deviance of the fitted model
%           t_stats: t-statistics of the parameter estimates
%           p_values: p-values of the parameter estimates
%
%   Example:
%   In this example, we generate some synthetic data and perform circular regression.
%       x = linspace(0, 80, 1000)';
%       y = wrapToPi(-0.06 * x + 0.001 * x.^2 - pi/5 + randn(size(x))/2);
%       stats = circular_regression(x, y);
%
%   Copyright 2014 Prerau Laboratory sleepEEG.org
%**********************************************

if nargin == 0
    % Plot raw data
    x = linspace(0, 80, 1000)';  % Generate 1000 points linearly spaced between 0 and 80
    y = wrapToPi(-0.06 * x + 0.001 * x.^2 - pi/5 + randn(size(x))/2);  % Generate wrapped phase data with noise
    b0 = [0 0 0]';
    order = 2;
    [estimated_phase,b,stats] =circular_regression(x, y, order,b0,1000,true,true);
    return;
end

% Make into columns
if isrow(x)
    x = x(:);
end
if isrow(y)
    y = y(:);
end
if nargin <3
    categorical_vars =[];
end

if nargin < 4
    varnames = {};
end

if nargin < 5
    order = 2;
end
if nargin < 6
    xcol_categorical_interactions = [];
end
if nargin < 7
    b0 = zeros(order + 1+size(categorical_vars,2), 1);
end

if nargin < 8
    iterations = 100;
end

if nargin < 9
    plot_on = false;
end

if nargin < 10
    verbose = false;
end

assert(isequal(size(x),size(y)),'x and y must have the same dimensions');

% Fit circular linear model - identity link
n = length(x);
X = zeros(n, order + 1);  % Preallocate the design matrix

% Intercept term
X(:, 1) = 1;  
%Order terms
for ii = 1:order
    X(:, ii + 1) = x.^ii;
end
if ~isempty(categorical_vars)
    X = [X,categorical_vars];
end
if ~isempty(xcol_categorical_interactions)&order>0
    categorical_varnames = varnames(2:end);
    interactions = categorical_varnames(xcol_categorical_interactions);
    xcol_terms = X(:,2:order+1);
    for ii = 1:length(interactions)
        curr_interaction = interactions{ii};
        categorical_idx = strcmpi(categorical_varnames,curr_interaction);
        interaction_term = xcol_terms.*categorical_vars(:,categorical_idx);
        X = [X,interaction_term];
    end
end
b = b0;  % Initial parameter estimates

% IRLS process
for iter = 1:iterations
    % Linear predictor
    eta = X * b;

    % Circular residuals
    residuals = y - eta;

    % Adjusted response
    z = eta + sin(residuals);

    % Weighted least squares update
    b = (X' * X) \ (X' * z);
end

% Estimated phase values
estimated_phase = wrapToPi(X * b);

% Plot results
if plot_on
    figure;
    subplot(1, 2, 1);
    hold on;
    raw_data_plot = plot(x, y, 'b.');
    fitted_line_plot = plot(x, estimated_phase, 'r.');
    legend([raw_data_plot, fitted_line_plot], {'Raw Data', 'Fitted Line'});
    xlabel('Predictor');
    ylabel('Phase');
    title('Model Fit');

    subplot(1, 2, 2);
    polarhistogram(y - estimated_phase, 50);  % Plot residuals
    title('Residuals');
end

% Model outputs
resid = wrapToPi(y - estimated_phase);
deviance = 2 * sum(1 - cos(resid));  % Deviance of the fitted model
kappa = min(700,circ_kappa(resid));  % Concentration parameter of the residuals. Put uper bound because not computable if value over 700
A1 = besseli(1, kappa) / besseli(0, kappa);  % Ratio of modified Bessel functions

% Covariance matrix of the parameter estimates
cov_b = (X' * X) \ eye(size(X, 2)) / kappa / A1;
standard_error = sqrt(diag(cov_b));  % Standard errors of the parameter estimates

% t-statistics and p-values for the parameter estimates
t_statistics = b ./ standard_error;
p_values = (1 - tcdf(abs(t_statistics), length(y) - length(b)));

% R Squared
n = length(x);
% p = order+1+size(categorical_vars,2)+order*sum(xcol_categorical_interactions);
p = size(X,2);     
DFE = n-p;
stats = get_circ_regression_stats(y,estimated_phase,n,p);
% Log-Likelihood
% sigma2 =(1/n)*stats.SSE; % MATLAB linearModel appears to use MSE for sigma2
% logL = (-(n/2) * log(2 * pi)) - ((n/2) * log(sigma2)) - (stats.SSE / (2 * sigma2));

logL = -n * log(2*pi*besseli(0,kappa)) + kappa * sum(cos(resid));

% Coeff names
coeffnames = cell(order+1+size(categorical_vars,2),1);
coeffnames{1} = '(Intercept)';
num_c = size(categorical_vars,2);
num_interactions = sum(xcol_categorical_interactions);
if isempty(varnames)
    for ii = 1:order
        coeffnames(ii+1) = {['x^',num2str(ii)]};
    end
    for ii = 1:num_c
        coeffnames(ii+1+order) = {['c',num2str(ii)]};
    end
else
    for ii = 1:order
        if ii ==1
            coeffnames(ii+1) = {varnames{1}};
        else
            coeffnames(ii+1) = {[varnames{1},'^',num2str(ii)]};
        end
    end
    for ii = 1:num_c
        coeffnames(ii+1+order) = {[varnames{1+ii}]};
    end
    if order>0
        for ii = 1:num_interactions
            interaction_names = cellfun(@(s) [s ':' interactions{ii}], coeffnames(2:order+1), 'UniformOutput', false);
            coeffnames(1+order+num_c+1:1+order+num_c+1+num_interactions*(order-1)) = interaction_names;
        end
    end
end
% SToring outputs in struct in similar format to LinearModel
num_coeff = length(b);
col_names = {'Estimate','SE','tStat','pValue'};
coeff_tbl = array2table(NaN(num_coeff,length(col_names)),"RowNames",coeffnames,"VariableNames",col_names);
for ii = 1:num_coeff
    coeff_tbl{coeffnames{ii},'Estimate'} = b(ii);
    coeff_tbl{coeffnames{ii},'tStat'} = t_statistics(ii);
    coeff_tbl{coeffnames{ii},'pValue'} = p_values(ii);
    coeff_tbl{coeffnames{ii},'SE'} = standard_error(ii);
end
mdl.CoefficientNames = coeffnames;
mdl.NumCoefficients = length(b);
mdl.Coefficients = coeff_tbl;
mdl.Deviance = deviance;
% mdl.Residuals = residuals;
mdl.LogLikelihood = logL;
mdl.DFE = DFE;
mdl.SSE = stats.SSE;
mdl.SST = stats.SST;
mdl.Rsquared.Ordinary = stats.Rsquared.Ordinary;
mdl.Rsquared.Adjusted = stats.Rsquared.Adjusted;
mdl.cov_b = cov_b;
% Display the results
if verbose
    disp('Fitted coefficients:');
    disp(b);
    disp('Deviance:');
    disp(deviance);
    disp('Concentration parameter (kappa):');
    disp(kappa);
    disp('Standard errors:');
    disp(standard_error);
    disp('t-statistics:');
    disp(t_statistics);
    disp('p-values:');
    disp(p_values);
end
end

function kappa = circ_kappa(alpha, w)
%KAPPA  Approximate the ML estimate of the concentration parameter kappa
%
%   Usage:
%       kappa = circ_kappa(alpha, [w])
%
%   Input:
%       alpha: vector - angles in radians or length resultant
%       w: vector - number of incidences for binned angle data (optional)
%
%   Output:
%       kappa: double - estimated value of kappa
%
%   References:
%       Statistical analysis of circular data, Fisher, equation p. 88
%
%   Circular Statistics Toolbox for Matlab
%   By Philipp Berens, 2009

alpha = alpha(:);
if nargin < 2
    w = ones(size(alpha));
else
    if size(w, 2) > size(w, 1)
        w = w';
    end
end

N = length(alpha);
if N > 1
    R = circ_r(alpha, w);
else
    R = alpha;
end

if R < 0.53
    kappa = 2 * R + R^3 + 5 * R^5 / 6;
elseif R >= 0.53 && R < 0.85
    kappa = -.4 + 1.39 * R + 0.43 / (1 - R);
else
    kappa = 1 / (R^3 - 4 * R^2 + 3 * R);
end

if N < 15 && N > 1
    if kappa < 2
        kappa = max(kappa - 2 * (N * kappa)^-1, 0);
    else
        kappa = (N - 1)^3 * kappa / (N^3 + N);
    end
end
end

function r = circ_r(alpha, w, d, dim)
%CIRC_R  Compute mean resultant vector length for circular data
%
%   Usage:
%       r = circ_r(alpha, [w, d, dim])
%
%   Input:
%       alpha: vector - sample of angles in radians
%       w: vector - number of incidences for binned angle data (optional)
%       d: double - spacing of bin centers for binned data in radians (optional)
%       dim: integer - dimension to compute along (default: 1)
%
%   Output:
%       r: double - mean resultant length
%
%   References:
%       Statistical analysis of circular data, N.I. Fisher
%       Topics in circular statistics, S.R. Jammalamadaka et al.
%       Biostatistical Analysis, J. H. Zar
%
%   Circular Statistics Toolbox for Matlab
%   By Philipp Berens, 2009

if nargin < 4
    dim = 1;
end

if nargin < 2 || isempty(w)
    w = ones(size(alpha));
else
    if size(w, 2) ~= size(alpha, 2) || size(w, 1) ~= size(alpha, 1)
        error('Input dimensions do not match');
    end
end

if nargin < 3 || isempty(d)
    d = 0;
end

% Compute weighted sum of cos and sin of angles
r = sum(w .* exp(1i * alpha), dim);

% Obtain length
r = abs(r) ./ sum(w, dim);

% Apply correction factor for binned data
if d ~= 0
    c = d / 2 / sin(d / 2);
    r = c * r;
end
end
