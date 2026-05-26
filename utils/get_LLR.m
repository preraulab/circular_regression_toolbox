function p_value = get_LLR(LL_1,LL_2,DFE_1,DFE_2)

% Compute the likelihood ratio test statistic
LR_statistic = -2 * (LL_1 - LL_2);

% Change in degrees of freedom: difference in number of parameters
dDOF = DFE_1 - DFE_2;

% Compute the p-value
p_value = 1 - chi2cdf(LR_statistic, dDOF);
end