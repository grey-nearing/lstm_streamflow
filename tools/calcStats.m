function stats = calcStats(O,M,Bw)

%% --- prepare data -------------------------------------------------------

% reshape
O = O(:);
M = M(:);
assert(length(O) == length(M))

% remove gradmas
Im = find(isnan(O)); O(Im) = []; M(Im) = [];
Im = find(isnan(M)); O(Im) = []; M(Im) = [];

% number of samples
stats.Ndata = length(O);

%% --- distribution statistics --------------------------------------------

% first four moments
stats.Bias     = (mean(M)     - mean(O))     ./ abs(mean(O));      % 1st moment (bias)
stats.Variance = (var(M)      - var(O))      ./ abs(var(O));       % 2nd moment (sigma)
stats.Skewness = (skewness(M) - skewness(O)) ./ abs(skewness(O));  % 3rd moment (skew)
stats.Kurtosis = (kurtosis(M) - kurtosis(O)) ./ abs(kurtosis(O));  % 4th moment (kurtosis)

% KL-divergence
stats.KLDiv = kldiv(O,M,Bw);

%% --- correlation statistics ---------------------------------------------

% linear correlation
if length(O) >= 10
    stats.Correlation = corr(O,M);
else
    stats.Correlation = 0/0;
end

% coefficient of determination
stats.CoD = stats.Correlation.^2;

% mutual information
stats.Information = mutual_info_ratio(O,M,Bw);

%% --- error statistics ---------------------------------------------------

% root mean squared error
stats.RMSE = sqrt(mean((O-M).^2));

% root error variance
[~,~,r] = regress(O,[ones(length(M),1),M]);
stats.Sigma2 = var(r);
stats.Sigma = sqrt(var(r)/var(O));

%% --- ad hoc statistics --------------------------------------------------

% root mean squared error
stats.RMSE = sqrt(mean((O-M).^2));

%% --- reporting stats ----------------------------------------------------

stats.RealBias = (mean(M - O));       % true bias
stats.AbsBias  = (mean(abs(M - O)));  % absolute bias
stats.StdError = (std(M - O));        % error standard deviation

%% --- small sample sizes -------------------------------------------------

% is not enough data then overwrite everything with grandmas
if length(O) < 6
    
    statNames = fieldnames(stats);
    Nstats = numel(statNames);
    
    for s = 1:Nstats
        stats.(statNames{s}) = 0/0;
    end
    
end

%% --- end function -------------------------------------------------------
