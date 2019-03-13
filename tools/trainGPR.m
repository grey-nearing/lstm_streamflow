function [trainedModel,Yhat] = trainGPR(X,Y,varargin)

% predictors and responses
[Nx,~ ] = size(X);
[Ny,Dy] = size(Y);

% check dimensions
assert(Dy == 1);
assert(Nx == Ny);

% pull training parameters
% https://www.mathworks.com/help/stats/kernel-covariance-function-options.html?searchHighlight=KernelFunction&s_tid=doc_srchtitle
if nargin > 2
    parms = varargin{1};
    if ~isfield(parms,'kernel'); parms.kernel = 'rationalquadratic'; end
    if ~isfield(parms,'standardize'); parms.standardize = true; end
else
    parms.kernel = 'rationalquadratic'; 
    parms.standardize = true; 
end

% --- Train ---------------------------------------------------------------

% Train a regression model
regressionGP = fitrgp(array2table(X),Y, ...
    'BasisFunction', 'constant', ...
    'KernelFunction', parms.kernel, ...
    'Standardize', parms.standardize, ...
    'ConstantSigma', false);
%     'SigmaLowerBound', 0.01, ...
%     'Sigma', 0.10, ...


% Create the result struct with predict function
gpPredictFcn = @(x) predict(regressionGP, x);
trainedModel.predictFcn = @(x) gpPredictFcn(array2table(x));

% add restrict function to trained model
trainedModel.RegressionGP = regressionGP;

% --- Validation ----------------------------------------------------------

% check for validation parameter
if nargin == 4
    
    % extreact parameter
    kfold = varargin{1};
    
    % Perform cross-validation
    partitionedModel = crossval(trainedModel.RegressionGP,'KFold',kfold);
    
    % Compute validation predictions
    Yhat = kfoldPredict(partitionedModel);
    
else
    Yhat = [];
    return
end

% --- End Function --------------------------------------------------------
