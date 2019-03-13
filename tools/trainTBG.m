function [trainedModel,Yhat] = trainTBG(X,Y,varargin)

% predictors and responses
[Nx,~ ] = size(X);
[Ny,Dy] = size(Y);

% check dimensions
assert(Dy == 1);
assert(Nx == Ny);

% pull training parameters
if nargin > 2
    parms = varargin{1};
    if ~isfield(parms,'cycles');      parms.cycles        = 30; end
    if ~isfield(parms,'MinLeafSize'); parms.MinLeafSize   = 8;  end
else
    parms.cycles        = 30; 
    parms.MinLeafSize   = 4;  
end

% --- Train ---------------------------------------------------------------

% set up individual learners
template = templateTree(...
    'MinLeafSize',parms.MinLeafSize);

% train enseble of learners
regressionEnsemble = fitrensemble(array2table(X),Y, ...
    'Method','Bag', ...
    'NumLearningCycles',parms.cycles, ...
    'Learners',template);

% Create the result struct with predict function
ensemblePredictFcn = @(x) predict(regressionEnsemble, x);
trainedModel.predictFcn = @(x) ensemblePredictFcn(array2table(x));

% add restrict function to trained model
trainedModel.RegressionEnsemble = regressionEnsemble;

% --- Validation ----------------------------------------------------------

% check for validation parameter
if nargin == 4
    
    % extreact parameter
    kfold = varargin{1};
    
    % Perform cross-validation
    partitionedModel = crossval(trainedModel.RegressionEnsemble,'KFold',kfold);
    
    % Compute validation predictions
    Yhat = kfoldPredict(partitionedModel);
    
else
    Yhat = [];
    return
end

% --- End Function --------------------------------------------------------