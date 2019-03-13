function net = trainLSTM(X,Y,parms,lstmNodeSize,gpuID)

% dimensions
Nsequences = length(X);
[Dx,Nt] = size(X{1});
assert(Nt == size(Y{1},2));
assert(parms.numResponses  == size(Y{1},1));
assert(Nsequences == length(Y));

% separate training/validation data
Nval = round(Nsequences*parms.valFraction);
iVal = randperm(Nsequences,Nval);
iTrain = setdiff(1:Nsequences,iVal);
Xtrain = X(iTrain);
Ytrain = Y(iTrain);
Xval = X(iVal);
Yval = Y(iVal);

% sort training data by sequence length
sequenceLengths = cellfun(@(X) size(X,2), Xtrain);
[~,idx] = sort(sequenceLengths);
Xtrain = Xtrain(idx);
Ytrain = Ytrain(idx);

% build the network
layers = [ ...
    sequenceInputLayer(Dx)
    lstmLayer(lstmNodeSize,'OutputMode','sequence')
%    lstmLayer(parms.lstmNodeSize,'OutputMode','sequence')
%     fullyConnectedLayer(parms.lstmNodeSize)
    dropoutLayer(parms.dropoutRate)
    fullyConnectedLayer(parms.numResponses)
    regressionLayer];

% training options
if gpuID > 0
   options = trainingOptions('adam', ...
      'MaxEpochs',            parms.epochs, ...
      'MiniBatchSize',        parms.miniBatchSize, ...
      'InitialLearnRate',     parms.initialLearnRate, ...
      'GradientThreshold',    1, ...
      'SequenceLength',       'shortest', ...
      'Shuffle',              'every-epoch', ...'never', ...  
      'ValidationData',       {Xval,Yval}, ...
      'ValidationPatience',   10, ...
      'ValidationFrequency',  5, ...
      'LearnRateSchedule',    'piecewise', ...
      'LearnRateDropPeriod',  parms.learnRateDropPeriod, ...
      'LearnRateDropFactor',  parms.learnRateDropFactor, ...
      'Plots',                'none',...'training-progress', ...
      'ExecutionEnvironment', 'gpu',...
      'Verbose',              0);
else
   options = trainingOptions('adam', ...
      'MaxEpochs',            parms.epochs, ...
      'MiniBatchSize',        parms.miniBatchSize, ...
      'InitialLearnRate',     parms.initialLearnRate, ...
      'GradientThreshold',    1, ...
      'SequenceLength',       'shortest', ...
      'Shuffle',              'every-epoch', ...'never', ...  
      'ValidationData',       {Xval,Yval}, ...
      'ValidationPatience',   10, ...
      'ValidationFrequency',  5, ...
      'LearnRateSchedule',    'piecewise', ...
      'LearnRateDropPeriod',  parms.learnRateDropPeriod, ...
      'LearnRateDropFactor',  parms.learnRateDropFactor, ...
      'Plots',                'none',...'training-progress', ...
      'ExecutionEnvironment', 'cpu',...
      'Verbose',              0);
end

% train the network
if gpuID > 0; gpuDevice(gpuID); end
net = trainNetwork(Xtrain,Ytrain,layers,options);


