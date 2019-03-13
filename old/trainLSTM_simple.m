function net = trainLSTM_simple(X,Y,parms,lstmNodeSize,gpuID)

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
    'Plots',                'training-progress', ...
    'ExecutionEnvironment', 'gpu',...
    'Verbose',              0);

    %'ExecutionEnvironment', 'cpu', ...

% train the network
gpuDevice(gpuID);
net = trainNetwork(Xtrain,Ytrain,layers,options);



