function [net,mu,sg] = trainLSTM(X,Y,varargin)

%% --- prepare data -------------------------------------------------------

% dimensions
Ns = length(X);
[Dx,Nt] = size(X{1});
assert(Nt == size(Y{1},2));
assert(1  == size(Y{1},1));
assert(Ns == length(Y));

% standardize
mu = mean([X{:}],2);
sg = std( [X{:}],0,2);
for s = 1:Ns
    X{s} = (X{s} - mu) ./ sg;
end

%% -- training parameters -------------------------------------------------

% training and network parameters
if nargin > 2
    parms = varargin{1};
    if ~isfield(parms,'epochs');         parms.epochs         = 5e4;              end
    if ~isfield(parms,'verbose');        parms.verbose        = 0;                end
    if ~isfield(parms,'connectedNodes'); parms.connectedNodes = [ceil(Dx/2+1),1]; end
    if ~isfield(parms,'lstmNodes');      parms.lstmNodes      = Dx*2;             end
    if ~isfield(parms,'miniBatchSize');  parms.miniBatchSize  = Ns;               end
    if ~isfield(parms,'droupout');       parms.dropout        = 0;                end
else
    parms.epochs         = 5e2;                       
    parms.verbose        = 0;                         
    parms.connectedNodes = [ceil(Dx/2+1),1]; 
    parms.lstmNodes      = Dx*2;                        
    parms.miniBatchSize  = Ns;     
    parms.dropout        = 0;
end
    
%% --- set up network -----------------------------------------------------

% add input layer
layers = sequenceInputLayer(Dx);

% add lstm layers
for l = 1:length(parms.lstmNodes)
    layers = [layers
        lstmLayer(parms.lstmNodes(l),'OutputMode','sequence')];
end % lstm-nodes

% add connected layers
for c = 1:length(parms.connectedNodes)
    layers = [layers
        fullyConnectedLayer(parms.connectedNodes(c))];
end % lstm-nodes

% add output layer
layers = [layers
    regressionLayer];

% training options
options = trainingOptions('adam', ...
    'MaxEpochs',parms.epochs, ...
    'MiniBatchSize',parms.miniBatchSize, ...
    'InitialLearnRate',0.01, ...
    'GradientThreshold',1, ...
    'Shuffle','never', ...
    'Plots','none',... % 'training-progress',...
    'Verbose',parms.verbose);

%% --- train the network --------------------------------------------------

% train the network
net = trainNetwork(X,Y,layers,options);

%% --- end function -------------------------------------------------------