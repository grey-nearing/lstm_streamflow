function net = trainANN(Xt,Yt,varargin)

if nargin > 2
    parms = varargin{1};
    if ~isfield(parms,'verbose');     parms.verbose       = 0;                              end
    if ~isfield(parms,'nodes');       parms.nodes         = min(5,round(size(Xt,2)/2+1));   end
    if ~isfield(parms,'trainRatio');  parms.trainRatio    = 0.65;                           end
    if ~isfield(parms,'max_fail');    parms.vmax_fail     = 100;                            end
    if ~isfield(parms,'epochs');      parms.epochs        = 5e4;                            end
    if ~isfield(parms,'trainFcn');    parms.trainFcn      = 'trainscg';                     end
    if ~isfield(parms,'performFcn');  parms.vperformFcn   = 'mse';                          end
else
    parms.verbose       = 0;
    parms.nodes         = min(5,round(size(Xt,2)/2+1));
    parms.trainRatio    = 0.65;
    parms.max_fail      = 100;
    parms.epochs        = 5e4;
    parms.trainFcn      = 'trainscg';
    parms.performFcn    = 'mse';
end
    
% presume data will come in with the following format:
% rows = samples
% columns = variables
x = Xt';
t = Yt';

% Training Functions
% For a list of all training functions type: help nntrain
% 'trainlm' is usually fastest.
% 'trainbr' takes longer but may be better for challenging problems.
% 'trainscg' uses less memory. Suitable in low memory situations.

% Create a Network
% net = fitnet(parms.nodes,parms.trainFcn);
net = feedforwardnet(parms.nodes,parms.trainFcn);

% Choose Input and Output Pre/Post-Processing Functions
% For a list of all processing functions type: help nnprocess
net.input.processFcns   = {'removeconstantrows','mapminmax'};
net.output.processFcns  = {'removeconstantrows','mapminmax'};

% Allow manual data partitioning
net.divideParam.trainRatio = parms.trainRatio;
net.divideParam.valRatio = 1-parms.trainRatio;
net.divideParam.testRatio = 0;

% Set up training parameters
net.trainParam.max_fail     = parms.max_fail;
net.trainParam.epochs       = parms.epochs;
net.trainParam.showWindow   = parms.verbose;

% Choose a Performance Function
% For a list of all performance functions type: help nnperformance
net.performFcn = parms.performFcn;  % Mean Squared Error
% net.performFcn = 'crossentropy';

% Train the Network
[net,~] = train(net,x,t);
