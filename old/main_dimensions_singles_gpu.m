%% --- Initialize Workspace --------------------------------------------------------------

hardware = 'gpu';

% clear workspace
clearvars -except hardware; 
close all force; 
clc;

% set path
restoredefaultpath; 
addpath(genpath(pwd));

% number of available GPUs
switch hardware
case 'gpu'
   nProcs = gpuDeviceCount;
case 'cpu'
   nProcs = 30;
end

% start parallel pool
delete(gcp('nocreate'));
parpool('local',nProcs);

%% --- Experiment Setup ------------------------------------------------------------------

% # random restarts
nReps = 1;

% # data splits
nKfold = 5;

% minimum series length for training
MinSeriesLength = 365*25;

% maximum number of memory states
%nStates = unique(round(logspace(0,2.699,28)));
nStates = 1:3;
nModels = length(nStates);

% lstm config
LSTMtrainParms.epochs               = 1e1;
LSTMtrainParms.miniBatchSize        = nKfold-1; 
LSTMtrainParms.numResponses         = 1;
LSTMtrainParms.dropoutRate          = 0.0;
LSTMtrainParms.initialLearnRate     = 0.01;
LSTMtrainParms.valFraction          = 1/(nKfold-1);
LSTMtrainParms.learnRateDropFactor  = 0;
LSTMtrainParms.learnRateDropPeriod  = 1e3;

% number of mutual information bins
nBins = 25;

%% --- Load & Prepare Data ---------------------------------------------------------------

% screen report
tic; fprintf('Loading and preparing data ...');

% --- load data -----------------------------------------------------------

% load the raw data file
fname = 'extracted_data_v2.mat';
load(fname);

% remove basins with missing attributes
im = find(any(isnan(data.att')));
data.att(im,:)   = [];
data.obs(:,im)   = [];
data.met(:,:,im) = [];
data.mod(:,im)   = [];

% dimensions
[nBasins,dAtt]   = size(data.att);
[nTimes,dMet,nb] = size(data.met); assert(nb == nBasins);
[nt,nb]          = size(data.obs); assert(nb == nBasins); assert(nt == nTimes);
[nt,nb]          = size(data.mod); assert(nb == nBasins); assert(nt == nTimes);

% --- standardize inputs and targets --------------------------------------

% standardize met forcings
forcings = zeros(nTimes,dMet,nBasins)./0;
for m = 1:dMet
    mData = data.met(:,m,:);
    mu = nanmean(mData(:));
    sg = nanstd(mData(:));
    if sg > 0
        forcings(:,m,:) = (data.met(:,m,:) - mu)./sg;
    else
        forcings(:,m,:) = data.met(:,m,:);
    end
end
        
% standardize attribute data
attributes = zeros(nBasins,dAtt)./0;
for a = 1:size(data.att,2)
    mu = mean(data.att(:,a));
    sg = std(data.att(:,a));
    attributes(:,a) = (data.att(:,a) - mu)./sg;
end

% transform obs streamflow
usgsobs = log(data.obs+1);

% transform modeled streamflow
modhat = log(data.mod+1);

% screen report
fprintf('. finished; t = %f[s] \n',toc); 
fprintf('There are %d basins with good data. \n',nBasins);

%% --- Prepare K-Fold Data ---------------------------------------------------------------

% screen report
tic; fprintf('Partitioning k-fold data splits ...');

% init storage
inputs = cell(nBasins,nKfold);
targets = cell(nBasins,nKfold);

% number of basins without enough data
nBad = 0;
iBad = zeros(nBasins,1);

% loop through basins to partition data
for b = 1:nBasins

    % pull unbroken observation series
    iSeries = find_longest_series(usgsobs(:,b)',MinSeriesLength);
    if length(iSeries) < MinSeriesLength; nBad = nBad+1; iBad(b) = 1; continue; end
    tSeries = length(iSeries);

    % number of samples per split
    nk = floor(tSeries/nKfold);
    rk = rem(tSeries,nk);
    nk = repmat(nk,[1,nKfold]);
    nk(1:rk) = nk(1:rk) + 1;

    % data series
    ics = [0;cumsum(nk')];
    for k = 1:nKfold
        iK = iSeries(ics(k)+1:ics(k+1));
        inputs{b,k} = forcings(iK,:,b)';
        targets{b,k} = usgsobs(iK,b)';
    end % k-loop
  
end % b-loop

% remove bad basins
inputs(find(iBad),:) = [];
targets(find(iBad),:) = [];
nBasins = size(inputs,1);

% screen report
fprintf('. finished; t = %f[s] \n',toc); 
fprintf('There are %d basins with good data. \n',nBasins);

%% --- Train Models ----------------------------------------------------------------------

% screen report
tTrain = tic; fprintf('Training models at each site ... \n');

% init prediction/model storage
Yhat        = cell(nBasins,nModels,nKfold,nReps);
lstmModel   = cell(nBasins,nModels,nKfold,nReps);

% init stats storage
statsNSE    = zeros(nBasins,nModels,nKfold,nReps)./0;
statsMI     = zeros(nBasins,nModels,nKfold,nReps)./0;

% gpu groups
nGroups = ceil(nBasins/nProcs);
gpuGroupID = mod(1:nBasins,nGroups)+1;

% loop through basin groups
for g = 1:nGroups

    % get basins in this group
    iGroup = find(gpuGroupID == g);
%    if length(iGroup) < nProcs; 
%       iGroup = cat(2,iGroup,repmat(1,[1,nProcs-length(iGroup)])); 
%    end 

    % start parallel jobs
    spmd(length(iGroup))

       % basin index
       b = iGroup(labindex);

       % screen report
       tBasin = tic;
       fprintf('--- Basin %d of %d ...',b,nBasins);

       % loop through model sizes
       for m = 1:nModels

          % kfold train/test loop
          for k = 1:nKfold

             % train/test split
             kTrain = 1:nKfold;
             kTrain(k) = [];

             % loop through random restarts
	     for r = 1:nReps

                % train model
                switch hardware
                case 'gpu'
                   iGPU = labindex;
                case 'cpu'
                   iGPU = -1;
                end
                
                compModel{m,k,r} = trainLSTM(inputs(b,kTrain),targets(b,kTrain),...
                     LSTMtrainParms,nStates(m),iGPU);

                % test predictions
                
                compY(m,k,r) = predict(compModel{m,k,r},inputs(b,k))';
                compNSE(m,k,r) = nse(compY{m,k,r}',targets{b,k}');
                compNMI(m,k,r)  = mutual_info(compY{m,k,r}',targets{b,k}',nBins);

             end % r-loop
          end % k-loop
       end % m-loop

       % screen report
       fprintf('. finished; Time = %f[s] \n',toc(tBasin));

   end % spmd partition

   % store composit`es in local arrays
   for ib = 1:length(iGroup)
      b = iGroup(ib);
      Yhat(b,:,:,:) = compY(ib);
      statsNSE(b,:,:,:) = compNSE{ib}; 
      statsNMI(b,:,:,:) = compNMI{ib}; 
      lstmModel(b,:,:,:) = compModel(ib);
   end

   clear tBasin compY compNSE compNMI r m k kTrain compModel b

end % g-loop

% screen report
fprintf('Finished training all models; Time = %f[s].\n',toc(tTrain));    

%% --- Save Progress ---------------------------------------------------------------------

% screen report
tic; fprintf('Saving progress ...');

% save only trained models
fname = './results/trained_dimension_models_gpu.mat';
save(fname,'lstmMmodel');

% save only perforance stats
fname = './results/test_dimension_stats_gpu.mat';
save(fname,'stats');

% screen report
fprintf('. finished; t = %f[s] \n',toc); 

%% --- End Script ------------------------------------------------------------------------





