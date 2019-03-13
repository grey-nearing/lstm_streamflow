%% --- Initialize Workspace --------------------------------------------------------------
% clear workspace
clearvars; 
close all force; 
clc;

% set path
restoredefaultpath; 
addpath(genpath(pwd));

% start parallel pool
nProcs = 40;
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
nStates = unique(round(logspace(0,2.699,28)));
nModels = length(nStates);

% lstm config
LSTMtrainParms.epochs               = 5e2;
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
fprintf('There are %d basins with good attributes. \n',nBasins);

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

% init storage
Yhat = cell(nBasins,nModels,nKfold,nReps);
stats = cell(nBasins,nModels,nKfold,nReps);

% train models 
for b = 1:nBasins
   for m = 1:nModels
      for k = 1:nKfold
	 for r = 1:nReps

            % train/test split
            kTrain = 1:nKfold;
            kTrain(k) = [];

            % train model
            Xtrain = inputs(b,kTrain);
            Ytrain = targets(b,kTrain);
            Xtest  = inputs(b,k);
            Ytest  = targets(b,k);
            compModel(b,m,k,r) = parfeval(gcp,@lstm,3,...
                      Xtrain,Ytrain,...
                      LSTMtrainParms,nStates(m),...
                      Xtest,Ytest,nBins,...
                      [b,m,k,r]);

         end % r-loop
      end % k-loop
   end % m-loop
end % b-loop

% screen report
fprintf('All models deployed. \n')

% Collect the results as they become available.
for cm = 1:numel(compModel)

   [~,ytemp,stemp,indexes] = fetchNext(compModel);
   b = indexes(1); m = indexes(2); k = indexes(3); r = indexes(4);
   Yhat{b,m,k,r} = ytemp;
   stats{b,m,k,r} = stemp;
%   fprintf('Harvested: Basin = %d/%d; Model = %d (%d/%d); K-fold = %d/%d; Rep = %d/%d; Time = %f[s] \n',...
%      b,nBasins,nStates(m),m,nModels,k,nKfold,r,nReps,toc(tRep(b,m,k,r)));

end

% screen report
fprintf('Finished training and harvesting all models; t = %f[s].\n',toc(tTrain));    

%% --- Save Stuff -----------------------------------------------------------------------

%% save trained models
%tic; fprintf('Saving trained models ...');
%fname = strcat('./results/dimensions_trained_models.mat');
%save(fname,'lstmModel','-v7.3');
%fprintf('. finished; t = %f[s] \n',toc); 

% save predictions
tic; fprintf('Saving test results ...');
fname = strcat('./results/dimensions_test_preds.mat');
save(fname,'Yhat','-v7.3');
fname = strcat('./results/dimensions_test_stats.mat');
save(fname,'stats','-v7.3');
fprintf('. finished; t = %f[s] \n',toc); 

%% --- End Script ------------------------------------------------------------------------





