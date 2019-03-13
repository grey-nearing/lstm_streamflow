%% --- Initialize Workspace --------------------------------------------------------------
clearvars; 
close all force; 
clc;
restoredefaultpath; 
addpath(genpath(pwd));

%% --- Experiment Setup ------------------------------------------------------------------

% # bootstraps
nReps = 1;

% minimum series length for training
MinSeriesLength = 365*25;

% maximum number of memory states
nStates = unique(round(logspace(0,1.7,26)));
% nStates = 1:10;
nModels = length(nStates);

% lstm config
LSTMtrainParms.epochs               = 4e2;
LSTMtrainParms.miniBatchSize        = 1; 
% LSTMtrainParms.lstmNodeSize         = 10;
LSTMtrainParms.numResponses         = 1;
LSTMtrainParms.dropoutRate          = 0.0;
LSTMtrainParms.initialLearnRate     = 0.01;
LSTMtrainParms.valFraction          = 0;
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
[nBasins,dAtt]    = size(data.att);
[nTimest,dMet,nb] = size(data.met); assert(nb == nBasins);
[nt,nb]           = size(data.obs); assert(nb == nBasins); assert(nt == nTimest);
[nt,nb]           = size(data.mod); assert(nb == nBasins); assert(nt == nTimest);

% --- standardize inputs and targets --------------------------------------

% standardize met forcings
forcings = zeros(nTimest,dMet,nBasins)./0;
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
% usgsobs = zeros(size(data.obs))./0;
% for b = 1:size(data.obs,3)
%     usgsobs(:,:,b) = data.obs(:,:,b) ./ data.att(b,20);
% end
% usgsobs = log(usgsobs+1);
usgsobs = log(data.obs+1);

% transform modeled streamflow
% modhat = zeros(size(data.mod))./0;
% for b = 1:size(data.mod,3)
%     modhat(:,b) = data.mod(:,b) ./ data.att(b,20);
% end
% modhat = log(modhat+1);
modhat = log(data.mod+1);

% screen report
fprintf('. finished; t = %f[s] \n',toc); 
fprintf('There are %d basins with good data. \n',nBasins);

%% --- Train Models ----------------------------------------------------------------------

% screen report
tTrain = tic; fprintf('Training models at each site ... \n');

% progress figure
fig = 1; figure(fig); close(fig); figure(fig);
set(gcf,'color','w')
set(gcf,'position',[1640         825        1487         673]);

% init storage
stats.NSE   = zeros(nBasins,nModels,nReps)./0;
stats.MI    = zeros(nBasins,nModels,nReps)./0;

% train models at each basin 
for b = 1:nBasins
    
    % pull unbroken observation series
    iSeries = find_longest_series(usgsobs(:,b)',MinSeriesLength);
    if length(iSeries) < MinSeriesLength; continue; end
    
    % training data
    targets = {usgsobs(iSeries,b)'};
    inputs = {forcings(iSeries,:,b)'};

    % loop through model sizes
    for m = 1:nModels

        % kfold train/test loop
        for k = 1:nReps
            
            % screen report
            tSingle = tic;
            fprintf('---- Site = %d/%d, Restart = %d/%d, Model Size = %d (%d/%d) ... ', ...
                b,nBasins,k,nReps,nStates(m),m,nModels);

            % train model
            LSTMtrainParms.lstmNodeSize = nStates(m);
            lstmModel{b,m,k} = trainLSTM_simple_noVal(inputs,targets,LSTMtrainParms);

            % test predictions
            Yhat = predict(lstmModel{b,m,k},inputs)';
            stats.NSE(b,m,k) = nse(Yhat{1}',targets{1}');
            stats.MI(b,m,k)  = mutual_info(Yhat{1},targets{1}',nBins);
            
            % screen report
            fprintf('. finished; NSE = %f, Time = %f[s] \n',stats.NSE(b,m,k),toc(tSingle));
            
        end % k-loop
        
        % add to figure
        plot(nStates,squeeze(mean(stats.NSE,3))','-o');
        grid on;
        xlabel('Number of Memory States','fontsize',20);
        ylabel('Nash-Sutcliffe Efficiency','fontsize',20);
        pause(0.1); % time to render
        
    end % m-loop

    % save figure
    fname = strcat('./results/trained_dimension_models_saved_figure_',num2str(b),'.mat');
    saveas(gcf,fname);
    
    % save progress
    fname = strcat('./results/trained_dimension_models_saved_progress_',num2str(b),'.mat');
    save(fname);
    
end % b-loop

% screen report
fprintf('Finished all models: t = %f[s] \n',toc(tTrain)); 

%% --- Save Progress ---------------------------------------------------------------------

% screen report
tic; fprintf('Saving progress ...');

% save only trained models
fname = './results/trained_dimension_models.mat';
save(fname,'lstmMmodel');

% save only perforance stats
fname = './results/test_dimension_stats.mat';
save(fname,'stats');

% screen report
fprintf('. finished; t = %f[s] \n',toc); 

%% --- End Script ------------------------------------------------------------------------





