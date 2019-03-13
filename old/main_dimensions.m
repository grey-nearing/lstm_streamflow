%% --- Initialize Workspace --------------------------------------------------------------

% clear workspace
clearvars; 
close all force; 
clc;

% set path
restoredefaultpath; 
addpath(genpath(pwd));

% grab plotting colors
h = plot(randn(7));
plotColors = zeros(7,3)./0;
for i = 1:7
    plotColors(i,:) = h(i).Color;
end
close all;

%% --- Experiment Setup ------------------------------------------------------------------

% # random restarts
nReps = 5;

% # data splits
nKfold = 5;

% minimum series length for training
MinSeriesLength = 365*25;

% maximum number of memory states
nStates = unique(round(logspace(0.7,2.699,25)));
% nStates = round(linspace(1,250,10));
% nStates = round([1:9,logspace(1,2.4,20)])
% nStates = 1:10;
nModels = length(nStates);

% lstm config
LSTMtrainParms.epochs               = 5e2;
LSTMtrainParms.miniBatchSize        = nKfold-1; 
% LSTMtrainParms.lstmNodeSize         = 10;
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
fig = 1; figure(fig); close(fig); figHandle = figure(fig);
set(gcf,'color','w')
set(gcf,'position',[1640         825        1487         673]);

% init storage
stats.NSE   = zeros(nBasins,nModels,nKfold,nReps)./0;
stats.MI    = zeros(nBasins,nModels,nKfold,nReps)./0;

% number of basins without enough data
nBad = 0;

% train models at each basin 
for b = 1:nBasins
    
    % screen report
    tBasin = tic;
    
    % pull unbroken observation series
    iSeries = find_longest_series(usgsobs(:,b)',MinSeriesLength);
    if length(iSeries) < MinSeriesLength; nBad = nBad+1; continue; end
        
    % split series
    inputs  = cell(nKfold,1);
    targets = cell(nKfold,1);
    for k = 1:nKfold
        iK{k} = iSeries;
        targets{k}  = usgsobs(iK{k},b)';
        inputs{k}   = forcings(iK{k},:,b)';
    end
    
    % loop through model sizes
    for m = 1:nModels
        
        % loop through random restarts
        for r = 1:nReps
            
            % kfold train/test loop
            for k = 1:nKfold
                
                % screen report
                tSingle = tic;
                fprintf('-- Site = %d/%d, Model Size = %d (%d/%d), Restart = %d/%d, K-fold = %d/%d, ...', ...
                    b,nBasins,nStates(m),m,nModels,r,nReps,k,nKfold);
                
                % train/test split
                kTrain = 1:nKfold;
                kTrain(k) = [];
                
                % train model
                LSTMtrainParms.lstmNodeSize = nStates(m);
                lstmModel{b,m,k} = trainLSTM_simple(inputs(kTrain),targets(kTrain),LSTMtrainParms);
                
                % test predictions
                Yhat = predict(lstmModel{b,m,k},inputs(k))';
                stats.NSE(b,m,k,r) = nse(Yhat{1}',targets{k}');
                stats.MI(b,m,k,r)  = mutual_info(Yhat{1},targets{k}',nBins);
                
                % screen report
                fprintf('. finished; NSE = %f, Time = %f[s],  Basin Time = %f[s] \n',stats.NSE(b,m,k),toc(tSingle),toc(tBasin));
                
            end % k-loop
            
            % add to figure
            plot(nStates,squeeze(mean(mean(stats.NSE,3),4))','-o');
            grid on;
            xlabel('Number of Memory States','fontsize',20);
            ylabel('Nash-Sutcliffe Efficiency','fontsize',20);
            pause(0.1); % time to render
                       
        end % r-loop    
        
    end % m-loop
    
    % save figure
    fname = strcat('./results/trained_dimension_models_saved_figure_',num2str(b),'.jpg');
    saveas(gcf,fname);
    
    % save progress
    fname = strcat('./results/trained_dimension_models_saved_progress_',num2str(b),'.mat');
    save(fname);
    
end % b-loop

% screen report
fprintf('Finished all models: Basins Used = %d, t = %f[s] \n',nBasins-nBad,toc(tTrain)); 

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





