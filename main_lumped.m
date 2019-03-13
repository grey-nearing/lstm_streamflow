%% --- Initialize Workspace --------------------------------------------------------------
clearvars; 
close all force; 
% clc;
restoredefaultpath; 
addpath(genpath(pwd));

%% --- Experiment Setup ------------------------------------------------------------------

% # k-fold splits
Nk = 5;

% maximum number of training data from each site
maxTrainLength = 15*365+4;

% validation period
Ival = 365*20+5;
maxTestLength = 5*365;

% lstm config
LSTMtrainParms.epochs               = 1e3;
LSTMtrainParms.miniBatchSize        = 438;% 73;  % this works for 15% validation rate with either 3 or 5 k-folds
LSTMtrainParms.lstmNodeSize         = 10;
LSTMtrainParms.numResponses         = 1;
LSTMtrainParms.dropoutRate          = 0.0;
LSTMtrainParms.initialLearnRate     = 0.01;
LSTMtrainParms.valFraction          = 0.15;
LSTMtrainParms.learnRateDropFactor  = 0.5;
LSTMtrainParms.learnRateDropPeriod  = 100;

% number of mutual informatio nbins
Nbins = 25;

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
[Nb,Da]    = size(data.att);
[Nt,Dm,nb] = size(data.met); assert(nb == Nb);
[nt,nb]    = size(data.obs); assert(nb == Nb); assert(nt == Nt);
[nt,nb]    = size(data.mod); assert(nb == Nb); assert(nt == Nt);

% --- standardize inputs and targets --------------------------------------

% standardize met forcings
forcings = zeros(Nt,Dm,Nb)./0;
for m = 1:Dm
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
attributes = zeros(Nb,Da)./0;
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
fprintf('There are %d basins with good data. \n',Nb);

%% --- Feature Selection -----------------------------------------------------------------

% --- manual feature selection --------------------------------------------

iuse = [6,8,10,14,17,18,31,33];
attributes = attributes(:,iuse);

% --- categorical inputs --------------------------------------------------

% % dimensions
% [Nb,Da]     = size(data.att);
% [Nt,Dm,nb]  = size(data.met); assert(nb == Nb);
% [nt,~,nb]   = size(data.obs); assert(nb == Nb); assert(nt == Nt);
% 
% % turn categorical data into one-hots - geological class
% geoClass = data.att(:,2);
% u = unique(geoClass);
% onehotGeo = zeros(Nb,max(u));
% for b = 1:Nb
%     onehotGeo(b,geoClass(b)) = 1;
% end
% onehotGeo(:,all(onehotGeo==0)) = [];
% 
% % turn categorical data into one-hots - vegetation class
% vegClass = data.att(:,36);
% u = unique(vegClass);
% onehotVeg = zeros(Nb,max(u));
% for b = 1:Nb
%     onehotVeg(b,vegClass(b)) = 1;
% end
% onehotVeg(:,all(onehotVeg==0)) = [];
% 
% % concatenate with one-hots
% attributes = [attributes];%,onehotVeg,onehotGeo];

% --- dimensions ----------------------------------------------------------
[~,Da] = size(attributes);

%% --- Train Models ----------------------------------------------------------------------

% --- prepare training/test data at each basin ----------------------------

% screen report
tic; fprintf('Extracting training and test data ...');

% init storage
trainTargets = cell(Nb,1);
trainInputs  = cell(Nb,1);

testTargets  = cell(Nb,1);
testInputs   = cell(Nb,1);

MODpreds     = cell(Nb,1);

for b = 1:Nb
    
    % --- training data ---
    % pull longest unbroken observation series
    iSeries = find_longest_series(usgsobs(:,b)',maxTrainLength);
    
    % pull trainig/test target data
    trainTargets{b} = usgsobs(iSeries,b)';
    
    % concatenate training input data
    trainInputs{b} = cat(2,forcings(iSeries,:,b),repmat(attributes(b,:),[length(iSeries),1]))';

    % --- test data ---
    % pull longest unbroken observation series
    iSeries = find_longest_series(usgsobs(Ival:end,b)',maxTestLength);
    iSeries = iSeries + Ival - 1;
    
    % pull trainig/test target data
    testTargets{b} = usgsobs(iSeries,b)';
    
    % concatenate training input data
    testInputs{b} = cat(2,forcings(iSeries,:,b),repmat(attributes(b,:),[length(iSeries),1]))';

    % pull model predictions
    MODpreds{b} = modhat(iSeries,b)';
    
end % b-loop

% screen report
fprintf('. finished; t = %f[s] \n',toc); 

% --- prepare k-fold splits -----------------------------------------------

% screen report
tic; fprintf('Preparing k-fold splits ...');

% init storage
Itest  = cell(Nk,1);
Itrain = cell(Nk,1);

% randomize data
ip = randperm(Nb);

% number of samples per split
nk = floor(Nb/Nk);
rk = rem(Nb,nk);
nk = repmat(nk,[1,Nk]);
nk(1:rk) = nk(1:rk) + 1;

% indexes of k groups
ics = [0;cumsum(nk')];
for k = 1:Nk
    Itest{k} = ip(ics(k)+1:ics(k+1));
    Itrain{k}  = ip;
    Itrain{k}(ics(k)+1:ics(k+1)) = [];
end % k-loop

% screen report
fprintf('. finished; t = %f[s] \n',toc); 

% --- train the k models --------------------------------------------------

% screen report
tic; fprintf('Training k-fold models ...');

% init storage
LSTMpreds = cell(Nb,1);
LSTMmodel = cell(Nk,1);

%%
% k-fold loop
for k = 1:Nk

    % error checking
    assert(isempty(intersect(Itest{k},Itrain{k})));
    assert(length(Itrain{k}) == Nb - nk(k));
    assert(length(Itest{k}) == nk(k));
   
    % train model
    LSTMmodel{k} = trainLSTM(trainInputs(Itrain{k}),trainTargets(Itrain{k}),LSTMtrainParms);

    % test predictions
    LSTMpreds(Itest{k}) = predict(LSTMmodel{k},testInputs(Itest{k}))';
 
end % k-loop
%%

% screen report
fprintf('. finished; t = %f[s] \n',toc); 

% --- save progress -------------------------------------------------------

% screen report
tic; fprintf('Saving progress ...');

% save only trained models
fname = './results/trained_pub_models.mat';
save(fname,'LSTMmodel');

% save only test predictions
fname = './results/test_pub_predictions.mat';
save(fname,'LSTMpreds','MODpreds','testTargets');

% screen report
fprintf('. finished; t = %f[s] \n',toc); 

%% --- Calculate Statistics --------------------------------------------------------------

% screen report
tic; fprintf('Calculating Statistics ...');

% NSE
lstmNSE = cellfun(@nse,testTargets,LSTMpreds);
modNSE  = cellfun(@nse,testTargets,MODpreds);

% NSE
lstmMI = cellfun(@mutual_info,testTargets,LSTMpreds,repmat({Nbins},Nb,1));
modMI  = cellfun(@mutual_info,testTargets,MODpreds,repmat({Nbins},Nb,1));

% screen report
fprintf('. finished; t = %f[s] \n',toc); 

%% --- Plot Results ----------------------------------------------------------------------

% --- view one of the trained networks ------------------------------------
% analyzeNetwork(LSTMmodel{1}.Layers)

% --- grab colors ---------------------------------------------------------
close all
figure(1)
hold on
h = plot(randn(7)); 
colors = zeros(7,3)./0;
for i = 1:7
    colors(i,:) = h(i).Color;
end
close(1);

% --- plot model comparison -----------------------------------------------

% plot stats distributions
fig = 1; figure(fig); close(fig); figure(fig);
set(gcf,'color','w');
set(gcf,'position',[ 1640         845        1322         660]);

% --- nash sutcliffe efficiency ---
subplot(2,3,1:2);
bins = linspace(-1,1,25);
lstmHist = lstmNSE; lstmHist(lstmHist<-1) = -0.99;
modHist  = modNSE;  modHist(modHist<-1)   = -0.99;
h(1) = histogram(modHist,bins); hold on
h(2) = histogram(lstmHist,bins); hold on

h(1).FaceColor = colors(5,:);
h(2).FaceColor = colors(4,:);

set(gca,'xlim',[-1.04,1.04]);
set(gca,'fontsize',12);
grid on;
ylabel('Frequency','fontsize',20);
xlabel('Nash-Sutcliffe Efficiency','fontsize',20);
l = legend('SAC-SMA','LSTM','location','nw');
l.FontSize = 14;
l.Color = 0.9 * ones(3,1);
titstr = sprintf('Model Performance over %d CAMELS Catchments',Nb);
title(titstr,'fontsize',24)

clear textstr
textstr{1} = sprintf('Percentile   SAC     LSTM');
pct = [5,25,50,75,95];
for p = 1:length(pct)
    textstr{p+1} = sprintf('    %2dth        %1.2f      %1.2f',...
        pct(p),percentile(modNSE,pct(p)),percentile(lstmNSE,pct(p)));
end
t = annotation('textbox');
t.FontSize = 14;
t.String = textstr;
t.BackgroundColor = 0.9 * ones(3,1);
t.Position = [1.37e-01   6.8e-01   1.31e-01   1.69e-01];

% --- nse scatter ---
subplot(2,3,3);
plot([-10,1],[-10,1],'k--','linewidth',2); hold on;
plot(modNSE,lstmNSE,'o','color',colors(1,:));
grid on;
set(gca,'xlim',[-0.5,1],'ylim',[-0.5,1],'fontsize',12);
xlabel('SAC-SMA NSE','fontsize',20);
ylabel('LSTM NSE','fontsize',20);

% --- mutual info ---
subplot(2,3,4:5);
bins = linspace(0,max([lstmMI;modMI])*1.1,25);
lstmHist = lstmMI; lstmHist(lstmHist<-1) = -0.99;
modHist  = modMI;  modHist(modHist<-1)   = -0.99;
h(1) = histogram(modHist,bins); hold on
h(2) = histogram(lstmHist,bins); hold on

h(1).FaceColor = colors(5,:);
h(2).FaceColor = colors(4,:);

set(gca,'xlim',[-0.2,0.9]);
set(gca,'fontsize',12);
grid on;
ylabel('Frequency','fontsize',20);
xlabel('Mutual Information Ratio','fontsize',20);
l = legend('SAC-SMA','LSTM','location','nw');
l.FontSize = 14;
l.Color = 0.9 * ones(3,1);

clear textstr
textstr{1} = sprintf('Percentile   SAC     LSTM');
pct = [5,25,50,75,95];
for p = 1:length(pct)
    textstr{p+1} = sprintf('    %2dth        %1.2f      %1.2f',...
        pct(p),percentile(modMI,pct(p)),percentile(lstmMI,pct(p)));
end
t = annotation('textbox');
t.FontSize = 14;
t.BackgroundColor = 0.9 * ones(3,1);
t.Position = [1.37e-01   2.05e-01   1.31e-01   1.69e-01];
t.String = textstr;

% --- mi scatter ---
subplot(2,3,6);
plot([-10,1],[-10,1],'k--','linewidth',2); hold on;
plot(modMI,lstmMI,'o','color',colors(1,:));
grid on;
set(gca,'xlim',[0,1],'ylim',[0,1],'fontsize',12);
xlabel('SAC-SMA Mutual Info Ratio','fontsize',20);
ylabel('LSTM Mutual Info Ratio','fontsize',20);

% --- save figure ---
fname = 'figures/lstm_sac_comparison.jpg';
saveas(gcf,fname);

%% --- End Script ------------------------------------------------------------------------





