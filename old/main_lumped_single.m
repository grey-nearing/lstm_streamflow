%% --- Initialize Workspace -----------------------------------------------
clear all; close all; clc;
restoredefaultpath; addpath(genpath(pwd));

%% --- Experiment Setup ---------------------------------------------------

% # k-fold splits
Nk = 2;

% maximum number of training data from each site
maxSeriesLength = 3*365;

% lstm config
LSTMtrainParms.numHiddenUnits   = 125;
LSTMtrainParms.numResponses     = 1;
LSTMtrainParms.epochs           = 2e3;
LSTMtrainParms.verbose          = 1;

%% --- Load & Prepare Data ------------------------------------------------

% load the raw data file
fname = 'extracted_data.mat';
load(fname);

% remove basins with any missing attributes
im = find(any(isnan(data.att')));
data.att(im,:)      = [];
data.obs(:,:,im)    = [];
data.met(:,:,im)    = [];

% dimensions
[Nb,Da]     = size(data.att);
[Nt,Dm,nb]  = size(data.met); assert(nb == Nb);
[nt,~,nb]   = size(data.obs); assert(nb == Nb); assert(nt == Nt);

% turn categorical data into one-hots - geological class
geoClass = data.att(:,2);
u = unique(geoClass);
onehotGeo = zeros(Nb,max(u));
for b = 1:Nb
    onehotGeo(b,geoClass(b)) = 1;
end
onehotGeo(:,all(onehotGeo==0)) = [];

% turn categorical data into one-hots - vegetation class
vegClass = data.att(:,36);
u = unique(vegClass);
onehotVeg = zeros(Nb,max(u));
for b = 1:Nb
    onehotVeg(b,vegClass(b)) = 1;
end
onehotVeg(:,all(onehotVeg==0)) = [];

% standardize all data - met forcings
% forcings = data.met;
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
        
% standardize all data - attributes
% attributes = data.att;
attributes = zeros(Nb,Da)./0;
for a = 1:Da
    mu = mean(data.att(:,a));
    sg = std(data.att(:,a));
    attributes(:,a) = (data.att(:,a) - mu)./sg;
end

% logarithm of streamflow obs
obsMin = min(data.obs(data.obs>0));
usgsobs = squeeze(data.obs);
usgsobs(usgsobs==0) = obsMin*0.99;
usgsobs = log(usgsobs);

% remove some of the attributes manually
iremove = [1,2,3,14,22,31,35,36,37,38];
attributes(:,iremove) = [];

% concatenate with one-hots
attributes = [attributes,onehotVeg,onehotGeo];
Da = size(attributes,2);

% k-fold splits
ip = randperm(Nb);
nk = floor(Nb/Nk);
rk = rem(Nb,nk);
nk = repmat(nk,[1,Nk]);
nk(1:rk) = nk(1:rk) + 1;
ics = [0;cumsum(nk')];

for k = 1:Nk
    Itest{k} = ip(ics(k)+1:ics(k+1));
    Itrain{k}  = ip;
    Itrain{k}(ics(k)+1:ics(k+1)) = [];
end

%% --- Train Models -------------------------------------------------------

% dimensions 
numFeatures = Dm + Da;

% init storage
Ztest = zeros(size(usgsobs))./0;
NSE = zeros(Nb,1)./0;

% k-fold loop
for k = 1:Nk
    
    % pull training data
    clear Ytrain Xtrain
    assert(length(Itrain{k}) == Nb-nk(k))
    for b = 1:Nb-nk(k)
        Ytrain{b} = usgsobs(:,Itrain{k}(b))';
        Mtrain = forcings(:,:,Itrain{k}(b));
        Atrain = attributes(Itrain{k}(b),:);
        Xtrain{b} = cat(2,Mtrain,repmat(Atrain,[Nt,1]))';
        iSeriesTrain = find_longest_series(Ytrain{b}',maxSeriesLength);
        Xtrain{b} = Xtrain{b}(:,iSeriesTrain);
        Ytrain{b} = Ytrain{b}(iSeriesTrain);
    end
    
    % pull test data
    clear Ytest Xtest
    assert(length(Itest{k})==nk(k))
    for b = 1:nk(k)
        Ytest{b} = usgsobs(:,Itest{k}(b))';
        Mtest = forcings(:,:,Itrain{k}(b));
        Atest = attributes(Itrain{k}(b),:);
        Xtest{b} = cat(2,Mtest,repmat(Atest,[Nt,1]))';
        iSeriesTest{b} = find_longest_series(Ytest{b}',1e6);
        Xtest{b} = Xtest{b}(:,iSeriesTest{b});
        Ytest{b} = Ytest{b}(iSeriesTest{b});
    end
    
    % train model
%     [LSTMmodel,mu,sg] = trainLSTM(Xtrain,Ytrain,LSTMtrainParms);
    LSTMmodel{k} = trainLSTM(Xtrain,Ytrain,LSTMtrainParms);
    
    % test model
    for b = 1:nk(k)
%         Xtest{b} = (Xtest{b}-mu)./sg;
        ztemp = predict(LSTMmodel{k},Xtest{b},'MiniBatchSize',1);
        Ztest(iSeriesTest{b},Itest{k}(b)) = ztemp;
        ztemp = exp(ztemp);    ztemp(ztemp<obsMin) = 0;
        ytemp = exp(Ytest{b}); ytemp(ytemp<obsMin) = 0;
        NSE(b) = nse(ytemp,ztemp);
    end
end

%% --- Save Results -------------------------------------------------------

% save everything
fname = 'trained_models.mat';
save(fname);

%% --- Calculate Statistics -----------------------------------------------

%% --- Plot Results -------------------------------------------------------

%% --- End Script ---------------------------------------------------------