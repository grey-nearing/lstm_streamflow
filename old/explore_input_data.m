%% --- Initialize Workspace --------------------------------------------------------------
clear all; close all; clc;
restoredefaultpath; addpath(genpath(pwd));

%% --- Experimental Setup ----------------------------------------------------------------

Nbins = 25;

%% --- Load & Prepare Data ---------------------------------------------------------------

% load catchment attributes and hydrological signatures
attTable = readtable('./camels_chars/camels_chars.txt');
attData = attTable{:,2:end};

% load catchment hydrological signatures
sigTable = readtable('./camels_chars/camels_sigs.txt');
sigData = sigTable{:,:};

% remove basins with missing values
im = find(any(isnan([sigData,attData]')));
sigData(im,:)       = [];
attData(im,:)       = [];

% dimensions 
[Nbasins,Natts] = size(attData);
[nb,Nsigs] = size(sigData);
assert(nb == Nbasins);

%% --- Calculate Pairwise Mutual Information ---------------------------------------------

% init storage
I = zeros(Nsigs,Natts)./0;
H = zeros(Nsigs,Natts)./0;

% loop through all pairs
for s = 1:Nsigs
    for a = 1:Natts

        [I(s,a),H(s,a)] = mutual_info(sigData(:,s),attData(:,a),Nbins);

    end
end

% plot info ratios
fig = 1; figure(fig); close(fig); figure(fig);
set(gcf,'color','w');
set(gcf,'position',[2018         916        1545         400]);
imagesc(I./H);
colorbar
set(gca,'fontsize',12);


%% --- End Script ------------------------------------------------------------------------