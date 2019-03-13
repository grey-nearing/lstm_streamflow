%% --- Initialize Workspace --------------------------------------------------------------
clearvars; close all force; clc;
restoredefaultpath; addpath(genpath(pwd));

%% --- Experiment Setup ------------------------------------------------------------------


%% --- Locate Data -----------------------------------------------------------------------

% screen report
tic; fprintf('Locating all data files ...');

% master dataa directory
dataDir = './basin_timeseries_v1p2_metForcing_obsFlow/basin_dataset_public_v1p2/';

% met forcing file list
metFormat = '%d %d %d %d %f %f %f %f %f %f %f';
metDir = strcat(dataDir,'basin_mean_forcing/daymet/');
filepattern = sprintf('%s/*/*.txt',metDir);
metFileList = dir(filepattern);
Nmet = length(metFileList);
Dmet = 6;

% model output file list
modFormat = '%d %d %d %d %f %f %f %f %f %f %f %f';
modDir = 'model_output_daymet/model_output/flow_timeseries/daymet/';
filepattern = sprintf('%s/*/*_model_output.txt',modDir);
modFileList = dir(filepattern);
Nmod = length(modFileList);
Dmod = 1;

% % usgs obs file list
% obsFormat = '%d %d %d %d %f %s';
% obsDir = strcat(dataDir,'usgs_streamflow/');
% filepattern = sprintf('%s/*/*.txt',obsDir);
% obsFileList = dir(filepattern);
% Nobs = length(obsFileList);
% Dobs = 1;

% catchment attributes
attFile = './camels_chars/camels_chars.txt';
attTable = readtable(attFile);
[Natt,Datt] = size(attTable);

% test dates in first model file
fid = fopen(modFileList(1).name);
fileContents = textscan(fid,modFormat,'headerlines',1);
fclose(fid);
dates(:,1) = fileContents{1};
dates(:,2) = fileContents{2};
dates(:,3) = fileContents{3};
checkSequentialDays(dates)
Ntimes = size(dates,1);

% time indexes for skill scores
Ival = (365*16):Ntimes;

% screen report
fprintf('. finished; time = %f[s] \n',toc);

%% --- Align Catchments ------------------------------------------------------------------

% screen report
tic; fprintf('Aligning catchments ...');

% init storage
metIDs = zeros(Nmet,1)./0;
modIDs = zeros(Nmod,1)./0;
% obsIDs = zeros(Nobs,1)./0;

% attribute basin IDs
attIDs = attTable{:,1};

% gather met basin ids
for f = 1:length(metFileList)
    textstr = strsplit(metFileList(f).name,'_');
    metIDs(f) = str2num(textstr{1});
end % f-loop

% gather model basin ids
for f = 1:length(modFileList)
    textstr = strsplit(modFileList(f).name,'_');
    modIDs(f) = str2num(textstr{1});
end % f-loop

% multiple calibrations in each catchment
[uMod,~,iMod] = unique(modIDs);         % find unique catchments
Nmod = length(uMod);                    % number of unique catchments
[~,Ncal] = mode(iMod);                  % maximum number of files per catchment
modIDsCals = zeros(Nmod,Ncal)./0;       % init storage of file nmbers per basin id
for u = 1:length(uMod)                  % find all files per basin id
    iMod = find(uMod(u) == modIDs);
    modIDsCals(u,1:length(iMod)) = iMod;
end % u-loop
modIDs = uMod;                          % only store unique model basin ids
    
% % gather obs basin ids
% for f = 1:length(obsFileList)
%     textstr = strsplit(obsFileList(f).name,'_');
%     obsIDs(f) = str2num(textstr{1});
% end % f-loop

% number of basins to work with
assert(Natt == Nmod);

% screen report
fprintf('. finished; time = %f[s] \n',toc);

%% --- Extract Best Calibration Data from Each Catchment ---------------------------------

% screen report
fprintf('Extracting best calibrations ... \n');
fprintf('%s\n',repmat('.',1,max(factor(Natt))));

% init storage
obsData = zeros(Ntimes,Natt,Ncal)./0;
modData = zeros(Ntimes,Natt,Ncal)./0;
NSE = zeros(Natt,Ncal)./0;
wrongDatesFlagMod = zeros(Natt,Ncal)./0;

% loop through catchments
for b = 1:Natt
    
    % screen report
    if rem(b,Natt/max(factor(Natt))) == 0; fprintf('.'); end
    
    % find corresponding basin in model output file lists
    iMod = find(modIDs == attIDs(b)); 
    assert(iMod == b);
    
    % number of calibration files in this catchment
    Ncal = sum(~isnan(modIDsCals(b,:)));
    
    % find best calibration in catchment
    for c = 1:Ncal
        
        % read file
        fid = fopen(modFileList(modIDsCals(b,c)).name);
        fileContents = textscan(fid,modFormat,'headerlines',1);
        fclose(fid);
        
        % check dates
        clear fdates
        fdates(:,1) = fileContents{1};
        fdates(:,2) = fileContents{2};
        fdates(:,3) = fileContents{3};
        
        datesMatch = 1;
        if ~isequal(fdates,dates)
            datesMatch = 0;
        end
        
        % extract data from file
        if datesMatch
            wrongDatesFlagMod(b,c) = 0;
            modData(:,b,c) = fileContents{11}; assert(length(modData(:,b,c)) == Ntimes);
            obsData(:,b,c) = fileContents{12}; assert(length(obsData(:,b,c)) == Ntimes);
        else % ~datesMatch
            wrongDatesFlagMod(b,c) = 1;
            for t = 1:Ntimes
                iy = find(dates(t,1) == fileContents{1});
                im = find(dates(t,2) == fileContents{2}(iy));
                id = find(dates(t,3) == fileContents{3}(iy(im)));
                if ~isempty(id)
                    modData(t,b,c) = fileContents{11}(iy(im(id)));
                    obsData(t,b,c) = fileContents{12}(iy(im(id)));
                end % logical
            end % t-loop
        end % logical
        
        % remove missing values
        im = find(obsData(:,b,c) < 0);
        obsData(im,b,c) = 0/0;
        modData(im,b,c) = 0/0;
        assert(sum(modData(:,b,c) < 0) == 0);
        
        % calculate skill score
        NSE(b,c) = nse(obsData(Ival,b,c),modData(Ival,b,c));
        assert(~isnan(NSE(b,c)));
        
    end % c-loop
    
end % b-loop

% extract obsrevation data
for b = 1:Natt
    for c = 1:Ncal
        assert(all(obsData(:,b,1) == obsData(:,b,c) | isnan(obsData(:,b,c))));
    end % c-loop
end % b-loop
obsData = obsData(:,:,1);

% extract best calibration
[~,iBest] = max(NSE');
modDataBest = zeros(Ntimes,Natt)./0;
NSEbest = zeros(Natt,1)./0;
for b = 1:Natt
    modDataBest(:,b) = modData(:,b,iBest(b));
    NSEbest(b) = NSE(b,iBest(b));
end % b-loop

% screen report
fprintf('. finished; time = %f[s] \n',toc);

%% --- Extract Corresponding Met Data ----------------------------------------------------

% screen report
fprintf('Extracting met data ... \n');
fprintf('%s\n',repmat('.',1,max(factor(Natt))));

% init storage
metData = zeros(Ntimes,Dmet,Natt)./0;
wrongDatesFlagMet = zeros(Natt,1)./0;

% loop through catchments
for b = 1:Natt
    
    % screen report
    if rem(b,Natt/max(factor(Natt))) == 0; fprintf('.'); end
    
    % find corresponding basin in met file list
    iMet = find(metIDs == attIDs(b));
    
    % read file
    fid = fopen(metFileList(iMet).name);
    fileContents = textscan(fid,metFormat,'headerlines',4);
    fclose(fid);
    
    % check dates
    clear fdates
    fdates(:,1) = fileContents{1};
    fdates(:,2) = fileContents{2};
    fdates(:,3) = fileContents{3};
    
    datesMatch = 1;
    if ~isequal(fdates,dates)
        datesMatch = 0;
    end
    
    % extract data from file
    if datesMatch
        wrongDatesFlagMod(b,c) = 0;
        for d = 1:Dmet
            metData(:,d,b) = fileContents{d+5};
        end % d-loop
    else % ~datesMatch
        wrongDatesFlagMod(b,c) = 1;
        for t = 1:Ntimes
            iy = find(dates(t,1) == fileContents{1});
            im = find(dates(t,2) == fileContents{2}(iy));
            id = find(dates(t,3) == fileContents{3}(iy(im)));
            if ~isempty(id)
                for d = 1:Dmet
                    metData(t,d,b) = fileContents{d+5}(iy(im(id)));
                end % d-loop
            end % logical
        end % t-loop
    end % logical

end % b-loop

% screen report
fprintf('. finished; time = %f[s] \n',toc);

%% --- Save Results ----------------------------------------------------------------------

data.att = attTable{:,2:end};
data.obs = obsData;
data.met = metData;
data.mod = modDataBest;

fname = 'extracted_data_v2.mat';
save(fname,'data','-v7.3');

%% --- End Script ------------------------------------------------------------------------




