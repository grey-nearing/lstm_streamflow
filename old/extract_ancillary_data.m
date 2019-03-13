% The paper that describes this data set is on EarthArXiv:
%   https://eartharxiv.org/2em53/

%% --- init workspace -----------------------------------------------------

clear all; close all; clc
restoredefaultpath; 
fignum = 0;

%% --- metadata setup -----------------------------------------------------

% data directories
rawDir = 'camels_chars/';  % where to find the raw data 
extDir = 'camels_chars/';  % where to put the exracted data

% raw data dimensions
N = 671;
Dh = 14;
Dc = 12;
Dg = 8;
Dt = 7;
Dv = 10;
Ds = 12;

geo_types = [	{'Siliciclastic sedimentary rocks'},...
    {'Acid plutonic rocks'},...
    {'Metamorphics'},...
    {'Carbonate sedimentary rocks'},...
    {'Basic volcanic rocks'},...
    {'Carbonate sedimentary rocks'},...
    {'Unconsolidated sediments'},...
    {'Mixed sedimentary rocks'},...
    {'Basic plutonic rocks'},...
    {'Intermediate volcanic rocks'},...
    {'Acid volcanic rocks'},...
    {'Water bodies'},...
    {'Pyroclastics'},...
    {'Intermediate plutonic rocks'}];

veg_types = [	{'Mixed Forests'},...
    {'Deciduous Broadleaf Forest'},...
    {'Cropland/Natural Vegetation Mosaic'},...
    {'Croplands'},...
    {'Grasslands'},...
    {'Savannas'},...
    {'Closed Shrublands'},...
    {'Open Shrublands'},...
    {'Barren or Sparsely Vegetated'},...
    {'Woody Savannas'},...
    {'Evergreen Needleleaf Forest'},...
    {'Evergreen Broadleaf Forest'}];

%% --- load data ----------------------------------------------------------

% screen report
tic;
fprintf('Loading data ...');

% open files
fh = fopen(strcat(rawDir,'/camels_hydro.txt'));
fc = fopen(strcat(rawDir,'/camels_clim.txt'));
fg = fopen(strcat(rawDir,'/camels_geol.txt'));
ft = fopen(strcat(rawDir,'/camels_topo.txt'));
fv = fopen(strcat(rawDir,'/camels_vege.txt'));
fs = fopen(strcat(rawDir,'/camels_soil.txt'));

% get variable names
hydr.headers = strsplit(fgets(fh),';');
clim.headers  = strsplit(fgets(fc),';');
geol.headers  = strsplit(fgets(fg),';');
topo.headers  = strsplit(fgets(ft),';');
vege.headers  = strsplit(fgets(fv),';');
soil.headers  = strsplit(fgets(fs),';');

% init storage
hydr.data = zeros(N,Dh)./0;
clim.data  = zeros(N,Dc)./0;
geol.data  = zeros(N,Dg)./0;
topo.data  = zeros(N,Dt)./0;
vege.data  = zeros(N,Dv)./0;
soil.data  = zeros(N,Ds)./0;

% read data
for n = 1:N
    
    % read hydro
    line = strsplit(fgets(fh),';');
    for i = 1:Dh
        if ~strcmpi(strtrim(line{i}),'NA')
            hydr.data(n,i) = str2num(line{i});
        end
    end
    
    % read clim
    line = strsplit(fgets(fc),';');
    for i = 1:Dc
        if ~strcmpi(strtrim(line{i}),'NA')
            if     i == 9 || i == 12 && strcmpi(strtrim(line{i}),'djf')
                clim.data(n,i) = 1;
            elseif i == 9 || i == 12 && strcmpi(strtrim(line{i}),'mam')
                clim.data(n,i) = 2;
            elseif i == 9 || i == 12 && strcmpi(strtrim(line{i}),'jja')
                clim.data(n,i) = 3;
            elseif i == 9 || i == 12 && strcmpi(strtrim(line{i}),'son')
                clim.data(n,i) = 4;
            else
                clim.data(n,i) = str2num(line{i});
            end
        end
    end
    
    % read geol
    line = strsplit(fgets(fg),';');
    for i = 1:Dg
        if ~strcmpi(strtrim(line{i}),'NA')
            try
                geol.data(n,i) = str2num(line{i});
            catch
                for g = 1:length(geo_types)
                    if strcmpi(line{i},geo_types{g})
                        geol.data(n,i) = g;
                        break;
                    end
                    if g == length(geo_types)
                        error('geo type not found: %s',line{i});
                    end
                end
            end
        end
    end
    
    % read topo
    line = strsplit(fgets(ft),';');
    for i = 1:Dt
        if ~strcmpi(strtrim(line{i}),'NA')
            topo.data(n,i) = str2num(line{i});
        end
    end
    
    % read vege
    line = strsplit(fgets(fv),';');
    for i = 1:Dv
        if ~strcmpi(strtrim(line{i}),'NA')
            try
                vege.data(n,i) = str2num(line{i});
            catch
                for v = 1:length(veg_types)
                    if strcmpi(strtrim(line{i}),veg_types{v})
                        vege.data(n,i) = v;
                        break;
                    end
                    if v == length(veg_types)
                        error('vege type not found: %s',line{i});
                    end
                end
            end
        end
    end
    
    % read soil
    line = strsplit(fgets(fs),';');
    for i = 1:Ds
        if ~strcmpi(strtrim(line{i}),'NA')
            soil.data(n,i) = str2num(line{i});
        end
    end
    
end

% make sure all basins align
assert(max(abs(hydr.data(:,1)-clim.data(:,1)))==0);
assert(max(abs(hydr.data(:,1)-geol.data(:,1)))==0);
assert(max(abs(hydr.data(:,1)-topo.data(:,1)))==0);
assert(max(abs(hydr.data(:,1)-vege.data(:,1)))==0);
assert(max(abs(hydr.data(:,1)-soil.data(:,1)))==0);

% pull only the data we want to regress against
geol.data  = geol.data(:,[1,2,3,6,7,8]);
clim.data  = clim.data(:,2:end);
topo.data  = topo.data(:,[2,3,4,6,7]);
soil.data  = soil.data(:,[2,3,6,7,8,9,10,11,12]);
vege.data  = vege.data(:,[2,3,5,7,8,9,10]);
hydr.data  = hydr.data(:,2:end);

% pull corresponding headers
geol.headers  = geol.headers([1,2,3,6,7,8]);
str = geol.headers{end};
geol.headers(end) = {str(1:end-1)};

clim.headers  = clim.headers( 2:end);
str = clim.headers{end};
clim.headers(end) = {str(1:end-1)};

topo.headers  = topo.headers([2,3,4,6,7]);
str = topo.headers{end};
topo.headers(end) = {str(1:end-1)};

soil.headers  = soil.headers([2,3,6,7,8,9,10,11,12]);
str = soil.headers{end};
soil.headers(end) = {str(1:end-1)};

vege.headers  = vege.headers([2,3,5,7,8,9,10]);
str = vege.headers{end};
vege.headers(end) = {str(1:end-1)};

hydr.headers = hydr.headers(2:end);
str = hydr.headers{end};
hydr.headers(end) = {str(1:end-1)};

% column labels
allHeaders = [geol.headers,clim.headers,topo.headers,soil.headers,vege.headers];

% aggregate inputs and targets
allData = [geol.data,clim.data,topo.data,soil.data,vege.data];

% screen report
t = toc;
fprintf('. finished - time = %f [seconds] \n',t);

%% --- save to file -------------------------------------------------------

% output data dimensions
[Nrows,Ncols] = size(allData);
assert(N == Nrows);
assert(size(allHeaders,2) == Ncols)

% file name
fname = strcat(extDir,'/camels_chars.txt');

% open file for writing
fid = fopen(fname,'w');

% print headers
for h = 1:Ncols-1
    fprintf(fid,'%s,',allHeaders{h});
end
fprintf(fid,'%s\n',allHeaders{Ncols});

% loop through lines (watersheds)
dFormat = strcat(repmat('%f,',[1,Ncols-1]),'%f\n'); % format string
for n = 1:N
    fprintf(fid,dFormat,allData(n,:));
end

% close file
fclose(fid);

%% --- save to file -------------------------------------------------------

% output data dimensions
[Nrows,Ncols] = size(hydr.data);
assert(N == Nrows);
assert(size(hydr.headers,2) == Ncols)

% file name
fname = strcat(extDir,'/camels_sigs.txt');

% open file for writing
fid = fopen(fname,'w');

% print headers
for h = 1:Ncols-1
    fprintf(fid,'%s,',hydr.headers{h});
end
fprintf(fid,'%s\n',hydr.headers{Ncols});

% loop through lines (watersheds)
dFormat = strcat(repmat('%f,',[1,Ncols-1]),'%f\n'); % format string
for n = 1:N
    fprintf(fid,dFormat,hydr.data(n,:));
end

% close file
fclose(fid);


%% --- END SCRIPT ---------------------------------------------------------




