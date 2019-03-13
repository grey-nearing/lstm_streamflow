function [V,L] = find_all_series(Y)

% Ig = all(~isnan(Y'));
Ig = ~isnan(Y);

[M,V] = regexp(sprintf('%i',Ig),'1+','match');
L = cellfun('length',M);
