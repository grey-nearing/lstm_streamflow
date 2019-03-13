function Iseries = find_longest_series(Y,Lmax)

% Ig = all(~isnan(Y'));
Ig = ~isnan(Y);

[M,V] = regexp(sprintf('%i',Ig),'1+','match');
L = cellfun('length',M);
[~,j] = max(L);
Iseries = V(j)-1+(1:L(j));

if nargin > 1 && length(Iseries) > Lmax
    Iseries = Iseries(1:Lmax);
end