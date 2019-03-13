function nse = nse(O,X)

Ix = find(isnan(X));
X(Ix) = [];
O(Ix) = [];

Io = find(isnan(O));
X(Io) = [];
O(Io) = [];

nse = mean((X-O).^2);
nse = nse / mean((O-mean(O)).^2);
nse = 1 - nse;