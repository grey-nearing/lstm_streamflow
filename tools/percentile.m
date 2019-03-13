function v = percentile(X,p)

N = length(X);
ip = round(p*N/100);
X = sort(X);
v = X(ip);