function [t,f,i] = sobol3way(X)

assert(ndims(X) == 3);

% --- total variance ------------------------------------------------------
v = var(X(:));

% --- total effect indexes ------------------------------------------------
z = squeeze(mean(X,1));
t(1) = 1 - var(z(:))/v;
z = squeeze(var(X,[],1));
t2(1) = mean(z(:))/v;

z = squeeze(mean(X,2));
t(2) = 1 - var(z(:))/v;
z = squeeze(var(X,[],2));
t2(2) = mean(z(:))/v;

z = squeeze(mean(X,3));
t(3) = 1 - var(z(:))/v;
z = squeeze(var(X,[],3));
t2(3) = mean(z(:))/v;

% --- first order indexes -------------------------------------------------
z = squeeze(mean(mean(X,2),3));
f(1) = var(z(:))/v;

z = squeeze(mean(mean(X,1),3));
f(2) = var(z(:))/v;

z = squeeze(mean(mean(X,1),2));
f(3) = var(z(:))/v;

% --- interaction terms ---------------------------------------------------
z = squeeze(mean(X,3));
i(1) = var(z(:))/v - f(1) - f(2);

z = squeeze(mean(X,2));
i(2)= var(z(:))/v - f(1) - f(3);

z = squeeze(mean(X,1));
i(3)= var(z(:))/v - f(2) - f(3);


