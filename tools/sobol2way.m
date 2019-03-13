function [t,f,i] = sobol2way(X)

assert(ndims(X) == 2);

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

% --- first order indexes -------------------------------------------------
z = squeeze(mean(X,2));
f(1) = var(z(:))/v;

z = squeeze(mean(X,1));
f(2) = var(z(:))/v;

% --- interaction terms ---------------------------------------------------
i(1) = 1 - f(1) - f(2);


