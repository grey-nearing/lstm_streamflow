function [Yhat,stats,indexes] = testLSTM2(model,inputs,targets,nBins,indexes)

% handle gpu/cpu splitting
currentTask = getCurrentTask;
useGpu = currentTask.ID < 5;
iGPU = -1;
if useGpu; iGPU = currentTask.ID; end

% predict
if iGPU > 0; gpuDevice(iGPU); end
Yhat = predict(model,inputs');

% dimensions
assert(all(isequal(size(Yhat{1}'),size(targets))));
assert(length(inputs{1}') == length(targets));

% stats
stats.NSE = nse(Yhat{1}',targets);
stats.MI  = mutual_info(Yhat{1}',targets,nBins);




