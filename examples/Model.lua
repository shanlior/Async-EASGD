
-- Requires


local grad = require 'autograd'
local util = require 'autograd.util'
local lossFuns = require 'autograd.loss'
local optim = require 'optim'



-- for CNNs, we rely on efficient nn-provided primitives:
local conv,params,bn,acts,pool = {},{},{},{},{}
local flatten,linear

-- Ensure same init on all nodes:
torch.manualSeed(0)

-- layer 1:
conv[1], params[1] = grad.nn.SpatialConvolutionMM(3, 64, 5,5, 1,1, 2,2)
bn[1], params[2] = grad.nn.SpatialBatchNormalization(64, 1e-3)
acts[1] = grad.nn.ReLU()
pool[1] = grad.nn.SpatialMaxPooling(2,2, 2,2)

-- layer 2:
conv[2], params[3] = grad.nn.SpatialConvolutionMM(64, 128, 5,5, 1,1, 2,2)
bn[2], params[4] = grad.nn.SpatialBatchNormalization(128, 1e-3)
acts[2] = grad.nn.ReLU()
pool[2] = grad.nn.SpatialMaxPooling(2,2, 2,2)

-- layer 3:
conv[3], params[5] = grad.nn.SpatialConvolutionMM(128, 256, 5,5, 1,1, 2,2)
bn[3], params[6] = grad.nn.SpatialBatchNormalization(256, 1e-3)
acts[3] = grad.nn.ReLU()
pool[3] = grad.nn.SpatialMaxPooling(2,2, 2,2)

-- layer 4:
conv[4], params[7] = grad.nn.SpatialConvolutionMM(256, 512, 5,5, 1,1, 2,2)
bn[4], params[8] = grad.nn.SpatialBatchNormalization(512, 1e-3)
acts[4] = grad.nn.ReLU()
pool[4] = grad.nn.SpatialMaxPooling(2,2, 2,2)

-- layer 5:
flatten = grad.nn.Reshape(512*2*2)
linear,params[9] = grad.nn.Linear(512*2*2, 10)


-- Make sure all the nodes have the same parameter values

-- Loss:
local logSoftMax = grad.nn.LogSoftMax()
local crossEntropy = grad.nn.ClassNLLCriterion()


-- Define our network
local function predict(params, input, target)
   local h = input
   local np = 1
   for i in ipairs(conv) do
      h = pool[i](acts[i](bn[i](params[np+1], conv[i](params[np], h))))
      np = np + 2
   end
   local hl = linear(params[np], flatten(h), 0.5)
   local out = logSoftMax(hl)
   return out
end

-- Define our loss function
local function f(params, input, target)
   local prediction = predict(params, input, target)
   local loss = crossEntropy(prediction, target)
   return loss, prediction
end

-- Get the gradients closure magically:
local df = grad(f, {
   optimize = true,              -- Generate fast code
   stableGradients = true,       -- Keep the gradient tensors stable so we can use CUDA IPC
})

return {
  params = params,
  f = f,
  df = df
}
