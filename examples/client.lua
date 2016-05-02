local opt = lapp [[
Train a CNN classifier on CIFAR-10 using Asynchronous.

   --nodeIndex         (default 1)         node index
   --numNodes          (default 1)         num nodes spawned in parallel
   --batchSize         (default 32)        batch size, per node
   --learningRate      (default .01)        learning rate
   --cuda                                  use cuda
   --gpu               (default 1)         which gpu to use (only when using cuda)
   --host              (default '127.0.0.1') host name of the server
   --port              (default 8080)      port number of the server
   --base              (default 2)         power of 2 base of the tree of nodes
   --clientIP          (default '127.0.0.1') host name of the client
   --verbose                               Print Communication details
   --communicationTime     (default 10) How many batches between communications?
]]

-- Requires
if opt.cuda then
   require 'cutorch'
   require 'cunn'
   cutorch.setDevice(opt.gpu)
end

local grad = require 'autograd'
local util = require 'autograd.util'
local lossFuns = require 'autograd.loss'
local optim = require 'optim'
local Dataset = require 'dataset.Dataset'
local walkTable = require 'ipc.utils'.walkTable
local tau = opt.communicationTime

require 'colorPrint' -- Print Server and Client in colors
if not opt.verbose then
  function printServer(string) end
  function printClient(string) end
end


-- Build the Network
local ipc = require 'libipc'
local Tree = require 'ipc.Tree'
local client, server
local serverBroadcast, clientBroadcast
local clientTest, serverTest

-- initialize client


clientBroadcast = ipc.client(opt.host, opt.port)
client = ipc.client(opt.host, opt.port + opt.nodeIndex)

local AsyncEA = require 'Async-EASGD.AsyncEA'(server, serverBroadcast, client, clientBroadcast, serverTest, clientTest, opt.numNodes, opt.nodeIndex,tau, 0.2)

-- Print only in instance server and tester if not on verbose mode!
if not opt.verbose then
   xlua.progress = function() end
   print = function() end
end


-- Load dataset
local data = require 'Data'(opt.nodeIndex, opt.numNodes, opt.batchSize, opt.cuda)
local getTrainingBatch = data.getTrainingBatch
local numTrainingBatches = data.numTrainingBatches
local getTestBatch = data.getTestBatch
local numTestBatches = data.numTestBatches
local classes = data.classes


local confusionMatrix = optim.ConfusionMatrix(classes)

-- for CNNs, we rely on efficient nn-provided primitives:

local Model = require 'Model'
local params = Model.params
local f = Model.f
local df = Model.df
-- Cast the parameters
params = grad.util.cast(params, opt.cuda and 'cuda' or 'float')

-- Train a neural network
AsyncEA.initClient(params)

for epoch = 1,100 do

   for i = 1,numTrainingBatches() do
      -- Next sample:
      local batch = getTrainingBatch()
      local x = batch.input
      local y = batch.target


      local grads, loss, prediction = df(params,x,y)

      -- sync with Master
      AsyncEA.syncClient(params) -- Syncs client with server if needed


      -- Update weights and biases
      for layer in pairs(params) do
         for i in pairs(params[layer]) do
            params[layer][i]:add(-opt.learningRate, grads[layer][i])
         end
      end

   end

end
