local opt = lapp [[
Train a CNN classifier on CIFAR-10 using Asynchronous.

   --nodeIndex         (default 1)         node index
   --numNodes          (default 1)         num nodes spawned in parallel
   --batchSize         (default 32)        batch size, per node
   --numEpochs         (default inf)       Total Number of epochs
   --learningRate      (default .01)        learning rate
   --cuda                                  use cuda
   --gpu               (default 1)         which gpu to use (only when using cuda)
   --host              (default '127.0.0.1') host name of the server
   --port              (default 8080)      port number of the server
   --base              (default 2)         power of 2 base of the tree of nodes
   --clientIP          (default '127.0.0.1') host name of the client
   --server                                 Client/Server
   --tester                                 Tester
   --verbose                                Print Communication details
   --testTime          (default 100) How many updates between tests?

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


require 'colorPrint' -- Print Server and Client in colors
-- if not verbose
if not opt.verbose then
  function printServer(string) end
  function printClient(string) end
end

-- Build the Network
local ipc = require 'libipc'
local Tree = require 'ipc.Tree'
local client, server
local clientTest, serverTest
local serverBroadcast, clientBroadcast

-- initialize server

serverBroadcast = ipc.server(opt.host, opt.port)
serverBroadcast:clients(opt.numNodes, function (client) end)
serverTest = ipc.server(opt.host, opt.port + opt.numNodes + 1)
serverTest:clients(1, function (client) end)
server = {}
for i=1,opt.numNodes do
  client_port = opt.port + i
  printServer("Port #".. client_port .." for client #" .. i)
  server[i] = ipc.server(opt.host, client_port)
  server[i]:clients(1, function (client) end)
end


local AsyncEA = require 'Async-EASGD.AsyncEA'(server, serverBroadcast, client, clientBroadcast,serverTest,clientTest,opt.numNodes, 1,10, 0.2)

-- Print only in server and tester nodes!
if not (opt.tester or opt.server) then
   xlua.progress = function() end
   print = function() end
end

-- Load dataset
local data = require 'Data'(1, 1 , opt.batchSize, opt.cuda)
local getTrainingBatch = data.getTrainingBatch
local numTrainingBatches = data.numTrainingBatches
local getTestBatch = data.getTestBatch
local numTestBatches = data.numTestBatches
local classes = data.classes


local confusionMatrix = optim.ConfusionMatrix(classes)

local Model = require 'Model'
local params = Model.params
local f = Model.f
local df = Model.df
-- Cast the parameters
params = grad.util.cast(params, opt.cuda and 'cuda' or 'float')



AsyncEA.initServer(params)

local epoch = 1

-- print('Training #' .. epoch)
-- Train a neural network
if opt.numEpochs == 'inf' then
  opt.numEpochs = 1/0
end

for syncID = 1,opt.numEpochs*opt.batchSize do

  AsyncEA.syncServer(params)

  xlua.progress(syncID % opt.testTime, opt.testTime)

  if syncID % opt.testTime == 0 then -- every 100 syncs test the net
    AsyncEA.testNet()
  end

end
