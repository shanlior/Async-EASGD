local opt = lapp [[
Train a CNN classifier on CIFAR-10 using AllReduceSGD.

   --numNodes          (default 1)         num nodes spawned in parallel
   --batchSize         (default 32)        batch size, per node
   --numEpochs         (default inf)       Total Number of epochs
   --cuda                                  use cuda
   --gpu               (default 1)         which gpu to use (only when using cuda)
   --host              (default '127.0.0.1') host name of the server
   --port              (default 8080)      port number of the server
   --base              (default 2)         power of 2 base of the tree of nodes
   --clientIP          (default '127.0.0.1') host name of the client
   --server                                 Client/Server
   --tester                                 Tester
   --verbose                                Print Communication details
   --save              (default 'log') Save location
   --visualise         (default 1)

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

if opt.save == 'log' then
  opt.save = os.date():gsub(' ','')
end

opt.save = paths.concat('./Results', opt.save)
os.execute('mkdir -p ' .. opt.save)
local cmd = torch.CmdLine()
cmd:log(opt.save .. '/Log.txt', opt)
local netFilename = paths.concat(opt.save, 'Net')
local logFilename = paths.concat(opt.save,'ErrorRate.log')
local optStateFilename = paths.concat(opt.save,'optState')
local Log = optim.Logger(logFilename)


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

-- initialize tester

clientTest = ipc.client(opt.host, opt.port + opt.numNodes + 1)

local AsyncEA = require 'distlearn.AsyncEA'(server, serverBroadcast, client, clientBroadcast, serverTest, clientTest, opt.numNodes, 1,10, 0.2)

-- Print only in server and tester nodes!
if not (opt.tester or opt.server) then
   xlua.progress = function() end
   print = function() end
end

-- Load dataset
local data = require 'Data'(1, 1, opt.batchSize, opt.cuda)
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

AsyncEA.initTester(params)

local epoch = 1

print('Training #' .. epoch)
-- Train a neural network
if opt.numEpochs == 'inf' then
  opt.numEpochs = 1/0
end

for syncID = 1,opt.numEpochs*opt.batchSize do

  AsyncEA.startTest(params)

  -- Check Training Error
  print('\nTraining Error Trial #'..epoch .. '\n')

  for i = 1,numTrainingBatches() do
     -- Next sample:
     local batch = getTrainingBatch()
     local x = batch.input
     local y = batch.target

     -- Prediction:
     local loss, prediction = f(params,x,y)

     -- Log performance:
     for b = 1,batch.batchSize do
        confusionMatrix:add(prediction[b], y[b])
     end

     -- Display progress:
     xlua.progress(i, numTestBatches())
  end

  print(confusionMatrix)
  local ErrTrain = (1-confusionMatrix.totalValid)
  print('Training Error = ' .. ErrTrain)
  confusionMatrix:zero()

  -- Check Test Error
  print('\nTesting Error Trial #' ..epoch .. '\n')


  for i = 1,numTestBatches() do
    -- Next sample:
    local batch = getTestBatch()
    local x = batch.input
    local y = batch.target

    -- Prediction:
    local loss, prediction = f(params,x,y)

    -- Log performance:
    for b = 1,batch.batchSize do
       confusionMatrix:add(prediction[b], y[b])
    end

    -- Display progress:
    xlua.progress(i, numTestBatches())
  end

  print(confusionMatrix)
  local ErrTest = (1-confusionMatrix.totalValid)
  print('Test Error = ' .. ErrTest .. '\n')
  confusionMatrix:zero()

  Log:add{['Training Error']= ErrTrain, ['Test Error'] = ErrTest}
  if opt.visualize == 1 then
      Log:style{['Training Error'] = '-', ['Test Error'] = '-'}
      Log:plot()
  end

  epoch = epoch + 1

  AsyncEA.finishTest(params)

  print('Training #' .. epoch .. '\n')

end
