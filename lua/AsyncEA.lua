
local ipc = require 'libipc'
local walkTable = require 'ipc.utils'.walkTable


local function AsyncEA(server, serverBroadcast, client, clientBroadcast, serverTest, clientTest, numNodes, node, tau, alpha)

  -- Keep track of how many steps each node does per epoch
  local step = 0

  -- Keep track of the center point (also need space for the delta)
  local center,delta,flatParam

  local currentClient
  local busyFlag = 0

  -- Clone the parameters to use a center point
  local function oneTimeInit(params)
    if not center then
      center = { }
      delta = { }
      flatParam = { }
      walkTable(params, function(param)
        table.insert(center, param:clone())
        table.insert(delta, param:clone())
        table.insert(flatParam, param)
      end)
    end
  end




  -- Helper Functions

  local function isSyncNeeded()

    step = step + 1

    if step % tau == 0 then
      return true
    end

    return false

  end


  ----------- Client Functions ------------

  local function initClient(params)
  -- initialize client parameters, let all the client share the same
  -- initial center and weights

    oneTimeInit(params)

    walkTable(center, function(valuei)
      return clientBroadcast:recv(valuei)
    end)
    local i = 1
    walkTable(params, function(param)
      param:copy(center[i])
      i = i + 1
    end)
  end



  local function clientEnterSync()
    -- A mutex like block that lets only one client sync enter the syncing
    -- process with the server

    printClient(node,"Waiting to sync")
    clientBroadcast:send({ q = "Enter?",
    clientID = node})
    assert(client:recv() == "Enter")
    printClient(node,"Entered Sync")

  end


  local function clientGetCenter(params)
    -- Ask server for the center variable and receive it

    client:send("Center?")

    walkTable(center, function(valuei)
      return client:recv(valuei)
    end)

    printClient(node,"Received center")

  end


  local function calculateUpdateDiff(params)
    -- Calculate the difference between the center node and the client
    -- and update client according to EASGD
    local i = 1
    walkTable(params, function(param)
      delta[i]:add(param, -1, center[i]):mul(alpha)
      param:add(-1, delta[i])
      i = i + 1
    end)

  end


  local function clientSendDiff(params)
    -- Send diff to server, inorder to update the server center node

  client:send("delta?")
  assert(client:recv() == "delta")
  printClient(node,"Received ack for sending delta")
  walkTable(delta, function(valuei)
    client:send(valuei)
    end)

  end

  local function syncClient(params)
    -- implements Async EA-SGD on client's end
    if isSyncNeeded() then
      clientEnterSync() -- Start communication if needed with server, stand in line
      clientGetCenter(params) -- Receive required parameters from the server
      calculateUpdateDiff(params) -- Do calculations locally
      clientSendDiff(params) -- Send updated values to the server
      return true -- if synced
    end

    return false

  end
----------- Server Functions ------------


  local function initServer(params)
    -- initialize servers parameters, let all the client share the same
    -- initial center and weights
    oneTimeInit(params)

    serverBroadcast:clients(function(client)
      walkTable(center, function(valuei)
        client:send(valuei)
      end)
    end)
  end


  local function serverEnterSync()
    -- A mutex like block that lets only one client sync enter the syncing
    -- process with the server
    printServer("Server waiting to sync")

    msg = serverBroadcast:recvAny()
    assert(msg.q == "Enter?")
    currentClient = msg.clientID
    printServer("Current client is #" .. currentClient)
    server[currentClient]:clients(1, function(client)
      client:send("Enter")
    end)


  end


  local function serverSendCenter(params)
  -- Sends the center node to the client which sent the request

    local function serverHandler(client)

      local msg = client:recv()
      assert(msg == "Center?")
      printServer("Client #" .. currentClient)
      walkTable(center, function(valuei)
        client:send(valuei)
      end)
      printServer("Server Sent Center to Client #" .. currentClient)
    end

  server[currentClient]:clients(1, serverHandler)

  end

  local function serverGetUpdateDiff(params)
    -- Sever gets the diff and update its own center node

    local function GetUpdateDiffHandler(client)

      assert(client:recv() == "delta?")
      client:send("delta")

      walkTable(delta, function(valuei)
        return client:recv(valuei)
      end)          -- update server-master node

      printServer("Received delta from client #" .. currentClient)

      local i = 1
      walkTable(center, function(param)
        param:add(delta[i])
        i = i + 1
      end)

    end

    server[currentClient]:clients(1, GetUpdateDiffHandler)

    local i = 1
    walkTable(params, function(param)
      param:copy(center[i])
      i = i + 1
    end)

  end

  local function syncServer(params)
    -- implements Async EA-SGD on server's end

    serverEnterSync() -- Enter critical section. Allow only one client connection
    serverSendCenter(params) -- Send required parameters to client
    serverGetUpdateDiff(params) -- Get parameters from client and update server

  end

  local function testNet()

    local function serverHandler(client)

      client:send("Test?")
      local msg = client:recv()
      assert(msg == "Center?")

      walkTable(center, function(valuei)
        client:send(valuei)
      end)

      msg = client:recv()
      assert(msg == "Ack")

    end

    serverTest:clients(1, serverHandler)

  end

  ----------- Tester Functions ------------
  local function initTester(params)
  -- initialize tester parameters
    oneTimeInit(params)

  end


  local function startTest(params)
    -- Ask server for the center variable and receive it

    local msg = clientTest:recv()
    assert(msg == "Test?")

    clientTest:send("Center?")

    walkTable(center, function(valuei)
      return clientTest:recv(valuei)
    end)
    local i = 1
    walkTable(params, function(param)
      param:copy(center[i])
      i = i + 1
    end)

  end

  local function finishTest()
    -- Ask server for the center variable and receive it

    clientTest:send("Ack")

  end

return {
  initServer = initServer,
  initClient = initClient,
  initTester = initTester,
  syncClient = syncClient,
  syncServer = syncServer,
  testNet = testNet,
  startTest = startTest,
  finishTest = finishTest
}
end

return AsyncEA
