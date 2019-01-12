Async-EASGD
=========

This Framework currently holds an implementation of [Asynchronous Elastic Averaging SGD](http://arxiv.org/abs/1412.6651)

This implementation can be modified as a framework for implementing other asynchronous distributed
algorithms for Deep Learning. This framework support heterogeneous devices.

The communication is based on a TCP handshake mechanism using [torch-ipc](https://github.com/twitter/torch-ipc) messages.



![alt tag](https://cloud.githubusercontent.com/assets/8818883/14954444/23b4e98e-107b-11e6-8407-a93647f3c2d0.png)


Install
--------


**Dependencies:**

1. [torch-dataset](https://github.com/twitter/torch-dataset)
2. [torch-ipc](https://github.com/twitter/torch-ipc)

Open a terminal and choose the wanted directory:
```
git clone https://github.com/shanlior/Async-EASGD
luarocks make
```

Or directly:
```
luarocks install https://raw.githubusercontent.com/shanlior/Async-EASGD/master/async-easgd-scm-1.rockspec
```

Server Node
------------

The server node is used for communicating with the client nodes.

The server node is holding the center node:
Its sole purpose is to send the most updated version to the clients, and retrieve the new updates.
The server also sends the center node to the tester.

An implementation of the server node can be found [here](examples/server.lua).




Client Node
------------

The client nodes are used for training the net.

Every determined number of iterations, the client node contacts the server in order to do the elastic averaging step -
and update the center node.

An implementation of the client node can be found [here](examples/client.lua).




Tester Node
------------

The tester node is used for testing the net on the dataset. Every determined number of iterations the tester node gets the center node from the server.
The tester is seperated from the server in order to be able to do fast calculations over a GPU, while the server itself works better as a CPU node.

An implementation of the tester node can be found [here](examples/tester.lua).




Example
--------

A working example can be found [here](examples/AsyncEASGD.sh).

The example is a classification task on the [Cifar-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset. The dataset is loaded using [torch-dataset](https://github.com/twitter/torch-dataset) repository.

Executing the example is done by running:

**./AsyncEASGD.sh [port]**

Note that in order to use several machines, this package and all of its dependencies must be installed on each machine.

In order to run a client node on a remote machine, use this syntax on the machine that holds the server node:

```
ssh -n -f [user]@[host] "sh -c 'cd [script dir] ; nohup /home/lior/torch/install/bin/th client.lua [params] > /dev/null 2>&1 &'"
```

or more specific example, using this client node as a third node:
```
ssh -n -f [user]@[host] "sh -c 'cd [Async-EASGD examples dir] ; nohup /home/lior/torch/install/bin/th client.lua --cuda --gpu 1 --numNodes $numNodes --nodeIndex 3 --batchSize 128 --port $port --host $serverip > /dev/null 2>&1 &'"
```


In order to run a script on a remote machine, use this syntax on the machine that holds the server node:

```
ssh -n -f [user]@[host] "sh -c 'cd [scriptDir] ; nohup ./[script] > /dev/null 2>&1 &'"
```





License
-------

Licensed under the Apache License, Version 2.0.
[See LICENSE file](LICENSE).
