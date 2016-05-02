#!/usr/bin/env bash

# run 4 nodes
numNodes=2

#################################################
# Try to close the ports if they are already used

if [ -z "$1" ]
  then
  port=`echo 8080`
else
  port=$1
fi


currPort=$port
numPorts=$(($numNodes + 1))
for i in `seq 0 $numPorts`;
do
  fuser -k $currPort/tcp
  # echo Kill port $currPort
  currPort=$(($currPort + 1))
done

# OPTIONAL: Uncomment if you want to close all luajit Processes running on GPU's
# kill $(nvidia-smi -g 0 | awk '$2=="Processes:" {p=1} p && $3 > 0 && $5~/luajit/ {print $3}')


#################################################

serverip=`ifconfig | awk '/inet addr/{print substr($2,6)}' | head -1`
echo Current server is located at ip: $serverip


th server.lua --server --numNodes $numNodes --numEpochs 50 --nodeIndex 0 --batchSize 128 --port $port --save testNet --host $serverip &
th tester.lua --tester --cuda --gpu 1 --numNodes $numNodes --numEpochs 50 --nodeIndex 0 --batchSize 128 --port $port --save testNet --host $serverip &

th client.lua --cuda --gpu 1 --numNodes $numNodes --nodeIndex 1 --batchSize 128 --port $port --host $serverip &
th client.lua --cuda --gpu 2 --numNodes $numNodes --nodeIndex 2 --batchSize 128 --port $port --host $serverip &
# OMP_NUM_THREADS=28 th remote_temp.lua --numNodes $numNodes --nodeIndex 3 --batchSize 128 --port $port --host $serverip &

# run on a remote client
# ssh -n -f lior@icri-lior "sh -c 'cd /home/lior/Playground/Torch/torch-distlearn/examples ; nohup /home/lior/torch/install/bin/th client.lua --cuda --gpu 1 --numNodes $numNodes --nodeIndex 3 --batchSize 128 --port $port --host $serverip > /dev/null 2>&1 &'"
# ssh -n -f ehoffer@cnn2-linux14 "sh -c 'cd /home/ehoffer/Playground/torch-distlearn/examples ; nohup /home/ehoffer/torch/install/bin/th client.lua --cuda --gpu 1 --numNodes $numNodes --nodeIndex 4 --batchSize 128 --port $port --host $serverip > /dev/null 2>&1 &'"
# ssh -n -f shai@shaiPC "sh -c 'cd /home/shai/Playground/torch-distlearn/examples ; nohup /home/shai/torch/install/bin/th client.lua --cuda --gpu 1 --numNodes $numNodes --nodeIndex 5 --batchSize 128 --port $port --host $serverip > /dev/null 2>&1 &'"


# run on a remote client-script example
# ssh -n -f lior@icri-lior "sh -c 'cd /home/lior/Playground/Torch/torch-distlearn/examples ; nohup ./remote_temp.sh $port > /dev/null 2>&1 &'"





# wait for them all
wait
