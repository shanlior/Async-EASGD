#!/usr/bin/env bash

# run 2 nodes
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

# run on a remote client
# ssh -n -f [user]@[host] "sh -c 'cd [script dir] ; nohup ./[script] $port > /dev/null 2>&1 &'"


# wait for them all
wait
