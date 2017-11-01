#!/bin/bash
#start server

for i in `seq 1 4`;
do
./HFO --offense-agents 1 --defense-npcs 1 --no-logging  --fullstate --frames 100000
wait
done   


