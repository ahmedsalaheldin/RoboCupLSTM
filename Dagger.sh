#!/bin/bash
#start server

for i in `seq 1 3`;
do
	python helloagent.py $((i-1)) true false MLP 
	wait
	python learnfromlog.py MLP $i 
	wait
done   


