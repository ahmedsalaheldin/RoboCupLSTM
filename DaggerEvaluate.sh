#!/bin/bash
#start server

for i in `seq 1 4`;
do
	python helloagent.py $((i-1)) false false MLP 

done   


