import sys
import os
import time
import cPickle
import numpy as np
import theano
import theano.tensor as T
import sklearn.metrics
import lasagne
from numpy import genfromtxt
import csv

def load_data(action_file):

    #states = genfromtxt(state_file, delimiter=',',dtype=np.float32)
    actions = genfromtxt(action_file, delimiter=',',dtype=np.float32)
    if (np.shape(actions)[1])>6:
    	actions=np.delete(actions,-1,1)

    #actions=actions[:20000]
    '''
    #states=states[:999900]
    #actions=actions[:999900]
    if (np.shape(actions)[1])>6:
    	actions=np.delete(actions,-1,1)
        print np.shape(actions)
    #actions=np.delete(actions,-1,1)

    actions[:,0]/=100#(1/100)
    actions[:,1]/=180#(1/180)
    actions[:,2]/=180#(1/180)
    actions[:,3]/=180#(1/180)
    actions[:,4]/=100#(1/100)
    actions[:,5]/=180#(1/180)

    print type(actions)
    print np.shape(actions)
    #print states[1]
    print actions[1]	'''


    return actions

def saveActions(action):
	with open("E_LSTMactions.csv",'a') as f_handle:
	   np.savetxt(f_handle,action, delimiter=",", newline="\n")


def main():
    actions =load_data('E_LSTMactions.csv')
    #saveActions(actions)
    counter =0
    index = 0
    while counter<20000:
	rowsum = np.sum(actions[index,:])
        #print actions[index,:]
	if rowsum > 0:
	   counter +=1
	index +=1
	
    print index
   
if __name__ == '__main__':
     main()
     #test()
