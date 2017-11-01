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

def load_data(state_file,action_file):

    states = genfromtxt(state_file, delimiter=',',dtype=np.float32)
    actions = genfromtxt(action_file, delimiter=',',dtype=np.float32)

    states=states[:30000]
    actions=actions[:30000]
    if (np.shape(states)[1])>66:
    	states=np.delete(states,-1,1)
        print np.shape(states)
    if (np.shape(actions)[1])>6:
    	actions=np.delete(actions,-1,1)
        print np.shape(actions)
    #actions=np.delete(actions,-1,1)

    #states = states.reshape(-1, 10, 66)
    '''for i in range(25):
    	 states=np.delete(states,i,0)
    	 actions=np.delete(actions,i,0)'''

    print type(states)
    print type(actions)
    print np.shape(states)
    print np.shape(actions)
    #print states[1]
    print actions[1]	

    #shuffled data
    #idx = np.random.permutation(len(states))
    #states,actions=states[idx],actions[idx]

    return states,actions

def build_lstm(input_var=None):

    l_in = lasagne.layers.InputLayer(shape=(None, 66),
                                     input_var=input_var)


    # Add a fully-connected layer of 800 units
    '''l_hid1 = lasagne.layers.LSTMLayer(
            l_in, num_units=100,
            nonlinearity=lasagne.nonlinearities.rectify)'''

    l_hid1 = lasagne.layers.DenseLayer(
            l_in, num_units=100,
            nonlinearity=lasagne.nonlinearities.rectify)

    l_hid2 = lasagne.layers.DenseLayer(
            l_hid1, num_units=50,
            nonlinearity=lasagne.nonlinearities.rectify)

    l_shp = lasagne.layers.ReshapeLayer(l_hid2,(1,-1,50))
    #reshape weights
    '''l_shp = lasagne.layers.ReshapeLayer(l_hid1,(-1,800))
    # Finally, we'll add the fully-connected output layer, of 10 softmax units:
    l_out = lasagne.layers.DenseLayer(
            l_shp, num_units=6,
            nonlinearity=lasagne.nonlinearities.linear)'''
    l_lstm1 = lasagne.layers.LSTMLayer(
             l_shp, num_units=50,
            nonlinearity=lasagne.nonlinearities.rectify)

    l_lstm2 = lasagne.layers.LSTMLayer(
             l_lstm1, num_units=6,
            nonlinearity=lasagne.nonlinearities.linear)

    l_shp2   = lasagne.layers.ReshapeLayer(l_lstm2,(-1, 6))

    #l_out = lasagne.layers.DenseLayer(
    #        l_shp, num_units=6,
    #        nonlinearity=lasagne.nonlinearities.linear)

    # Each layer is linked to its incoming layer(s), so we only need to pass
    # the output layer to give access to a network in Lasagne:
    return l_shp2

def build_mlp(input_var=None):

    l_in = lasagne.layers.InputLayer(shape=(None, 66),
                                     input_var=input_var)

    # Apply 20% dropout to the input data:
    #l_in_drop = lasagne.layers.DropoutLayer(l_in, p=0.0)

    # Add a fully-connected layer of 800 units
    l_hid1 = lasagne.layers.DenseLayer(
            l_in, num_units=100,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())

    # We'll now add dropout of 50%:
    #l_hid1_drop = lasagne.layers.DropoutLayer(l_hid1, p=0.0)

    # Another 800-unit layer:
    l_hid2 = lasagne.layers.DenseLayer(
            l_hid1, num_units=50,
            nonlinearity=lasagne.nonlinearities.rectify)

    # 50% dropout again:
    #l_hid2_drop = lasagne.layers.DropoutLayer(l_hid2, p=0.0)

    # Finally, we'll add the fully-connected output layer, of 10 softmax units:
    l_out = lasagne.layers.DenseLayer(
            l_hid2, num_units=6,
            nonlinearity=lasagne.nonlinearities.linear)

    # Each layer is linked to its incoming layer(s), so we only need to pass
    # the output layer to give access to a network in Lasagne:
    return l_out


def iterate_minibatches(inputs, targets, batchsize,sequence_length, shuffle=False):
    #assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
	    excerpty = indices[start_idx*10:start_idx*10 + batchsize*10]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
            excerpty = slice(start_idx*10, start_idx*10 + batchsize*10)
        yield inputs[excerpt], targets[excerpt]

def save_network(network, filename):
    net_file = open(filename, 'w')
    cPickle.dump(network, net_file, -1)
    net_file.close()

def main(num_epochs=1500):

    augment_file=  sys.argv[1]
    #augment_Flag = (sys.argv[2]=='true')
    agentcount = sys.argv[2]

    states, actions =load_data('E_states1.csv','E_actions1.csv')


    stateshandle1=  "".join(('aug_states',augment_file,'.csv'))
    actionshandle1=  "".join(('aug_actions',augment_file,'.csv'))
    states2, actions2 =load_data(stateshandle1,actionshandle1)
    states = np.concatenate((states,states2),axis=0)
    actions = np.concatenate((actions,actions2),axis=0)

    #shuffle data
    #idx = np.random.permutation(len(states))
    #states,actions=states[idx],actions[idx]

    '''states2, actions2 =load_data('E_states_MLP_noShuffle_aug1.csv','E_actions_MLP_noShuffle_aug1.csv')
    states = np.concatenate((states,states2),axis=0)
    actions = np.concatenate((actions,actions2),axis=0)

    states2, actions2 =load_data('E_states_MLP_noShuffle_aug2.csv','E_actions_MLP_noShuffle_aug2.csv')
    states = np.concatenate((states,states2),axis=0)
    actions = np.concatenate((actions,actions2),axis=0)'''

    print "state shpe ", np.shape(states)

    Test_states, Test_actions =load_data('E_states2.csv','E_actions2.csv')
    normvec = np.array([1/100,1/180,1/180,1/180,1/100,1/180])

    input_var = T.fmatrix('inputs')
    #input_var = T.tensor3('inputs')
    target_var = T.fmatrix('targets')

    network = build_mlp(input_var)

    prediction = lasagne.layers.get_output(network)

    



    loss = lasagne.objectives.squared_error(prediction, target_var)
    loss = loss.mean()

    #test_acc = T.mean(T.eq(T.argmax(abs(prediction), axis=1), abs(target_var)))

    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.adam(loss, params,learning_rate=0.0001)
    #updates = lasagne.updates.rmsprop(loss, params, learning_rate=0.0001)
    #updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.01, momentum=0.9)

    train_fn = theano.function([input_var, target_var], loss, updates=updates)
    test_fn = theano.function([input_var, target_var], loss)
    predict_fn = theano.function([input_var], prediction)

    tempout= predict_fn(Test_states)
    print np.shape(tempout)
    print tempout[0]

	#TRAINING
    print "TRAINING"
    start_time = time.time()
    for epoch in range(num_epochs):
	train_err = 0
	for batch in iterate_minibatches(states, actions, 500,10, shuffle=False):
	    inputs, targets = batch
	    #print np.shape(inputs) TRY 50 MINIBATCH SIZE
	    #print np.shape(targets)
	    #inputs=inputs.reshape(-1,10,66)
	    #targets=targets.reshape(-1,10,6)
	    train_err += train_fn(inputs, targets)
	
	print "epoch ", epoch ," Training Loss = ",train_err,"    time: ",time.time() - start_time

	test_err = 0
	test_batches = 0
	'''for batch in iterate_minibatches(Test_states, Test_actions, 500,10, shuffle=False):
	inputs, targets = batch
	#inputs=inputs.reshape(-1,10,66)
	#targets=targets.reshape(-1,10,6)
	#print predict_fn(inputs)
	test_err += test_fn(inputs, targets)
	test_batches += 1'''

	test_err = test_fn(Test_states, Test_actions)
	print " Test Loss = ",test_err

	class_predictions = np.argmax(abs(predict_fn(Test_states)), axis=1)
	label = np.argmax(abs(Test_actions), axis=1)

	accuracy = sklearn.metrics.accuracy_score(label, class_predictions)
	print " Test accuracy = ",accuracy
	with open("E_results_aug2.csv",'a') as csvfile:
		spamwriter = csv.writer(csvfile,delimiter=",")
		spamwriter.writerow([epoch,train_err,time.time() - start_time,test_err,accuracy])
    	save_network(network, "".join(('E_agentAug',agentcount,'.pkl')))
	#TESTING
    
    test_err = 0
    test_batches = 0
    '''for batch in iterate_minibatches(Test_states, Test_actions, 500,10, shuffle=False):
	inputs, targets = batch
	#inputs=inputs.reshape(-1,10,66)
	#targets=targets.reshape(-1,10,6)
        #print predict_fn(inputs)
        test_err += test_fn(inputs, targets)
        test_batches += 1'''

    test_err = test_fn(Test_states, Test_actions)
    print " Test Loss = ",test_err
  
    class_predictions = np.argmax(abs(predict_fn(Test_states)), axis=1)
    label = np.argmax(abs(Test_actions), axis=1)

    accuracy = sklearn.metrics.accuracy_score(label, class_predictions)
    print " Test accuracy = ",accuracy

def test():

    handle = open('MLP.pkl', 'r')
    network = cPickle.load(handle)

    target_var = T.fmatrix('targets')
    prediction = lasagne.layers.get_output(network)
    inputlayer = lasagne.layers.get_all_layers(network)[0]
    loss = lasagne.objectives.squared_error(prediction, target_var)
    loss = loss.mean()
    test_fn = theano.function([inputlayer.input_var, target_var], loss)
    predict_fn = theano.function([inputlayer.input_var], prediction)

    Test_states, Test_actions =load_data('states_2.csv','actions_2.csv')
    test_err = 0
    test_batches = 0
    for batch in iterate_minibatches(Test_states, Test_actions, 500,10, shuffle=False):
	inputs, targets = batch
	#inputs=inputs.reshape(-1,10,66)
	#targets=targets.reshape(-1,10,6)
        #print predict_fn(inputs)
        test_err += test_fn(inputs, targets)
        test_batches += 1

    print " Test Loss = ",test_err/ test_batches


if __name__ == '__main__':
     main()
     #test()
