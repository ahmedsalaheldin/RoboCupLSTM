# Author: Justice Amoh
# Date:   08/10/2015
# Description: Script to train Recurrent Neural Network for MNIST Digit Recognition
# Dependencies: Theano, Lasagne

from __future__ import print_function
from __future__ import division

import numpy as np
import time
import theano
import theano.tensor as T
import lasagne
import sklearn.metrics
import lasagne.layers as L
import sys
import os

#from mnist import load_dataset 
from lasagne.layers import InputLayer, LSTMLayer, ReshapeLayer, DenseLayer, GaussianNoiseLayer

def load_dataset():
    # We first define a download function, supporting both Python 2 and 3.
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve

    def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
        print("Downloading %s" % filename)
        urlretrieve(source + filename, filename)

    # We then define functions for loading MNIST images and labels.
    # For convenience, they also download the requested files if needed.
    import gzip

    def load_mnist_images(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the inputs in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        # The inputs are vectors now, we reshape them to monochrome 2D images,
        # following the shape convention: (examples, channels, rows, columns)
        data = data.reshape(-1, 1, 28, 28)
        # The inputs come as bytes, we convert them to float32 in range [0,1].
        # (Actually to range [0, 255/256], for compatibility to the version
        # provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
        return data / np.float32(256)

    def load_mnist_labels(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the labels in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        # The labels are vectors of integers now, that's exactly what we want.
        return data

    # We can now download and read the training and test set images and labels.
    X_train = load_mnist_images('train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

    # We reserve the last 10000 training examples for validation.
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    # We just return all the arrays in order, as expected in main().
    # (It doesn't matter how we do this as long as we can read them again.)
    return X_train, y_train, X_val, y_val, X_test, y_test



# Number of Units in hidden layers
L1_UNITS = 50
L2_UNITS = 100

# Training Params 
LEARNING_RATE = 0.001
N_BATCH = 100
NUM_EPOCHS = 10

# Load the dataset
print("Loading data...")
X_train, y_train, X_valid, y_valid, X_test, y_test = load_dataset()

#y_train = np.repeat(y_train,28)
print(np.shape(y_train))

print (np.shape(y_valid))
X_train = np.squeeze(X_train)
print (np.shape(X_train))
X_valid = np.squeeze(X_valid)
X_test  = np.squeeze(X_test)

num_feat    = X_train.shape[1]
seq_len     = X_train.shape[2]
num_classes = np.unique(y_train).size

# Generate sequence masks (redundant here)
mask_train = np.ones((X_train.shape[0], X_train.shape[1]))
mask_valid = np.ones((X_valid.shape[0], X_valid.shape[1]))
mask_test  = np.ones((X_test.shape[0], X_test.shape[1]))


#################
## BUILD MODEL ##
#################
tanh = lasagne.nonlinearities.tanh
relu = lasagne.nonlinearities.rectify
soft = lasagne.nonlinearities.softmax

# Network Architecture
l_in = InputLayer(shape=(None, None, num_feat))
batchsize, seqlen, _ = l_in.input_var.shape

#l_noise = GaussianNoiseLayer(l_in, sigma=0.6) 
#l_mask  = InputLayer(shape=(batchsize, seqlen))

l_rnn_1 = LSTMLayer(l_in, num_units=L1_UNITS)#, mask_input=l_mask)
l_rnn_2 = LSTMLayer(l_rnn_1, num_units=L2_UNITS)
l_shp   = ReshapeLayer(l_rnn_2,(-1, L2_UNITS))
l_out = DenseLayer(l_shp, num_units=num_classes, nonlinearity=soft)
l_out2   = ReshapeLayer(l_out, (batchsize, seqlen, num_classes)) 


# Symbols and Cost Function
target_values = T.ivector('target_output')

network_output   = L.get_output(l_out)
prediction = T.argmax(network_output, axis=1)
#seq_prediction = T.argmax(network_output, axis=2)
#network_output_seq = network_output.reshape(-1,28,10)
network_output_seq= L.get_output(l_out2)
prediction_per_image = network_output_seq[:, -1]
predicted_label = T.argmax(prediction_per_image, axis=1)


all_params = L.get_all_params(l_out2, trainable=True)

cost = lasagne.objectives.categorical_crossentropy(network_output, target_values)
cost = cost.mean()

costseq = lasagne.objectives.categorical_crossentropy(prediction_per_image, target_values)
costseq = costseq.mean()



# Compute SGD updates for training
print("Computing updates ...")
updates = lasagne.updates.rmsprop(cost, all_params, LEARNING_RATE)

# Theano functions for training and computing cost
print("Compiling functions ...")
train   = theano.function(
    [l_in.input_var, target_values], cost, updates=updates)
predict = theano.function([l_in.input_var], predicted_label)
predict2 = theano.function([l_in.input_var], prediction)
#predictseq = theano.function([l_in.input_var], seq_prediction)
compute_cost = theano.function([l_in.input_var, target_values], costseq)



##############
## TRAINING ##
##############

print("Training ...")
num_batches_train = int(np.ceil(len(X_train) / N_BATCH))
train_losses = []
valid_losses = []

for epoch in range(NUM_EPOCHS):
    now = time.time
    losses = []        

    batch_shuffle = np.random.choice(X_train.shape[0], X_train.shape[0], False)
    sequences   = X_train[batch_shuffle]
    labels      = y_train[batch_shuffle]
    #train_masks = mask_train[batch_shuffle]

    for batch in range(num_batches_train):
        batch_slice = slice(N_BATCH * batch,
                            N_BATCH * (batch + 1))
        
        X_batch = sequences[batch_slice]
        y_batch = labels[batch_slice]
	y_batch = np.repeat(y_batch,28)
        #m_batch = train_masks[batch_slice]

	#print ("#############")
	#print (np.shape(X_batch))
	#print (np.shape(y_batch))

        loss = train(X_batch, y_batch)
        losses.append(loss)
                
    train_loss = np.mean(losses)
    train_losses.append(train_loss)
    
    valid_loss = compute_cost(X_valid, y_valid)
    valid_losses.append(valid_loss)

    y_pred   = predict(X_valid)
    accuracy = sklearn.metrics.accuracy_score(y_valid, y_pred)




    print("\nEpoch {0}/{1} - tloss: {2:.4} - vloss: {3:.4} - acc: {4:.4}".format(
            epoch+1, NUM_EPOCHS, train_loss, valid_loss, accuracy))


    #seq_pred   = predictseq(X_valid[1:3])
    print(predict(X_valid[1:3]))
    print(predict2(X_valid[1:3]))
    print(y_valid[1:3])



