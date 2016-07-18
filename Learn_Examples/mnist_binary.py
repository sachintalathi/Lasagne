# Copyright 2015 Matthieu Courbariaux

# This file is part of BinaryConnect.

# BinaryConnect is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# BinaryConnect is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with BinaryConnect.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import print_function

import sys
import sys
if '/Users/sachintalathi/Work/Python/Virtualenvs/DL_Lasagne/lib/python2.7/site-packages' in sys.path:
  del(sys.path[sys.path.index('/Users/sachintalathi/Work/Python/Virtualenvs/DL_Lasagne/lib/python2.7/site-packages')])
sys.path.append('/usr/local/lib/python2.7/site-packages/')
import os
import time

import numpy as np
np.random.seed(1234)  # for reproducibility

# specifying the gpu to use
# import theano.sandbox.cuda
# theano.sandbox.cuda.use('gpu1') 
import theano
import theano.tensor as T

import lasagne

import cPickle as pickle
import gzip
import batch_norm
import lasagne.binary_connect as binary_connect

# from pylearn2.datasets.mnist import MNIST
# from pylearn2.utils import serial

from collections import OrderedDict

if __name__ == "__main__":
    
    # BN parameters
    batch_size = 100
    print("batch_size = "+str(batch_size))
    # alpha is the exponential moving average factor
    alpha = .15
    print("alpha = "+str(alpha))
    epsilon = 1e-4
    print("epsilon = "+str(epsilon))
    
    # MLP parameters
    num_units = 100
    print("num_units = "+str(num_units))
    n_hidden_layers = 1
    print("n_hidden_layers = "+str(n_hidden_layers))
    
    # Training parameters
    num_epochs = 1
    print("num_epochs = "+str(num_epochs))
    
    # Dropout parameters
    dropout_in = 0. # 0. means no dropout
    print("dropout_in = "+str(dropout_in))
    dropout_hidden = 0.
    print("dropout_hidden = "+str(dropout_hidden))
    
    # BinaryConnect
    binary = True
    print("binary = "+str(binary))
    stochastic = True
    print("stochastic = "+str(stochastic))
    # (-H,+H) are the two binary values
    # H = "Glorot"
    H = 1.
    print("H = "+str(H))
    # W_LR_scale = 1.    
    W_LR_scale = "Glorot" # "Glorot" means we are using the coefficients from Glorot's paper
    print("W_LR_scale = "+str(W_LR_scale))
    
    # Decaying LR 
    LR_start = .001
    print("LR_start = "+str(LR_start))
    LR_fin = 0.000003
    print("LR_fin = "+str(LR_fin))
    LR_decay = (LR_fin/LR_start)**(1./num_epochs)
    print("LR_decay = "+str(LR_decay))
    # BTW, LR decay might good for the BN moving average...
    
    print('Loading MNIST dataset...')
    
    # train_set = MNIST(which_set= 'train', start=0, stop = 50000, center = True)
#     valid_set = MNIST(which_set= 'train', start=50000, stop = 60000, center = True)
#     test_set = MNIST(which_set= 'test', center = True)
#
    ####### Read data and Generate Data Batch Generator ############
    class data_set(object):
      def __init__(self,X,y):
        self.X=X
        self.y=y
    data_dir='/Users/sachintalathi/Work/Python/Data'
    trs,vrs,ts=pickle.load(gzip.open('%s/mnist.pkl.gz'%data_dir))
    
###############Data for Hinge Loss Analysis#################
    # train_set=data_set(trs[0].reshape(-1, 1, 28, 28),np.hstack(trs[1]))
#     valid_set=data_set(vrs[0].reshape(-1, 1, 28, 28),np.hstack(vrs[1]))
#     test_set=data_set(ts[0].reshape(-1, 1, 28, 28),np.hstack(ts[1]))
#
#     # Onehot the targets
#     train_set.y = np.float32(np.eye(10)[train_set.y])
#     valid_set.y = np.float32(np.eye(10)[valid_set.y])
#     test_set.y = np.float32(np.eye(10)[test_set.y])

#     # for hinge loss
#     train_set.y = 2* train_set.y - 1.
#     valid_set.y = 2* valid_set.y - 1.
#     test_set.y = 2* test_set.y - 1.
  
##########Data for Categorical entropy loss########
    train_set=data_set(trs[0].reshape(-1, 1, 28, 28),trs[1])
    valid_set=data_set(vrs[0].reshape(-1, 1, 28, 28),vrs[1])
    test_set=data_set(ts[0].reshape(-1, 1, 28, 28),ts[1])

    print('Building the MLP...') 
    
    # Prepare Theano variables for inputs and targets
    input = T.tensor4('inputs')
    
    ## theano variable for hinge loss
    #target = T.matrix('targets')
    
    ## theano variable for categorical loss
    target=T.ivector('targets')
    
    LR = T.scalar('LR', dtype=theano.config.floatX)

    mlp = lasagne.layers.InputLayer(
            shape=(None, 1, 28, 28),
            input_var=input)

    mlp = lasagne.layers.DropoutLayer(
            mlp, 
            p=dropout_in)
    
    for k in range(n_hidden_layers):

        mlp = binary_connect.DenseLayer(
                mlp, 
                binary=binary,
                stochastic=stochastic,
                H=H,
                nonlinearity=lasagne.nonlinearities.identity,
                num_units=num_units)                  
        
        mlp = batch_norm.BatchNormLayer(
                mlp,
                epsilon=epsilon, 
                alpha=alpha,
                nonlinearity=lasagne.nonlinearities.rectify)
                
        mlp = lasagne.layers.DropoutLayer(
                mlp, 
                p=dropout_hidden)
    
    mlp = binary_connect.DenseLayer(
                mlp, 
                binary=binary,
                stochastic=stochastic,
                H=H,
                nonlinearity=lasagne.nonlinearities.identity,
                num_units=10)      
                  
    mlp = batch_norm.BatchNormLayer(
            mlp,
            epsilon=epsilon, 
            alpha=alpha,
            nonlinearity=lasagne.nonlinearities.identity)

    train_output = lasagne.layers.get_output(mlp, deterministic=False)
    
    # squared hinge loss
    #loss = T.mean(T.sqr(T.maximum(0.,1.-target*train_output)))
    
    #cross entropy loss
    loss=T.mean(lasagne.objectives.categorical_crossentropy(train_output,target))
    
    if binary:
        
        # W updates
        W = lasagne.layers.get_all_params(mlp, binary=True)
        W_grads = binary_connect.compute_grads(loss,mlp)
        updates = lasagne.updates.sgd(loss_or_grads=W_grads, params=W, learning_rate=LR)
        updates = binary_connect.clipping_scaling(updates,mlp)
        
        # other parameters updates
        params = lasagne.layers.get_all_params(mlp, trainable=True, binary=False)
        updates = OrderedDict(updates.items() + lasagne.updates.sgd(loss_or_grads=loss, params=params, learning_rate=LR).items())
        
    else:
        params = lasagne.layers.get_all_params(mlp, trainable=True)
        updates = lasagne.updates.adam(loss_or_grads=loss, params=params, learning_rate=LR)

    test_output = lasagne.layers.get_output(mlp, deterministic=True)
    
    #Test Categorical Loss
    test_loss=T.mean(lasagne.objectives.categorical_crossentropy(test_output,target))
    test_err = T.mean(T.neq(T.argmax(test_output,axis=1), target),dtype=theano.config.floatX)
    
    #Test Hinge Loss
    #test_loss = T.mean(T.sqr(T.maximum(0.,1.-target*test_output)))
    #test_err = T.mean(T.neq(T.argmax(test_output,axis=1), T.argmax(target, axis=1)),dtype=theano.config.floatX)
    
    
    # Compile a function performing a training step on a mini-batch (by giving the updates dictionary) 
    # and returning the corresponding training loss:
    train_fn = theano.function([input, target, LR], loss, updates=updates,allow_input_downcast=True)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input, target], [test_loss, test_err],allow_input_downcast=True)

    print('Training...')
    
    binary_connect.train(
            train_fn,val_fn,
            batch_size,
            LR_start,LR_decay,
            num_epochs,
            train_set.X,train_set.y,
            valid_set.X,valid_set.y,
            test_set.X,test_set.y)
    
    # print("display histogram")
    
    # W = lasagne.layers.get_all_layers(mlp)[2].W.get_value()
    # print(W.shape)
    
    # histogram = np.histogram(W,bins=1000,range=(-1.1,1.1))
    # np.savetxt(str(dropout_hidden)+str(binary)+str(stochastic)+str(H)+"_hist0.csv", histogram[0], delimiter=",")
    # np.savetxt(str(dropout_hidden)+str(binary)+str(stochastic)+str(H)+"_hist1.csv", histogram[1], delimiter=",")
    
    # Optionally, you could now dump the network weights to a file like this:
    # np.savez('model.npz', lasagne.layers.get_all_param_values(network))