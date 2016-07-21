from __future__ import print_function

import sys
if '/Users/sachintalathi/Work/Python/Virtualenvs/DL_Lasagne/lib/python2.7/site-packages' in sys.path:
  del(sys.path[sys.path.index('/Users/sachintalathi/Work/Python/Virtualenvs/DL_Lasagne/lib/python2.7/site-packages')])
sys.path.append('/usr/local/lib/python2.7/site-packages/')
import os
import time
import numpy as np
import pylab as py
import theano
import theano.tensor as T
import lasagne
from lasagne.layers import batch_norm
import cPickle as pickle
import gzip
from lasagne.layers.quantize import compute_grads,clipping_scaling,train
from collections import OrderedDict
py.ion()
np.random.seed(1234) 

def batch_gen(X,y,N):
  while True:
    idx=np.random.choice(len(y),N)
    yield X[idx].astype('float32'),y[idx].astype('float32') 

def mlp_network(input,dim=784,num_hidden_layers=1,num_hidden_nodes_per_layer=2048,dropout=0.,binary=False,H=1.):
  net={}
  net['l_in']=lasagne.layers.InputLayer((None,dim),input_var=input)
  net['l_d']=lasagne.layers.DropoutLayer(net['l_in'],p=.1)
  for i in range(num_hidden_layers):
    if i==0:
      if binary:
        net[i]=batch_norm(lasagne.layers.qDenseLayer(net['l_d'],H=H,num_units=num_hidden_nodes_per_layer,nonlinearity=lasagne.nonlinearities.rectify))     
      else:
        net[i]=batch_norm(lasagne.layers.DenseLayer(net['l_d'],num_units=num_hidden_nodes_per_layer,nonlinearity=lasagne.nonlinearities.rectify))
    else:
      if binary:
        net[2*i]=batch_norm(lasagne.layers.qDenseLayer(net[2*(i-1)],H=H,num_units=num_hidden_nodes_per_layer,nonlinearity=lasagne.nonlinearities.rectify))
      else:
        net[2*i]=batch_norm(lasagne.layers.DenseLayer(net[2*(i-1)],num_units=num_hidden_nodes_per_layer,nonlinearity=lasagne.nonlinearities.rectify))
    net[2*i+1]=lasagne.layers.DropoutLayer(net[2*i],p=dropout)
  
  if binary:
    net['l_out']=batch_norm(lasagne.layers.qDenseLayer(net[2*num_hidden_layers-1],H=H,num_units=10,nonlinearity=lasagne.nonlinearities.identity))
  else:
    net['l_out']=batch_norm(lasagne.layers.DenseLayer(net[2*num_hidden_layers-1],num_units=10,nonlinearity=lasagne.nonlinearities.identity))
  return net

if __name__ == "__main__":
    
    # BN parameters
    batch_size = 100
    num_units = 2048
    n_hidden_layers = 1
    num_epochs = 10
    dropout_in = 0. # 0. means no dropout
    dropout_hidden = 0.
    binary = True
    stochastic = True
    H = 1.
    W_LR_scale = "Glorot" # "Glorot" means we are using the coefficients from Glorot's paper
    LR_start = .001
    LR_fin = 0.000003
    LR_decay = (LR_fin/LR_start)**(1./num_epochs)
    
    print('Building the MLP...') 
    
    # Prepare Theano variables for inputs and targets
    input=T.matrix()
    target=T.ivector('targets')
    #target = T.matrix('targets')
  
    LR = T.scalar('LR', dtype=theano.config.floatX)

    ###define network
    net=mlp_network(input,binary=binary)
    mlp=net['l_out']
    train_output = lasagne.layers.get_output(mlp)
    loss=T.mean(lasagne.objectives.multiclass_hinge_loss(train_output,target))
    err=T.mean(T.neq(T.argmax(train_output,axis=1), target),dtype=theano.config.floatX)

    if binary:
        W = lasagne.layers.get_all_params(mlp, binary=True)
        W_grads = compute_grads(loss,mlp)
        updates = lasagne.updates.adam(loss_or_grads=W_grads, params=W, learning_rate=LR)
        updates = clipping_scaling(updates,mlp)
        params = lasagne.layers.get_all_params(mlp, trainable=True, binary=False)
        updates = OrderedDict(updates.items() + lasagne.updates.sgd(loss_or_grads=loss, params=params, learning_rate=LR).items())
        
    else:
        params = lasagne.layers.get_all_params(mlp, trainable=True)
        updates = lasagne.updates.adam(loss_or_grads=loss, params=params, learning_rate=LR)

    test_output = lasagne.layers.get_output(mlp, deterministic=True)
    
    #Test Hinge Loss
    test_loss=T.mean(lasagne.objectives.multiclass_hinge_loss(test_output,target))
    test_err = T.mean(T.neq(T.argmax(test_output,axis=1), target),dtype=theano.config.floatX)
 
    
    # Compile a function performing a training step on a mini-batch (by giving the updates dictionary) 
    # and returning the corresponding training loss:
    train_fn = theano.function([input, target, LR], [loss,err], updates=updates,allow_input_downcast=True)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input, target], [test_loss, test_err],allow_input_downcast=True)

   
    print("Reading Data")
    ####### Read data and Generate Data Batch Generator ############
    class data_set(object):
      def __init__(self,X,y):
        self.X=X
        self.y=y
    data_dir='/prj/neo_lv/user/stalathi/DataSets'
    trs,vrs,ts=pickle.load(gzip.open('%s/mnist.pkl.gz'%data_dir))
    
    train_set=data_set(trs[0],trs[1])
    valid_set=data_set(vrs[0],vrs[1])
    test_set=data_set(ts[0],ts[1])

    ## Generator for Training and Validation data
    N_generator={}
    N_generator['train']=batch_gen(train_set.X,train_set.y,batch_size)
    N_generator['val']=batch_gen(valid_set.X,valid_set.y,batch_size)
    N_generator['test']=batch_gen(test_set.X,test_set.y,batch_size)
    
    N_Batches={}
    N_Batches['train']=len(train_set.X)//batch_size
    N_Batches['val']=len(valid_set.X)//batch_size
    N_Batches['test']=len(test_set.X)//batch_size



    print('Training...')
    results=train(train_fn,val_fn,N_Batches,N_generator,LR_start,LR_decay,num_epochs)
        
    print("display histogram")
    values=lasagne.layers.get_all_param_values(mlp)
    val=values[0].reshape(784*2048,)
    [N,I]=np.histogram(val)
    py.figure();py.plot(I[0:-1],N)
