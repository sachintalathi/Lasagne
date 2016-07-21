import time

from collections import OrderedDict

import numpy as np

# specifying the gpu to use
# import theano.sandbox.cuda
# theano.sandbox.cuda.use('gpu1') 
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from .helper import get_all_layers
import time

def hard_sigmoid(x):
    return T.clip((x+1.)/2.,0,1)

# The binarization function
def binarization(W,H,binary=True,deterministic=False,stochastic=False,srng=None):
    
    # (deterministic == True) <-> test-time <-> inference-time
    if not binary or (deterministic and stochastic):
        # print("not binary")
        Wb = W
    
    else:
        
        # [-1,1] -> [0,1]
        Wb = hard_sigmoid(W/H)
        
        # Stochastic BinaryConnect
        if stochastic:
        
            # print("stoch")
            Wb = T.cast(srng.binomial(n=1, p=Wb, size=T.shape(Wb)), theano.config.floatX)

        # Deterministic BinaryConnect (round to nearest)
        else:
            # print("det")
            Wb = T.round(Wb)
        
        # 0 or 1 -> -1 or 1
        Wb = T.cast(T.switch(Wb,H,-H), theano.config.floatX)
    
    return Wb

def Conv_H_Norm(incoming,filter_size):
  if type(filter_size)=='int':
    num_inputs = filter_size*filter_size*incoming.output_shape[1]
  else:
    num_inputs=np.prod(filter_size)*incoming.output_shape[1]
  
  H=np.float32(np.sqrt(2.0/num_inputs))
  W_scale=np.float32(1./H)
  return H,W_scale
  

def compute_grads(loss,network):
    layers = get_all_layers(network)
    grads = []
    for layer in layers:
        params = layer.get_params(binary=True)
        if params:
            assert(layer.Wb)
            grads.append(theano.grad(loss, wrt=layer.Wb))
                
    return grads

# This functions clips the weights after the parameter update
def clipping_scaling(updates,network):
    
    layers = get_all_layers(network)
    updates = OrderedDict(updates)
    
    for layer in layers:
    
        params = layer.get_params(binary=True)
        for param in params:
            assert layer.W_LR_scale
            updates[param] = param + layer.W_LR_scale*(updates[param] - param) ## Learning rate scale factor
            updates[param] = T.clip(updates[param], -layer.H,layer.H)     

    return updates


def train(train_fn,val_fn,N_Batches,N_generator,LR_start,LR_decay,num_epochs):

    def train_on_batch(train_batches,train_generator,LR):
        train_loss=0;train_err=0;
        for _ in range(train_batches):
            X,y=next(train_generator)
            loss,err=train_fn(X,y,LR)
            train_loss+=loss
            train_err+=err
        train_loss=train_loss/train_batches
        train_err=train_err/train_batches *100
        return train_loss,train_err

    def val_on_batch(val_batches,val_generator):
        val_loss=0;val_err=0;
        for _ in range(val_batches):
            X,y=next(val_generator)
            loss,err=val_fn(X,y)
            val_loss+=loss
            val_err+=err
        val_loss=val_loss/val_batches
        val_err=val_err/val_batches *100
        return val_loss,val_err

    per_epoch_performance_stats=[]
    LR=LR_start
    assert('train' in N_Batches.keys() and 'train' in N_generator)
    train_batches=N_Batches['train']; train_generator=N_generator['train']
    assert('val' in N_Batches.keys() and 'val' in N_generator)
    val_batches=N_Batches['val'];val_generator=N_generator['val']
    assert('test' in N_Batches.keys() and 'test' in N_generator)
    test_batches=N_Batches['test']; test_generator=N_generator['test']

    for epoch in range(num_epochs):
        tic=time.time()
        train_loss,train_err=train_on_batch(train_batches,train_generator,LR)
        val_loss,val_err=val_on_batch(val_batches,val_generator)
        test_loss,test_err=val_on_batch(test_batches,test_generator)
        per_epoch_performance_stats.append([epoch,train_loss,val_loss,test_loss,train_err,val_err,test_err])
        print ('Epoch %d Learning_Rate %0.04f Train (Val)  Error: %.03f%% (%.03f%%)  Time  %.03f s '%(epoch,LR,train_err,val_err,time.time()-tic))

        LR*=LR_decay
    return per_epoch_performance_stats

