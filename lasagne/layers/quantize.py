import time
import sys
from collections import OrderedDict

import numpy as np

# specifying the gpu to use
# import theano.sandbox.cuda
# theano.sandbox.cuda.use('gpu1') 
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from .helper import get_all_layers
import pickle
import time

def prob_sigmoid(x):
  return 0.5*(x/T.max(x)+1)

def hard_sigmoid(x):
    return T.clip((x+1.)/2.,0,1)

# The binarization function
def binarization(W,H,deterministic=False,stochastic=False,srng=None,quantization=None):
    if quantization==None:
        print("no quantization")
        Wb = W
    elif quantization.upper()=='BINARY':
        Wb = hard_sigmoid(W/H)
        if stochastic and not deterministic:
            Wb = T.cast(srng.binomial(n=1, p=Wb, size=T.shape(Wb)), theano.config.floatX)
        else:
            Wb = T.round(Wb)
        Wb = T.cast(T.switch(Wb,H,-H), theano.config.floatX)
    
    elif quantization.upper()=='ROUND':
        Wb=T.round(W)
    
    elif quantization.upper()=='SCALE_ROUND':
        alpha=T.max(W)
        Wb=alpha*T.round(W/alpha)
    
    elif quantization.upper()=='SIGN':
        alpha=T.max(W)
        Wb=alpha*T.sgn(W)
    
    elif quantization.upper()=='POW':
        alpha=T.max(W)
        beta=np.array(2.0,dtype='float32')
        Wb=alpha*(W/alpha)**beta*T.sgn(W)
    
    elif quantization.upper()=='STOCM':
        gamma=np.array(.5,dtype='float32')
        gamma_inv=np.array(1./gamma,dtype='float32')
        stocW=T.max(W)*srng.uniform(low=gamma,high=gamma_inv,size=T.shape(W))
        Wb = prob_sigmoid(W)
        # # Stochastic BinaryConnect
        if stochastic and not deterministic:
            print("stoch")
            Wb = T.cast(srng.binomial(n=1, p=Wb, size=T.shape(Wb)), theano.config.floatX)
            Wb = stocW*T.cast(T.switch(Wb,1,-1), theano.config.floatX)    
            #Wb=Wb*W
        else:
            print("det")
            Wb =T.clip(W,-T.max(W),T.max(W))*srng.uniform(low=gamma,high=gamma_inv,size=T.shape(W))

        
        #print Wb.eval()
    else:
        print 'Error in specifying quantizatoin type'
        print 'Allowed Values: {Round, Binary, None}'
        sys.exit(0)
    return Wb

# This functions clips the weights after the parameter update
def clipping_scaling(updates,network,quantization):
    
    layers = get_all_layers(network)
    updates = OrderedDict(updates)
    
    for layer in layers:
    
        params = layer.get_params(quantize=True)
        for param in params:
            assert layer.W_LR_scale
            updates[param] = param + layer.W_LR_scale*(updates[param] - param) ## Learning rate scale factor
            if quantization.upper()=='BINARY' or quantization.upper()=='ROUND':
                print "Clip Range:[",-layer.H,layer.H,"]"
                updates[param] = T.clip(updates[param], -layer.H,layer.H)     
            if quantization.upper()=='SCALE_ROUND' or quantization.upper()=='SCALE_BINARY' or \
            quantization.upper()=='SIGN' or quantization.upper()=='POW':
                clip_val=T.max(layer.W)
                print "Clip Range:[",-clip_val.eval(),",",clip_val.eval(),"]"
                updates[param] = T.clip(updates[param], -clip_val,clip_val)
            if quantization.upper()=='STOCM':
                clip_val=T.max(layer.W)
                gamma=np.array(2.,dtype='float32')
                print "Clip Range:[",-clip_val.eval()/gamma,",",clip_val.eval()/gamma,"]"
                #updates[param] = T.clip(updates[param], -layer.H,layer.H)
                updates[param] = T.clip(updates[param], -clip_val,clip_val)

    return updates


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
        params = layer.get_params(quantize=True)
        if params:
            assert(layer.Wb)
            grads.append(theano.grad(loss, wrt=layer.Wb))
                
    return grads



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



def batch_train(datagen,N_train_batches,mini_batch_size,f_train,f_val,lr_start,lr_decay,img_dim=(3,32,32),\
    epochs=10,test_interval=1,data_dir='/Users/sachintalathi/Work/Python/Data/cifar-10-batches-py',\
    data_augment_bool=False,train_bool=True):
    
    def Get_Data_Stats(data_dir):
        s=np.zeros((3072,))
        sq=np.zeros((3072,))
        for ind in range(5):
            D=pickle.load(open('%s/data_batch_%d'%(data_dir,ind+1)))
            data=D['data'].astype('float32')
            s+=np.sum(data,axis=0)
            sq+=np.sum(data*data,axis=0)

        Mean=1.0*s/50000
        Sq_Mean=1.0*sq/50000
        Mean_Sq=Mean*Mean
        Std=np.sqrt(Sq_Mean-Mean_Sq)
        return Mean,Std
    
    def batch_gen(X,y,N):
        while True:
            idx=np.random.choice(len(y),N)
            yield X[idx].astype('float32'),y[idx].astype('float32')

    def train_on_batch(batch_index,data_dir,Data_Mean,Data_Std,data_augment_bool,img_dim,mini_batch_size,f_train,LR):
        D=pickle.load(open('%s/data_batch_%d'%(data_dir,batch_index+1)))
        assert 'data' in D.keys()
        data=D['data'].astype('float32')
        data=data/255
        #data=(data-Data_Mean)/Data_Std
        data = data.reshape(data.shape[0], img_dim[0], img_dim[1], img_dim[2])
        assert 'labels' in D.keys()
        labels=np.array(D['labels'])
        train_loss_per_batch=0
        train_err_per_batch=0
        if data_augment_bool:
            train_batches=datagen.flow(data,labels,mini_batch_size) ### Generates data augmentation on the fly
        else:
            train_batches=batch_gen(data,labels,mini_batch_size) ### No data augmentation applied
        N_mini_batches=len(labels)//mini_batch_size
        
        for mini_batch in range(N_mini_batches):
            X,y=next(train_batches)
            loss,err=f_train(X,y,LR)
            train_loss_per_batch+=loss
            train_err_per_batch+=err
        train_loss_per_batch=train_loss_per_batch/N_mini_batches
        train_err_per_batch=train_err_per_batch/N_mini_batches
        return train_loss_per_batch,train_err_per_batch
    
    def val_on_batch(data_dir,img_dim,mini_batch_size,f_val):
        val_loss=0
        val_err=0    
        D_val=pickle.load(open('%s/test_batch'%(data_dir)))
        assert 'data' in D_val.keys()
        data=D_val['data'].astype('float32')
        data=data/255
        #data=(data-Data_Mean)/Data_Std
        data=data.reshape(data.shape[0], img_dim[0], img_dim[1], img_dim[2])
        assert 'labels' in D_val.keys()
        labels=np.array(D_val['labels'])
        val_batches=batch_gen(data,labels,mini_batch_size)
        N_val_batches=len(labels)//mini_batch_size
        for _ in range(N_val_batches):
            X,y=next(val_batches)
            loss,err=f_val(X,y)
            val_loss+=loss
            val_err+=err
        val_loss=val_loss/N_val_batches
        val_err=val_err/N_val_batches
        return val_loss,val_err

    per_epoch_train_stats=[];per_epoch_val_stats=[]
    Data_Mean,Data_Std=Get_Data_Stats(data_dir)

    print('Running Epochs')
    LR=lr_start
    for epoch in range(epochs):
        if train_bool:
            train_loss=0
            train_err=0
            for ind in range(N_train_batches):
                tic=time.clock()
                tlpb,tapb=train_on_batch(ind,data_dir,Data_Mean,Data_Std,data_augment_bool,img_dim,mini_batch_size,f_train,LR)
                toc=time.clock()
                print ('Epoch %d Data_Batch (Time) %d (%0.03f s) Learning_Rate %0.04f Train Loss (Error)\
                    %.03f (%.03f)'%(epoch,ind,toc-tic,LR,tlpb,tapb))
                train_loss+=tlpb
                train_err+=tapb
            train_loss=train_loss/N_train_batches
            train_err=train_err/N_train_batches
            per_epoch_train_stats.append([epoch,train_loss,train_err])
        if (epoch+1)%test_interval==0:
            val_loss,val_err=val_on_batch(data_dir,img_dim,mini_batch_size,f_val)
            per_epoch_val_stats.append([epoch,val_loss,val_err])
            print ('Epoch  (Time) %d (%0.03f s) Learning_Rate %0.04f Test Loss (Error)\
                %.03f (%.03f)'%(epoch,toc-tic,LR,val_loss,val_err))
        LR*=lr_decay

    return per_epoch_train_stats,per_epoch_val_stats