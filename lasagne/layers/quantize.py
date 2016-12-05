import time
import sys
from collections import OrderedDict

import numpy as np
import numpy.random as nr
# specifying the gpu to use
# import theano.sandbox.cuda
# theano.sandbox.cuda.use('gpu1') 
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from .helper import get_all_layers
import pickle
import time

QUANTIZATION_STEP = [1.596, 0.996, 0.586, 0.335, 0.188, 0.104, 0.057, 0.031, 0.0168, 0.00914, 0.00496, 0.00270, 0.00146, 0.00079, 0.00043, 0.00023, \
                     0, 0, 0, 0, 0, 0, 0, 1.77e-6, 0, 0, 0, 0, 0, 0, 0, 1.34e-8]
STEP_ADJ_FACTOR=3.

def prob_sigmoid(x):
  return 0.5*(x/T.max(x)+1)

def hard_sigmoid(x):
    return T.clip((x+1.)/2.,0,1)

def quantize(value, step_size, num_bits):
    quant_input = value / (step_size)
    quant_input = np.round(quant_input)
    quant_input = np.clip(quant_input, -2**(num_bits-1), 2**(num_bits-1)-1)
#         print np.max(quant_input), np.min(quant_input)
    quant_input = quant_input * (step_size)
    return quant_input

# The binarization function
def binarization(W,H,deterministic=False,stochastic=False,srng=None,quantization=None):
    if quantization==None:
        print("no quantization")
        Wb = W
    
    elif quantization.upper()=='8BITS':
        num_bits=int(8)
        W_std=T.std(W)
        W_mean=T.mean(W)
        step_size= QUANTIZATION_STEP[num_bits-1] * (W_mean+W_std) * STEP_ADJ_FACTOR
        num_frac_bits = np.ceil(-np.log2(step_size))
        step_size = 2**(-num_frac_bits)
        Wb = T.cast(quantize(weights, step_size, num_bits),theano.config.floatX)

    elif quantization.upper()=='16BITS':
        num_bits=int(16)
        W_std=T.std(W)
        W_mean=T.mean(W)
        step_size= QUANTIZATION_STEP[num_bits-1] * (W_mean+W_std) * STEP_ADJ_FACTOR
        num_frac_bits = np.ceil(-np.log2(step_size))
        step_size = 2**(-num_frac_bits)
        Wb = T.cast(quantize(weights, step_size, num_bits),theano.config.floatX)

    elif quantization.upper()=='BINARY':
        Wb = hard_sigmoid(W/H)
        if stochastic and not deterministic:
            Wb = T.cast(srng.binomial(n=1, p=Wb, size=T.shape(Wb)), theano.config.floatX)
        else:
            Wb = T.round(Wb)
        Wb = T.cast(T.switch(Wb,H,-H), theano.config.floatX)
    
    elif quantization.upper()=='TERNARY':
        Wb=T.cast(T.gt(W,H)-T.lt(W,-H),theano.config.floatX)

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

def get_data_dims(num_channels=3,inner_size=28,idx=0):
        return inner_size**2 * num_channels if idx == 0 else 1

def trim_borders(x, num_channels=3, inner_size=28, img_size=32,test=False,multiview=False,num_views=9):
    target=[]
    y = x.reshape(np.shape(x)[0], num_channels, img_size, img_size)
    if inner_size>img_size:
        target=x
    
    border_size=(img_size-inner_size)/2
    
    if test: # don't need to loop over cases
        if multiview:
            start_positions = [(0,0), (0, border_size), (0, border_size*2),
                              (border_size, 0), (border_size, border_size), (border_size, border_size*2),
                              (border_size*2, 0), (border_size*2, border_size), (border_size*2, border_size*2)]
            end_positions = [(sy+inner_size, sx+inner_size) for (sy,sx) in start_positions]
            for i in xrange(num_views):
                val=y[:,:,start_positions[i][0]:end_positions[i][0],start_positions[i][1]:end_positions[i][1]]
                val_inv=val[:,:,:,::-1]
                target.append(val.reshape((x.shape[0],get_data_dims(num_channels=num_channels,inner_size=inner_size)),))
                target.append(val_inv.reshape((x.shape[0],get_data_dims(num_channels=num_channels,inner_size=inner_size)),))
            return np.array(target)            
        else:
            pic = y[:,:,border_size:border_size+inner_size,border_size:border_size+inner_size] # just take the center for now
            target= pic.reshape((x.shape[0],get_data_dims(num_channels=num_channels,inner_size=inner_size)))
            return target
    else:
        for c in xrange(x.shape[0]): # loop over cases
            startY, startX = nr.randint(0,border_size*2 + 1), nr.randint(0,border_size*2 + 1)
            endY, endX = startY + inner_size, startX + inner_size
            pic = y[c,:,startY:endY,startX:endX]
            if nr.randint(2) == 0: # also flip the image with 50% probability
                pic = pic[:,:,::-1]
            target.append(pic.reshape((get_data_dims(num_channels=num_channels,inner_size=inner_size),)))
        return np.array(target)

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

def batch_train(datagen,N_train_batches,mini_batch_size,f_train,f_val,lr_start,lr_decay,img_size=(3,32,32),img_dim=(3,28,28),\
    epochs=10,test_interval=1,data_dir='/Users/sachintalathi/Work/Python/Data/cifar-10-batches-py',\
    data_augment_bool=False,train_bool=True,trim_bool=False,multiview=False):
    
    if not trim_bool:
        img_dim=img_size

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

    def train_on_batch(batch_index,data_dir,data_augment_bool,img_size,img_dim,mini_batch_size,f_train,LR):
        D=pickle.load(open('%s/data_batch_%d'%(data_dir,batch_index+1)))
        assert 'data' in D.keys()
        data=D['data'].astype('float32')
        data=data/255
        if trim_bool:
            data=trim_borders(data, num_channels=3, inner_size=img_dim[1], img_size=img_size[1])

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
    
    def per_view_val_score(val_batches,N_val_batches):
        val_loss=0;val_err=0
        for _ in range(N_val_batches):
            X,y=next(val_batches)
            loss,err=f_val(X,y)
            val_loss+=loss
            val_err+=err
        val_loss=val_loss/N_val_batches
        val_err=val_err/N_val_batches
        return val_loss,val_err

    def val_on_batch(data_dir,img_dim,mini_batch_size,f_val,multiview):
        val_loss=0
        val_err=0    
        D_val=pickle.load(open('%s/test_batch'%(data_dir)))
        assert 'data' in D_val.keys()
        data=D_val['data'].astype('float32')
        assert 'labels' in D_val.keys()
        labels=np.array(D_val['labels'])
        data=data/255
        if trim_bool:
            data=trim_borders(data, num_channels=3, inner_size=img_dim[1], img_size=img_size[1],test=True,multiview=multiview,num_views=9)
            if multiview:  
                score=[]
                for i in range(np.shape(data)[0]):
                    data_view=data[i,:,:].reshape(data.shape[1], img_dim[0], img_dim[1], img_dim[2])
                    val_batches=batch_gen(data_view,labels,mini_batch_size)
                    N_val_batches=len(labels)//mini_batch_size
                    val_loss_per_view,val_err_per_view=per_view_val_score(val_batches,N_val_batches)
                    val_loss+=val_loss_per_view
                    val_err+=val_err_per_view
                    score.append(val_err_per_view)
                val_loss=val_loss/data.shape[0]
                val_err=val_err/data.shape[0]
                #print ['%.3f'%i for i in score],'mean:',np.mean(score)
            else:
                data_view=data.reshape(data.shape[0], img_dim[0], img_dim[1], img_dim[2])
                val_batches=batch_gen(data_view,labels,mini_batch_size)
                N_val_batches=len(labels)//mini_batch_size
                val_loss,val_err=per_view_val_score(val_batches,N_val_batches)

        else:
            data_view=data.reshape(data.shape[0], img_dim[0], img_dim[1], img_dim[2])
            val_batches=batch_gen(data_view,labels,mini_batch_size)
            N_val_batches=len(labels)//mini_batch_size
            val_loss,val_err=per_view_val_score(val_batches,N_val_batches)

        # for _ in range(N_val_batches):
        #     X,y=next(val_batches)
        #     loss,err=f_val(X,y)
        #     val_loss+=loss
        #     val_err+=err
        # val_loss=val_loss/N_val_batches
        # val_err=val_err/N_val_batches
        return val_loss,val_err

    per_epoch_train_stats=[];per_epoch_val_stats=[]
    #Data_Mean,Data_Std=Get_Data_Stats(data_dir)

    print('Running Epochs')
    LR=lr_start
    for epoch in range(epochs):
        if train_bool:
            train_loss=0
            train_err=0
            for ind in range(N_train_batches):
                tic=time.clock()
                tlpb,tapb=train_on_batch(ind,data_dir,data_augment_bool,img_size,img_dim,mini_batch_size,f_train,LR)
                toc=time.clock()
                print ('Epoch %d Data_Batch (Time) %d (%0.03f s) Learning_Rate %0.04f Train Loss (Error)\
                    %.03f (%.03f)'%(epoch,ind,toc-tic,LR,tlpb,tapb))
                train_loss+=tlpb
                train_err+=tapb
            train_loss=train_loss/N_train_batches
            train_err=train_err/N_train_batches
            per_epoch_train_stats.append([epoch,train_loss,train_err])
        if (epoch+1)%test_interval==0:
            tic=time.clock()
            val_loss,val_err=val_on_batch(data_dir,img_dim,mini_batch_size,f_val,multiview)
            per_epoch_val_stats.append([epoch,val_loss,val_err])
            toc=time.clock()
            print ('Epoch  (Time) %d (%0.03f s) Learning_Rate %0.04f Test Loss (Error)\
                %.03f (%.03f)'%(epoch,toc-tic,LR,val_loss,val_err))
        LR*=lr_decay

    return per_epoch_train_stats,per_epoch_val_stats
