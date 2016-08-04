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
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import lasagne
from lasagne.layers import batch_norm
from lasagne.layers.quantize import compute_grads,clipping_scaling,train,binarization
import cPickle as pickle
import gzip
from collections import OrderedDict
import optparse
py.ion()
np.random.seed(1234) 

def batch_gen(X,y,N):
  while True:
    idx=np.random.choice(len(y),N)
    yield X[idx].astype('float32'),y[idx].astype('float32') 

def get_first_layer_weight_stats(net):
    values=lasagne.layers.get_all_param_values(net['l_out'])
    val=values[0].flat    
    [N,I]=np.histogram(val[0:],bins=1000,range=(-2.1,2.1))
    return N,I


def mlp_network(input,dim=784,num_hidden_layers=1,num_hidden_nodes_per_layer=2048,dropout=0.,\
    stochastic=False,H=1.,nonlinearity=lasagne.nonlinearities.rectify,quantization=None):
  net={}
  net['l_in']=lasagne.layers.InputLayer((None,dim),input_var=input)
  net['l_d']=lasagne.layers.DropoutLayer(net['l_in'],p=.1)
  for i in range(num_hidden_layers):
    if i==0:
      net[2*i]=batch_norm(lasagne.layers.qDenseLayer(net['l_d'],stochastic=stochastic,H=H,\
        num_units=num_hidden_nodes_per_layer,nonlinearity=nonlinearity,quantization=quantization))     
    else:
        net[2*i]=batch_norm(lasagne.layers.qDenseLayer(net[2*(i-1)],stochastic=stochastic,\
            H=H,num_units=num_hidden_nodes_per_layer,nonlinearity=lasagne.nonlinearities.rectify,quantization=quantization))
    net[2*i+1]=lasagne.layers.DropoutLayer(net[2*i],p=dropout)
  
  net['l_out']=batch_norm(lasagne.layers.qDenseLayer(net[2*num_hidden_layers-1],stochastic=stochastic,\
    H=H,num_units=10,nonlinearity=lasagne.nonlinearities.softmax,quantization=quantization))
  return net

def cnn_network(input,dim=784,num_hidden_layers=1,num_filters=16,filter_size=3,stochastic=False,\
    H=1.,nonlinearity=lasagne.nonlinearities.rectify,quantization=None):
    #Defining a CNN network
    net={}
    net['l_in']=lasagne.layers.InputLayer((None,dim),input_var=input)
    net[0]=lasagne.layers.ReshapeLayer(net['l_in'],(-1,1,28,28))
    for i in range(1,num_hidden_layers+1):
        net[i]=batch_norm(lasagne.layers.qConv2DLayer(net[i-1],num_filters,filter_size,pad=1,\
            stochastic=stochastic,H=H,nonlinearity=nonlinearity,quantization=quantization))
    net['l_out']=batch_norm(lasagne.layers.qDenseLayer(net[num_hidden_layers],stochastic=stochastic,\
        H=H,num_units=10,nonlinearity=lasagne.nonlinearities.softmax,quantization=quantization))
    return net

if __name__ == "__main__":
    
    parser=optparse.OptionParser()
    parser.add_option("-M", "--MLP",action="store_true",dest="mlp",default=False,help="Use MLP Network")
    parser.add_option("-C", "--cnn",action="store_true",dest="cnn",default=False,help="Use CNN Network")
    parser.add_option("-B", "--batch-norm",action="store_true",dest="batch_norm",default=False,help="Invoke Batchnorm")
    parser.add_option("--stochastic",action="store_true",dest="stochastic",default=False,help="Stochastic Training")
    parser.add_option("--batch-size",help="Batch Size",dest="batch_size",type=int,default=64)
    parser.add_option("-c","--cool",action="store_true",dest="cool",default=False,help="Cool Learning Rate")
    parser.add_option("--cool-factor",help="Cool Factor",dest="cool_factor",type=int,default=10)
    parser.add_option("--epochs",help="epochs",dest="epochs",type=int,default=10)
    parser.add_option("--learning-rate",help="learning rate",dest="learning_rate",type=float,default=0.001)
    parser.add_option("--data-dir",help="Data Path",dest="data_dir",type=str,default='/Users/sachintalathi/Work/Python/Data')
    parser.add_option("--nonlinearity",help="Nonlinearity type",dest="nonlinearity",type=str,default='RELU')
    parser.add_option("--num-units",help="num hidden units",dest="num_hidden_units",type=int,default=2048)
    parser.add_option("--num-layers",help="num layers",dest="num_hidden_layers",type=int,default=1)
    parser.add_option("--quantization",help="Quantization Method (Binary, Round, Pow, StocM)",dest="quantization",type=str,default=None)
    parser.add_option("--save-dir",help="Save Directory",dest="save_dir",type=str,default='/prj/neo_lv/user/stalathi/Lasagne_Models')
    parser.add_option("--memo",help="Memo",dest="memo",type=str,default=None)

    (opts,args)=parser.parse_args()



    # Commandline parameters
    quantization=opts.quantization
    batch_size = opts.batch_size
    num_units = opts.num_hidden_units
    n_hidden_layers = opts.num_hidden_layers
    num_epochs = opts.epochs
    dropout_in = 0. # 0. means no dropout
    dropout_hidden = 0.
    stochastic = opts.stochastic
    LR_start = opts.learning_rate
    if opts.nonlinearity=='RELU':
        nonlinearity=lasagne.nonlinearities.rectify
    if opts.nonlinearity=='ELU':
        nonlinearity=lasagne.nonlinearities.elu
    if opts.cool:
        LR_fin = 0.000003
    else:
        LR_fin=opts.learning_rate
    LR_decay = (LR_fin/LR_start)**(1./num_epochs)

    ## Hard coded parameters
    H = 1.
    W_LR_scale = "Glorot" # "Glorot" means we are using the coefficients from Glorot's paper

    
    print('Build the Network...') 
    
    # Prepare Theano variables for inputs and targets
    input=T.matrix()
    target=T.ivector('targets')
    #target = T.matrix('targets')
  
    LR = T.scalar('LR', dtype=theano.config.floatX)

    ###define network
    if opts.mlp:
        net=mlp_network(input,stochastic=stochastic,num_hidden_layers=n_hidden_layers,\
            num_hidden_nodes_per_layer=num_units,H=H,nonlinearity=nonlinearity,dropout=dropout_hidden,quantization=quantization)

    if opts.cnn:
        net=cnn_network(input,num_hidden_layers=n_hidden_layers,num_filters=128,filter_size=3,\
            stochastic=stochastic,H=H,nonlinearity=nonlinearity,quantization=quantization)
    
    # Generate Figure:
    fig_on=0
    
    if fig_on:
        py.figure();py.subplot(131)
        N,I=get_first_layer_weight_stats(net)
        py.plot(I[0:-1],N)

    print ('Define Train Output Theano Function..')
    train_output = lasagne.layers.get_output(net['l_out'],deterministic=False)
    
    loss=T.mean(lasagne.objectives.categorical_crossentropy(train_output,target))
    #loss=T.mean(lasagne.objectives.multiclass_hinge_loss(train_output,target))
    err=T.mean(T.neq(T.argmax(train_output,axis=1), target),dtype=theano.config.floatX)

    if quantization==None:
        params = lasagne.layers.get_all_params(net['l_out'], trainable=True)
        updates = lasagne.updates.adam(loss_or_grads=loss, params=params, learning_rate=LR)
    else:
        paramsB = lasagne.layers.get_all_params(net['l_out'],  quantize=True)
        params = lasagne.layers.get_all_params(net['l_out'], trainable=True, quantize=False)
        W_grads = compute_grads(loss,net['l_out'])
        updates = lasagne.updates.adam(loss_or_grads=W_grads, params=paramsB, learning_rate=LR)
        updates = clipping_scaling(updates,net['l_out'],quantization)
        updates = OrderedDict(updates.items() + lasagne.updates.adam(loss_or_grads=loss, params=params, learning_rate=LR).items())
        

    print ('Define Test Output Theano Function..')
    
    test_output = lasagne.layers.get_output(net['l_out'], deterministic=True)
    test_loss=T.mean(lasagne.objectives.categorical_crossentropy(test_output,target))
    test_err = T.mean(T.neq(T.argmax(test_output,axis=1), target),dtype=theano.config.floatX)
 
    
    # Compile a function performing a training step on a mini-batch (by giving the updates dictionary) 
    # and returning the corresponding training loss:
    train_fn = theano.function([input, target, LR], [loss,err], updates=updates,allow_input_downcast=True,)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input, target], [test_loss, test_err],allow_input_downcast=True)

   
    print("Read Data...")
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



    print('Train the network...')
    results=train(train_fn,val_fn,N_Batches,N_generator,LR_start,LR_decay,num_epochs)
    values=lasagne.layers.get_all_param_values(net['l_out'])

    if opts.save_dir!=None and opts.memo!=None:
        save_file='%s/Model_%s.pkl'%(opts.save_dir,opts.memo)
        o=open(save_file,'wb')
        pickle.dump([results,values],o)
        o.close()

    if fig_on:
        py.subplot(132)
        N,I=get_first_layer_weight_stats(net)
        py.plot(I[0:-1],N)
        srng = RandomStreams(lasagne.random.get_rng().randint(1, 2147462579))
        [N,I]=np.histogram(binarization(values[0].flat[0:],1.0,deterministic=True,stochastic=False,\
            quantization=quantization,srng=srng).eval(),bins=1000,range=(-2.1,2.1))
        py.subplot(133)
        py.plot(I[0:-1],N)
        raw_input()