import sys
## Some local adjustments to paths (not required for ubuntu machine)
if '/Users/sachintalathi/Work/Python/Virtualenvs/DL_Lasagne/lib/python2.7/site-packages' in sys.path:
  del(sys.path[sys.path.index('/Users/sachintalathi/Work/Python/Virtualenvs/DL_Lasagne/lib/python2.7/site-packages')])
sys.path.append('/usr/local/lib/python2.7/site-packages/')

import pylab as py
import theano
import lasagne
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from lasagne.regularization import regularize_layer_params_weighted, l2, l1
from lasagne.regularization import regularize_layer_params
from lasagne.layers.quantize import compute_grads,clipping_scaling,train,binarization,batch_train
from lasagne.layers import batch_norm
from lasagne.image import ImageDataGenerator
import theano.tensor as T
import numpy as np
import pickle,gzip
import optparse
from collections import OrderedDict
import time

def simple_cnn_network(input_var,batchnorm_bool=0, dropout_bool=0, node_type='ReLU'):
  #Simple cnn network for prototyping
  if node_type.upper()=='RELU':
    nonlinearity=lasagne.nonlinearities.rectify
  if node_type.upper()=='ELU':
    nonlinearity=lasagne.nonlinearities.elu
  
  net={}
  net['l_in']=lasagne.layers.InputLayer((None,3,32,32))
  
  if batchnorm_bool:
    net[0]=batch_norm(lasagne.layers.Conv2DLayer(net['l_in'],16,3,W=lasagne.init.HeUniform(),pad=1,nonlinearity=nonlinearity))
  else:
    net[0]=lasagne.layers.Conv2DLayer(net['l_in'],16,3,W=lasagne.init.HeUniform(),pad=1,nonlinearity=nonlinearity)
  
  net['l_out']=lasagne.layers.DenseLayer(net[0],num_units=10,nonlinearity=lasagne.nonlinearities.softmax)
  return net

def cnn_network(input_var,node_type='ReLU',quantization=None,stochastic=False,H=1.):
  ## The architecture that I designed for the Hyper-parameter Opt paper
  ## Assumne batch-norm throughout
  if node_type.upper()=='RELU':
    nonlinearity=lasagne.nonlinearities.rectify
  if node_type.upper()=='ELU':
    nonlinearity=lasagne.nonlinearities.elu
    
  net={}
  net['l_in']=lasagne.layers.InputLayer(shape=(None,3,32,32),input_var=input_var)
  
  if quantization==None:
    net[0]=batch_norm(lasagne.layers.Conv2DLayer(net['l_in'],256,3,pad=0,nonlinearity=nonlinearity))
    net[1]=batch_norm(lasagne.layers.Conv2DLayer(net[0],128,3,stride=(2,2),pad=1,W=lasagne.init.HeUniform(),nonlinearity=nonlinearity))
    net[2]=batch_norm(lasagne.layers.Conv2DLayer(net[1],256,3,stride=(1,1),pad=0,W=lasagne.init.HeUniform(),nonlinearity=nonlinearity))
    net[3]=batch_norm(lasagne.layers.Conv2DLayer(net[2],256,3,stride=(1,1),pad=0,W=lasagne.init.HeUniform(),nonlinearity=nonlinearity))
    net[4]=batch_norm(lasagne.layers.Conv2DLayer(net[3],256,3,stride=(2,2),pad=1,W=lasagne.init.HeUniform(),nonlinearity=nonlinearity))
    net[5]=batch_norm(lasagne.layers.Conv2DLayer(net[4],128,7,stride=(5,5),pad=2,W=lasagne.init.HeUniform(),nonlinearity=nonlinearity))
    net['l_out']=lasagne.layers.qDenseLayer(net[5],num_units=10,nonlinearity=lasagne.nonlinearities.softmax)
  else:
    net[0]=batch_norm(lasagne.layers.qConv2DLayer(net['l_in'],256,3,pad=0,nonlinearity=nonlinearity,H=H,stochastic=stochastic,quantization=quantization))
    net[1]=batch_norm(lasagne.layers.qConv2DLayer(net[0],128,3,stride=(2,2),pad=1,nonlinearity=nonlinearity,H=H,stochastic=stochastic,quantization=quantization))
    net[2]=batch_norm(lasagne.layers.qConv2DLayer(net[1],256,3,stride=(1,1),pad=0,nonlinearity=nonlinearity,H=H,stochastic=stochastic,quantization=quantization))
    net[3]=batch_norm(lasagne.layers.qConv2DLayer(net[2],256,3,stride=(1,1),pad=0,nonlinearity=nonlinearity,H=H,stochastic=stochastic,quantization=quantization))
    net[4]=batch_norm(lasagne.layers.qConv2DLayer(net[3],256,3,stride=(2,2),pad=1,nonlinearity=nonlinearity,H=H,stochastic=stochastic,quantization=quantization))
    net[5]=batch_norm(lasagne.layers.qConv2DLayer(net[4],128,7,stride=(5,5),pad=2,nonlinearity=nonlinearity,H=H,stochastic=stochastic,quantization=quantization))
    net['l_out']=lasagne.layers.qDenseLayer(net[5],num_units=10,nonlinearity=lasagne.nonlinearities.softmax,\
      H=H,stochastic=stochastic,quantization=quantization)
  
  return net

if __name__=="__main__":
  parser=optparse.OptionParser()
  parser.add_option("-S", "--simple-cnn",action="store_true",dest="simple_cnn",default=False,help="Use Simple CNN Network")
  parser.add_option("-C", "--cnn",action="store_true",dest="cnn",default=False,help="Use CNN Network")
  parser.add_option("-A", "--augment",action="store_true",dest="augment",default=False,help="Augment Training Data")
  parser.add_option("--stochastic",action="store_true",dest="stochastic",default=False,help="Stochastic Training")
  parser.add_option("--batch-size",help="Batch Size",dest="batch_size",type=int,default=64)
  parser.add_option("-c","--cool",action="store_true",dest="cool",default=False,help="Cool Learning Rate")
  parser.add_option("--epochs",help="epochs",dest="epochs",type=int,default=10)
  parser.add_option("--learning-rate",help="learning rate",dest="learning_rate",type=float,default=0.1)
  parser.add_option("--data-dir",help="Data Path",dest="data_dir",type=str,default='/Users/sachintalathi/Work/Python/Data')
  parser.add_option("--nonlinearity",help="Nonlinearity type",dest="nonlinearity",type=str,default='RELU')
  parser.add_option("--quantization",help="Quantizatoin Type",dest="quantization",type=str,default=None)

  (opts,args)=parser.parse_args()
  np.random.seed(42)
  

  LR_start = opts.learning_rate
  if opts.cool:
    LR_fin = 0.000003
  else:
    LR_fin=opts.learning_rate
  LR_decay = (LR_fin/LR_start)**(1./opts.epochs)
  
  ####### Read data and Generate Data Batch Generator ############
  ## Data is being directly read into batch_train module
  
  datagen = ImageDataGenerator(
          featurewise_center=False,  # set input mean to 0 over the dataset
          samplewise_center=False,  # set each sample mean to 0
          featurewise_std_normalization=False,  # divide inputs by std of the dataset
          samplewise_std_normalization=False,  # divide each input by its std
          zca_whitening=False,  # apply ZCA whitening
          rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)
          width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
          height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
          horizontal_flip=True,  # randomly flip images
          vertical_flip=True)  # randomly flip images
  
  ######## Define theano variables and theano functions for training/validation and testing#############
  input=T.tensor4()
  target=T.ivector()
  LR = T.scalar('LR', dtype=theano.config.floatX)

  #Theano definition for output probability distribution and class prediction
  print 'Set up the network and theano variables'
  if opts.simple_cnn:
    net=simple_cnn_network(input,batchnorm_bool=batch_norm)
  elif opts.cnn:
    net=cnn_network(input,node_type=opts.nonlinearity,quantization=opts.quantization,stochastic=opts.stochastic,H=1.)
  else:
    print ('No Network Defined')
    sys.exit(0)
          
      
  train_output=lasagne.layers.get_output(net['l_out'],input,deterministic=False)
  train_pred=train_output.argmax(-1)
  loss=T.mean(lasagne.objectives.categorical_crossentropy(train_output,target))
  err=T.mean(T.neq(T.argmax(train_output,axis=1), target),dtype=theano.config.floatX)

  if opts.quantization==None:
        params = lasagne.layers.get_all_params(net['l_out'], trainable=True)
        updates = lasagne.updates.adam(loss_or_grads=loss, params=params, learning_rate=LR)
  else:
      paramsB = lasagne.layers.get_all_params(net['l_out'],  quantize=True)
      params = lasagne.layers.get_all_params(net['l_out'], trainable=True, quantize=False)
      W_grads = compute_grads(loss,net['l_out'])
      updates = lasagne.updates.adam(loss_or_grads=W_grads, params=paramsB, learning_rate=LR)
      updates = clipping_scaling(updates,net['l_out'],opts.quantization)
      updates = OrderedDict(updates.items() + lasagne.updates.adam(loss_or_grads=loss, params=params, learning_rate=LR).items())

  val_output=lasagne.layers.get_output(net['l_out'],input,deterministic=True)
  val_loss=T.mean(lasagne.objectives.categorical_crossentropy(val_output,target))
  val_err = T.mean(T.neq(T.argmax(val_output,axis=1), target),dtype=theano.config.floatX)
  val_pred=val_output.argmax(-1)
  
  #define training function
  f_train=theano.function([input,target,LR],[loss,err],updates=updates,allow_input_downcast=True)

  #define the validation function
  f_val=theano.function([input,target],[val_loss,val_err],allow_input_downcast=True)

 
  #Begin Training
  print 'begin training'
  tic=time.clock()
  pt,pv=batch_train(datagen,5,32,f_train,f_val,LR_start,LR_decay,(3,32,32),\
    epochs=opts.epochs,test_interval=1,data_augment_bool=opts.augment,data_dir=opts.data_dir)
  toc=time.clock()
  print(toc-tic)