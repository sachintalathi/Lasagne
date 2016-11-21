import sys
import os
import matplotlib
if 'MACOSX' in matplotlib.get_backend().upper():
  matplotlib.use('TKAgg')
#Command to Train network
#python Learn_Examples/cifar10_binary.py -S --train --epochs 10 --home-dir /prj/neo-nas/users/stalathi/sachin-repo/Neo/SysPTSD/Lasagne --quantization ternary --memo test_ternary

#Command to run test on trained model
#python Learn_Examples/cifar10_binary.py -C --epochs 10 --home-dir /prj/neo-nas/users/stalathi/sachin-repo/Neo/SysPTSD/Lasagne --quantization pow --model-file /prj/neo_lv/user/stalathi/Lasagne_Models/Model_Cifar10_Pow.pkl 

### Example of how the code is to be run on cluster machines
#bsub -Ip -q QcDev -P Neo -a neogpu -app 1gpuEP -R nvTitanX run_lasagnetest.sh > /prj/neo_lv/user/stalathi/Lasagne_Models/LogFiles/Log_Cifar10_NoQuantize.log 2
  

## Some local adjustments to paths (not required for ubuntu machine)
## Not mandatory
if '/Users/sachintalathi/Work/Python/Virtualenvs/DL_Lasagne/lib/python2.7/site-packages' in sys.path:
  del(sys.path[sys.path.index('/Users/sachintalathi/Work/Python/Virtualenvs/DL_Lasagne/lib/python2.7/site-packages')])
sys.path.append('/usr/local/lib/python2.7/site-packages/')

import pylab as py
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
import pickle,gzip
import optparse
from collections import OrderedDict
import time

def Generate_Viz(data,fig_bool=1):
  if ('data' not in data.keys()) and ('filenames' not in data.keys()):
    print 'Incorrect Cifar-10 data'
    sys.exit(0)
  Max_Ind=len(data['data'])
  while True:
    ind=np.random.randint(0,Max_Ind)
    d=data['data'][ind,:].reshape(3,32,32).transpose(1,2,0)
    filename=data['filenames'][ind]
    label=data['labels'][ind]
    if fig_bool:
      py.figure()
      py.imshow(d)
      py.title(filename+'-'+str(label))
    yield ind

def keras_cnn_network(input_var,quantization=None,H=1.,stochastic=False,node_type='ReLU'):
    if node_type.upper()=='RELU':
      nonlinearity=lasagne.nonlinearities.rectify
    if node_type.upper()=='ELU':
      nonlinearity=lasagne.nonlinearities.elu
    net={}
    net['l_in']=lasagne.layers.InputLayer((None,3,32,32),input_var=input_var)
    
    if quantization==None:
      net[0]=batch_norm(lasagne.layers.Conv2DLayer(net['l_in'],32,3,W=lasagne.init.HeUniform(),pad='same',nonlinearity=nonlinearity))
      net[1]=batch_norm(lasagne.layers.Conv2DLayer(net[0],32,3,W=lasagne.init.HeUniform(),pad='same',nonlinearity=nonlinearity))
    else:
      net[0]=batch_norm(lasagne.layers.qConv2DLayer(net['l_in'],32,3,pad='same',nonlinearity=nonlinearity,H=H,stochastic=stochastic,quantization=quantization))
      net[1]=batch_norm(lasagne.layers.qConv2DLayer(net[0],32,3,pad='same',nonlinearity=nonlinearity,H=H,stochastic=stochastic,quantization=quantization))

    net[2]=lasagne.layers.MaxPool2DLayer(net[1],2)
    net[3]=lasagne.layers.DropoutLayer(net[2],p=.25)
    
    if quantization==None:
      net[4]=batch_norm(lasagne.layers.Conv2DLayer(net[3],64,3,W=lasagne.init.HeUniform(),pad='same',nonlinearity=nonlinearity))
      net[5]=batch_norm(lasagne.layers.Conv2DLayer(net[4],64,3,W=lasagne.init.HeUniform(),pad='same',nonlinearity=nonlinearity))
    else:
      net[4]=batch_norm(lasagne.layers.qConv2DLayer(net[3],64,3,pad='same',nonlinearity=nonlinearity,H=H,stochastic=stochastic,quantization=quantization))
      net[5]=batch_norm(lasagne.layers.qConv2DLayer(net[4],64,3,pad='same',nonlinearity=nonlinearity,H=H,stochastic=stochastic,quantization=quantization))

    net[6]=lasagne.layers.MaxPool2DLayer(net[5],2)
    net[7]=lasagne.layers.DropoutLayer(net[6],p=.25)
   
    if quantization==None:
      net[8]=batch_norm(lasagne.layers.DenseLayer(net[7],num_units=512,nonlinearity=lasagne.nonlinearities.rectify))
    else:
      net[8]=batch_norm(lasagne.layers.qDenseLayer(net[7],num_units=512,nonlinearity=lasagne.nonlinearities.rectify,H=H,stochastic=stochastic,quantization=quantization))

    net[9]=lasagne.layers.DropoutLayer(net[8],p=.5)
    
    if quantization==None:
      net['l_out']=lasagne.layers.DenseLayer(net[9],num_units=10,nonlinearity=lasagne.nonlinearities.softmax)
    else:
      net['l_out']=lasagne.layers.qDenseLayer(net[9],num_units=10,nonlinearity=lasagne.nonlinearities.softmax,H=H,stochastic=stochastic,quantization=quantization)
    
    return net


def simple_cnn_network(input_var, dropout_bool=0, node_type='ReLU',quantization=None,H=1.,stochastic=False):
  #Simple cnn network for prototyping
  if node_type.upper()=='RELU':
    nonlinearity=lasagne.nonlinearities.rectify
  if node_type.upper()=='ELU':
    nonlinearity=lasagne.nonlinearities.elu
  
  net={}
  net['l_in']=lasagne.layers.InputLayer((None,3,32,32))
  
  if quantization==None:
    net[0]=batch_norm(lasagne.layers.Conv2DLayer(net['l_in'],16,3,W=lasagne.init.HeUniform(),pad=1,nonlinearity=nonlinearity))
    net['l_out']=lasagne.layers.DenseLayer(net[0],num_units=10,nonlinearity=lasagne.nonlinearities.softmax)
  else:
    net[0]=batch_norm(lasagne.layers.qConv2DLayer(net['l_in'],16,3,nonlinearity=nonlinearity,H=H,stochastic=stochastic,quantization=quantization))
    net['l_out']=lasagne.layers.qDenseLayer(net[0],num_units=10,nonlinearity=lasagne.nonlinearities.softmax,\
      H=H,stochastic=stochastic,quantization=quantization)
  
  return net

def cnn_network(input_var,node_type='ReLU',quantization=None,stochastic=False,H=1.):
  ## The architecture that I designed for the Hyper-parameter Opt paper
  ## Assumne batch-norm throughout
  # Currently unable to reproduce findings from cuda-convnet.. Need debugging!
  if node_type.upper()=='RELU':
    nonlinearity=lasagne.nonlinearities.rectify
  if node_type.upper()=='ELU':
    nonlinearity=lasagne.nonlinearities.elu
    
  net={}
  net['l_in']=lasagne.layers.InputLayer(shape=(None,3,32,32),input_var=input_var)
  net['l_d']=lasagne.layers.DropoutLayer(net['l_in'],p=.5)
  if quantization==None:
    net[0]=batch_norm(lasagne.layers.Conv2DLayer(net['l_d'],256,3,pad=0,nonlinearity=nonlinearity))
    net['rnorm1']=lasagne.layers.LocalResponseNormalization2DLayer(net[0])
    net[1]=batch_norm(lasagne.layers.Conv2DLayer(net['rnorm1'],128,3,stride=(2,2),pad=1,W=lasagne.init.HeUniform(),nonlinearity=nonlinearity))
    net[2]=batch_norm(lasagne.layers.Conv2DLayer(net[1],256,3,stride=(1,1),pad=0,W=lasagne.init.HeUniform(),nonlinearity=nonlinearity))
    net[3]=batch_norm(lasagne.layers.Conv2DLayer(net[2],256,3,stride=(1,1),pad=0,W=lasagne.init.HeUniform(),nonlinearity=nonlinearity))
    net[4]=batch_norm(lasagne.layers.Conv2DLayer(net[3],256,3,stride=(2,2),pad=1,W=lasagne.init.HeUniform(),nonlinearity=nonlinearity))
    net[5]=batch_norm(lasagne.layers.Conv2DLayer(net[4],128,7,stride=(5,5),pad=2,W=lasagne.init.HeUniform(),nonlinearity=nonlinearity))
    net['rnorm2']=lasagne.layers.LocalResponseNormalization2DLayer(net[5],n=3)
    net['o_d']=lasagne.layers.DropoutLayer(net['rnorm2'],p=.5)
    net['l_out']=lasagne.layers.DenseLayer(net['o_d'],num_units=10,nonlinearity=lasagne.nonlinearities.softmax)
  else:
    net[0]=batch_norm(lasagne.layers.qConv2DLayer(net['l_d'],256,3,pad=0,nonlinearity=nonlinearity,H=H,stochastic=stochastic,quantization=quantization))
    net[1]=lasagne.layers.LocalResponseNormalization2DLayer(net[0])
    net[2]=batch_norm(lasagne.layers.qConv2DLayer(net[1],128,3,stride=(2,2),pad=1,nonlinearity=nonlinearity,H=H,stochastic=stochastic,quantization=quantization))
    net[3]=batch_norm(lasagne.layers.qConv2DLayer(net[2],256,3,stride=(1,1),pad=0,nonlinearity=nonlinearity,H=H,stochastic=stochastic,quantization=quantization))
    net[4]=batch_norm(lasagne.layers.qConv2DLayer(net[3],256,3,stride=(1,1),pad=0,nonlinearity=nonlinearity,H=H,stochastic=stochastic,quantization=quantization))
    net[5]=batch_norm(lasagne.layers.qConv2DLayer(net[4],256,3,stride=(2,2),pad=1,nonlinearity=nonlinearity,H=H,stochastic=stochastic,quantization=quantization))
    net[6]=batch_norm(lasagne.layers.qConv2DLayer(net[5],128,7,stride=(5,5),pad=2,nonlinearity=nonlinearity,H=H,stochastic=stochastic,quantization=quantization))
    net[7]=lasagne.layers.LocalResponseNormalization2DLayer(net[6],n=3)

    net['o_d']=lasagne.layers.DropoutLayer(net[7],p=.5)
    net['l_out']=lasagne.layers.qDenseLayer(net['o_d'],num_units=10,nonlinearity=lasagne.nonlinearities.softmax,\
      H=H,stochastic=stochastic,quantization=quantization)  ## Will produce an out prediction vector of dimension 10
  
  return net

if __name__=="__main__":
  #### Command Line Parser ########
  parser=optparse.OptionParser()
  parser.add_option("-S", "--simple-cnn",action="store_true",dest="simple_cnn",default=False,help="Use Simple CNN Network")
  parser.add_option("-C", "--cnn",action="store_true",dest="cnn",default=False,help="Use CNN Network")
  parser.add_option("-K", "--kcnn",action="store_true",dest="kcnn",default=False,help="Use Keras CNN Network")
  parser.add_option("-A", "--augment",action="store_true",dest="augment",default=False,help="Augment Training Data")
  parser.add_option("--train",action="store_true",dest="train",default=False,help="Train the Network")
  parser.add_option("--stochastic",action="store_true",dest="stochastic",default=False,help="Stochastic Training")
  parser.add_option("--batch-size",help="Batch Size",dest="batch_size",type=int,default=64)
  parser.add_option("-c","--cool",action="store_true",dest="cool",default=False,help="Cool Learning Rate")
  parser.add_option("--epochs",help="epochs",dest="epochs",type=int,default=10)
  parser.add_option("--home-dir",help="Home Directory",dest="home_dir",type=str,default='')
  parser.add_option("--learning-rate",help="learning rate",dest="learning_rate",type=float,default=0.1)
  parser.add_option("--data-dir",help="Data Path",dest="data_dir",type=str,default='/prj/neo_lv/user/stalathi/DataSets/cifar-10-batches-py')
  parser.add_option("--nonlinearity",help="Nonlinearity type",dest="nonlinearity",type=str,default='RELU')
  parser.add_option("--quantization",help="Quantizatoin Type",dest="quantization",type=str,default=None)
  parser.add_option("--save-dir",help="Save Directory",dest="save_dir",type=str,default='/prj/neo_lv/user/stalathi/Lasagne_Models')
  parser.add_option("--model-file",help="Trained Model Pickle File",dest="model_file",type=str,default='')
  parser.add_option("--memo",help="Memo",dest="memo",type=str,default=None)

  (opts,args)=parser.parse_args()
  np.random.seed(42)
  
  if os.path.isdir(opts.home_dir):
    sys.path.append(opts.home_dir)
  else:
    print "Home Directory: %s, not available"%opts.home_dir
    sys.exit(0)

  import lasagne
  from lasagne.regularization import regularize_layer_params_weighted, l2, l1
  from lasagne.regularization import regularize_layer_params
  from lasagne.layers.quantize import compute_grads,clipping_scaling,train,binarization,batch_train
  from lasagne.layers import batch_norm
  from lasagne.image import ImageDataGenerator

  ### Learning_Rate Cooling Set up.. Currently supporting exponential cooling only 
  LR_start = opts.learning_rate
  if opts.cool:
    LR_fin = 0.000003
  else:
    LR_fin=opts.learning_rate
  LR_decay = (LR_fin/LR_start)**(1./opts.epochs)
  
  ####### Read data and Generate Data Batch Generator ############
  ## Inline data augmentation generator
  datagen = ImageDataGenerator(
          featurewise_center=False,  # set input mean to 0 over the dataset
          samplewise_center=False,  # set each sample mean to 0
          featurewise_std_normalization=False,  # divide inputs by std of the dataset
          samplewise_std_normalization=False,  # divide each input by its std
          zca_whitening=False,  # apply ZCA whitening
          rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
          width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
          height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
          horizontal_flip=True,  # randomly flip images
          vertical_flip=False)  # randomly flip images
  
  ######## Define theano variables and theano functions for training/validation and testing#############
  input=T.tensor4()  ## Input data images: for example, (10,32,32,3) implies.. we have 10 images of size 32,32,3
  target=T.ivector() ## Target data labels: For above example... we have vector of 10x1
  LR = T.scalar('LR', dtype=theano.config.floatX)

  #Theano definition for output probability distribution and class prediction
  print 'Set up the network and theano variables'
  if opts.simple_cnn:
    net=simple_cnn_network(input,quantization=opts.quantization)
  elif opts.cnn:
    net=cnn_network(input,node_type=opts.nonlinearity,quantization=opts.quantization,stochastic=opts.stochastic,H=1.)
  elif opts.kcnn:
    net=keras_cnn_network(input,node_type=opts.nonlinearity,quantization=opts.quantization)
  else:
    print ('No Network Defined')
    sys.exit(0)
  
  if len(opts.model_file)!=0:
    if os.path.isfile(opts.model_file):
      [pt,pv,values]=pickle.load(open(opts.model_file))
      lasagne.layers.set_all_param_values(net['l_out'],values)
    else:
      print('Trained model does not exist')
      sys.exit(0)
      
  train_output=lasagne.layers.get_output(net['l_out'],input,deterministic=False)
  train_pred=train_output.argmax(-1)
  loss=T.mean(lasagne.objectives.categorical_crossentropy(train_output,target))
  err=T.mean(T.neq(T.argmax(train_output,axis=1), target),dtype=theano.config.floatX)

  if opts.quantization==None:
    layers={}
    for k in net.keys():
      layers[net[k]]=0.0005
    l2_penalty = regularize_layer_params_weighted(layers, l2)  
    loss=loss+l2_penalty
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
  pt,pv=batch_train(datagen,5,32,f_train,f_val,LR_start,LR_decay,(3,32,32),epochs=opts.epochs,\
    test_interval=1,data_augment_bool=opts.augment,data_dir=opts.data_dir,train_bool=opts.train)
  if opts.train:
    values=lasagne.layers.get_all_param_values(net['l_out'])
  toc=time.clock()
  
  print(toc-tic)

  ## Save Results
  if opts.save_dir!=None and opts.memo!=None:
    save_file='%s/Model_%s.pkl'%(opts.save_dir,opts.memo)
    o=open(save_file,'wb')
    pickle.dump([pt,pv,values],o)
    o.close()