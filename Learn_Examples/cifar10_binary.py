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
import numpy.random as nr
import pickle,gzip
import optparse
from collections import OrderedDict
import time

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
                    target.append(y[:,:,start_positions[i][0]:end_positions[i][0],start_positions[i][1]:end_positions[i][1]].reshape((x.shape[0],get_data_dims(num_channels=num_channels,inner_size=inner_size))))
                return np.array(target)            
            else:
                pic = y[:,:,border_size:border_size+inner_size,border_size:border_size+inner_size] # just take the center for now
                target= pic.reshape((x.shape[0],get_data_dims()))
                return target
        else:
            for c in xrange(x.shape[0]): # loop over cases
                startY, startX = nr.randint(0,border_size*2 + 1), nr.randint(0,border_size*2 + 1)
                endY, endX = startY + inner_size, startX + inner_size
                pic = y[c,:,startY:endY,startX:endX]
                if nr.randint(2) == 0: # also flip the image with 50% probability
                    pic = pic[:,:,::-1]
                target.append(pic.reshape((get_data_dims(),)))
            return np.array(target)

def Generate_Viz(data,num_channels=3,img_size=(32,32),inner_size=(28,28),trim_bool=1,fig_bool=1):
  ## data is of shape (N,size)
  if ('data' not in data.keys()) and ('filenames' not in data.keys()):
    print 'Incorrect Cifar-10 data'
    sys.exit(0)
  Max_Ind=len(data['data'])
  while True:
    ind=np.random.randint(0,Max_Ind)
    d=data['data']
    if trim_bool:
      dt=trim_borders(d,num_channels,inner_size=inner_size[0],img_size=img_size[0])
      sz=inner_size
    else:
      dt=d
      sz=img_size
    img=dt[ind,:].reshape(num_channels,sz[0],sz[1]).transpose(1,2,0)
    filename=data['filenames'][ind]
    label=data['labels'][ind]
    if fig_bool:
      py.figure()
      py.imshow(img)
      py.title(filename+'-'+str(label))
    yield ind

def keras_cnn_network(input_var,num_channels=3,img_size=(32,32),quantization=None,H=1.,stochastic=False,node_type='ReLU'):
    if node_type.upper()=='RELU':
      nonlinearity=lasagne.nonlinearities.rectify
    if node_type.upper()=='ELU':
      nonlinearity=lasagne.nonlinearities.elu
    net={}
    net['l_in']=lasagne.layers.InputLayer((None,3,img_size[0],img_size[1]))
    
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


def simple_cnn_network(input_var,num_channels=3,img_size=(32,32), dropout_bool=0, node_type='ReLU',quantization=None,H=1.,stochastic=False):
  #Simple cnn network for prototyping
  if node_type.upper()=='RELU':
    nonlinearity=lasagne.nonlinearities.rectify
  if node_type.upper()=='ELU':
    nonlinearity=lasagne.nonlinearities.elu
  
  net={}
  net['l_in']=lasagne.layers.InputLayer((None,num_channels,img_size[0],img_size[1]))
  
  if quantization==None:
    net[0]=batch_norm(lasagne.layers.Conv2DLayer(net['l_in'],16,3,W=lasagne.init.HeUniform(),pad=1,nonlinearity=nonlinearity))
    net['l_out']=lasagne.layers.DenseLayer(net[0],num_units=10,nonlinearity=lasagne.nonlinearities.softmax)
  else:
    net[0]=batch_norm(lasagne.layers.qConv2DLayer(net['l_in'],16,3,nonlinearity=nonlinearity,H=H,stochastic=stochastic,quantization=quantization))
    net['l_out']=lasagne.layers.qDenseLayer(net[0],num_units=10,nonlinearity=lasagne.nonlinearities.softmax,\
      H=H,stochastic=stochastic,quantization=quantization)
  
  return net

def cnn_network(input_var,num_channels=3,img_size=(32,32),node_type='ReLU',quantization=None,stochastic=False,H=1.):
  ## The architecture that I designed for the Hyper-parameter Opt paper
  ## Assumne batch-norm throughout
  # Currently unable to reproduce findings from cuda-convnet.. Need debugging!
  if node_type.upper()=='RELU':
    nonlinearity=lasagne.nonlinearities.rectify
  if node_type.upper()=='ELU':
    nonlinearity=lasagne.nonlinearities.elu
    
  net={}
  net['l_in']=lasagne.layers.InputLayer(shape=(None,num_channels,img_size[0],img_size[1]),input_var=input_var)
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
  parser.add_option("--trim",action="store_true",dest="trim_bool",default=False,help="Boolean to trim data")
  parser.add_option("--epochs",help="epochs",dest="epochs",type=int,default=10)
  parser.add_option("--home-dir",help="Home Directory",dest="home_dir",type=str,default='')
  parser.add_option("--learning-rate",help="learning rate",dest="learning_rate",type=float,default=0.1)
  parser.add_option("--data-dir",help="Data Path",dest="data_dir",type=str,default='/prj/neo_lv/user/stalathi/DataSets/cifar-10-batches-py')
  parser.add_option("--nonlinearity",help="Nonlinearity type",dest="nonlinearity",type=str,default='RELU')
  parser.add_option("--quantization",help="Quantizatoin Type",dest="quantization",type=str,default=None)
  parser.add_option("--save-dir",help="Save Directory",dest="save_dir",type=str,default='/prj/neo_lv/user/stalathi/Lasagne_Models')
  parser.add_option("--model-file",help="Trained Model Pickle File",dest="model_file",type=str,default='')
  parser.add_option("--memo",help="Memo",dest="memo",type=str,default=None)
  parser.add_option("--inner-size",help="cropped img size",dest="inner_size",type=int,nargs=2)
  parser.add_option("--img-size",help="actual img size",dest="img_size",type=int,nargs=2)
  parser.add_option("--multiview",action="store_true",dest="multiview",default=False,help="multiview testing")
  (opts,args)=parser.parse_args()
  np.random.seed(42)
  
  if opts.inner_size==None and opts.trim_bool:
    opts.inner_size=(28,28)

  if opts.inner_size==None and not opts.trim_bool:
    opts.inner_size=(32,32)

  if opts.inner_size!=None:
    opts.trim_bool=True

  if opts.img_size==None:
    opts.img_size=(32,32)

  if os.path.isdir(opts.home_dir):
    sys.path.append(opts.home_dir)
  else:
    print "Home Directory: %s, not available"%opts.home_dir
    sys.exit(0)

  import lasagne
  from lasagne.regularization import regularize_layer_params_weighted, l2, l1
  from lasagne.regularization import regularize_layer_params
  import lasagne.layers.quantize as qt
  reload(qt)
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
  inp=T.tensor4()  ## Input data images: for example, (10,32,32,3) implies.. we have 10 images of size 32,32,3
  target=T.ivector() ## Target data labels: For above example... we have vector of 10x1
  LR = T.scalar('LR', dtype=theano.config.floatX)

  #Theano definition for output probability distribution and class prediction
  print 'Set up the network and theano variables'
  if opts.simple_cnn:
    net=simple_cnn_network(inp,img_size=opts.inner_size,quantization=opts.quantization)
  elif opts.cnn:
    net=cnn_network(inp,img_size=opts.inner_size,node_type=opts.nonlinearity,quantization=opts.quantization,stochastic=opts.stochastic,H=1.)
  elif opts.kcnn:
    net=keras_cnn_network(inp,img_size=opts.inner_size,node_type=opts.nonlinearity,quantization=opts.quantization)
  else:
    print ('No Network Defined')
    sys.exit(0)
  
  
  layers=lasagne.layers.get_all_layers(net['l_out'])
  for l in layers:
    print l.output_shape

  if len(opts.model_file)!=0:
    if os.path.isfile(opts.model_file):
      [pt,pv,values]=pickle.load(open(opts.model_file))
      lasagne.layers.set_all_param_values(net['l_out'],values)
    else:
      print('Trained model does not exist')
      sys.exit(0)
      
  train_output=lasagne.layers.get_output(net['l_out'],inp,deterministic=False)
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
    W_grads = qt.compute_grads(loss,net['l_out'])
    updates = lasagne.updates.adam(loss_or_grads=W_grads, params=paramsB, learning_rate=LR)
    updates = qt.clipping_scaling(updates,net['l_out'],opts.quantization)
    updates = OrderedDict(updates.items() + lasagne.updates.adam(loss_or_grads=loss, params=params, learning_rate=LR).items())

  val_output=lasagne.layers.get_output(net['l_out'],inp,deterministic=True)
  val_loss=T.mean(lasagne.objectives.categorical_crossentropy(val_output,target))
  val_err = T.mean(T.neq(T.argmax(val_output,axis=1), target),dtype=theano.config.floatX)
  val_pred=val_output.argmax(-1)
  
  #define training function
  f_train=theano.function([inp,target,LR],[loss,err],updates=updates,allow_input_downcast=True)

  #define the validation function
  f_val=theano.function([inp,target],[val_loss,val_err],allow_input_downcast=True)

  #Begin Training
  print 'begin training'
  tic=time.clock()
  pt,pv=qt.batch_train(datagen,5,32,f_train,f_val,LR_start,LR_decay,\
    img_dim=(3,opts.inner_size[0],opts.inner_size[1]),epochs=opts.epochs,test_interval=1,\
    data_augment_bool=opts.augment,data_dir=opts.data_dir,train_bool=opts.train,trim_bool=opts.trim_bool,multiview=opts.multiview)
  if opts.train:
    values=lasagne.layers.get_all_param_values(net['l_out'])
  toc=time.clock()
  
  print(toc-tic)

  ## Save Results
  if opts.save_dir!=None and opts.memo!=None:
    print 'Saving Model....'
    save_file='%s/Model_%s.pkl'%(opts.save_dir,opts.memo)
    o=open(save_file,'wb')
    pickle.dump([pt,pv,values],o)
    o.close()