import sys
## Some local adjustments to paths (not required for ubuntu machine)
if '/Users/sachintalathi/Work/Python/Virtualenvs/DL_Lasagne/lib/python2.7/site-packages' in sys.path:
  del(sys.path[sys.path.index('/Users/sachintalathi/Work/Python/Virtualenvs/DL_Lasagne/lib/python2.7/site-packages')])
sys.path.append('/usr/local/lib/python2.7/site-packages/')

import matplotlib
import pylab as py
import theano
import lasagne
from lasagne.regularization import regularize_layer_params_weighted, l2, l1
from lasagne.regularization import regularize_layer_params
from lasagne.layers import batch_norm
import theano.tensor as T
import numpy as np
import pickle,gzip
import optparse
from lasagne.image import ImageDataGenerator
import time
def Get_Data_Stats(data_dir):
  ## Cifar-10 data is saved in 5 data batches.. 
  #this module will generate the mean and std. dev. stats over the entire training dataset
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
  #Will receive the training data tensor (Num images, Channels, Img_X, Img_Y) and
  #The training data labels (integers)
  while True:
    idx=np.random.choice(len(y),N)
    yield X[idx].astype('float32'),y[idx].astype('float32')

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

def cnn_network(input_var,batchnorm_bool=0, dropout_bool=0, node_type='ReLU'):
  ## The architecture that I designed for the Hyper-parameter Opt paper
  if node_type.upper()=='RELU':
    nonlinearity=lasagne.nonlinearities.rectify
  if node_type.upper()=='ELU':
    nonlinearity=lasagne.nonlinearities.elu
    
  net={}
  net['l_in']=lasagne.layers.InputLayer(shape=(None,3,32,32),input_var=input_var)
  
  if not batchnorm_bool:
    net[0]=lasagne.layers.Conv2DLayer(net['l_in'],256,3,pad=0,nonlinearity=nonlinearity)
    net[1]=lasagne.layers.Conv2DLayer(net[0],128,3,stride=(2,2),pad=1,W=lasagne.init.HeUniform(),nonlinearity=nonlinearity)
    net[2]=lasagne.layers.Conv2DLayer(net[1],256,3,stride=(1,1),pad=0,W=lasagne.init.HeUniform(),nonlinearity=nonlinearity)
    net[3]=lasagne.layers.Conv2DLayer(net[2],256,3,stride=(1,1),pad=0,W=lasagne.init.HeUniform(),nonlinearity=nonlinearity)
    net[4]=lasagne.layers.Conv2DLayer(net[3],256,3,stride=(2,2),pad=1,W=lasagne.init.HeUniform(),nonlinearity=nonlinearity)
    net[5]=lasagne.layers.Conv2DLayer(net[4],128,7,stride=(5,5),pad=2,W=lasagne.init.HeUniform(),nonlinearity=nonlinearity)
  if batchnorm_bool:
    net[0]=batch_norm(lasagne.layers.Conv2DLayer(net['l_in'],256,3,pad=0,nonlinearity=nonlinearity))
    net[1]=batch_norm(lasagne.layers.Conv2DLayer(net[0],128,3,stride=(2,2),pad=1,W=lasagne.init.HeUniform(),nonlinearity=nonlinearity))
    net[2]=batch_norm(lasagne.layers.Conv2DLayer(net[1],256,3,stride=(1,1),pad=0,W=lasagne.init.HeUniform(),nonlinearity=nonlinearity))
    net[3]=batch_norm(lasagne.layers.Conv2DLayer(net[2],256,3,stride=(1,1),pad=0,W=lasagne.init.HeUniform(),nonlinearity=nonlinearity))
    net[4]=batch_norm(lasagne.layers.Conv2DLayer(net[3],256,3,stride=(2,2),pad=1,W=lasagne.init.HeUniform(),nonlinearity=nonlinearity))
    net[5]=batch_norm(lasagne.layers.Conv2DLayer(net[4],128,7,stride=(5,5),pad=2,W=lasagne.init.HeUniform(),nonlinearity=nonlinearity))
  
  net['l_out']=lasagne.layers.DenseLayer(net[5],num_units=10,nonlinearity=lasagne.nonlinearities.softmax)
  
  return net
  
def batch_train(datagen,N_train_batches,mini_batch_size,f_train,f_val,cool_bool,lr,cool_factor=10,\
epochs=10,data_dir='/Users/sachintalathi/Work/Python/Data/cifar-10-batches-py',data_augment_bool=0):
  ##Note Validation batch is fixed 
  
  per_epoch_performance_stats=[]
  count=0
  epoch_list=[epochs/3,2*epochs/3,epochs]
  plr=theano.function([],lr)
  flr=theano.function([],lr,updates={lr:lr/cool_factor})
  
  Data_Mean,Data_Std=Get_Data_Stats(data_dir)
  
  print('Running Epochs')
  for epoch in range(epochs):
    if cool_bool:
      if epoch>epoch_list[count] and epoch<=epoch_list[count]+1:
        print ('Cool the learning rate')
        flr()
        count+=1
      
    train_loss=0
    train_acc=0
    for ind in range(N_train_batches):
      #print ('load Data Batch: %s/data_batch_%d'%(data_dir,ind+1))
      tic=time.clock()
      D=pickle.load(open('%s/data_batch_%d'%(data_dir,ind+1)))
      data=D['data'].astype('float32')
      data=(data-Data_Mean)/Data_Std
      data = data.reshape(data.shape[0], 3, 32, 32)
      labels=np.array(D['labels'])
      train_loss_per_batch=0
      train_acc_per_batch=0
      if data_augment_bool:
        train_batches=datagen.flow(data,labels,mini_batch_size) ### Generates data augmentation on the fly
      else:
        train_batches=batch_gen(data,labels,mini_batch_size) ### No data augmentation applied
      N_mini_batches=len(labels)//mini_batch_size
      for mini_batch in range(N_mini_batches):
        X,y=next(train_batches)
        loss,acc=f_train(X,y)
        train_loss_per_batch+=loss
        train_acc_per_batch+=acc
      train_loss_per_batch=train_loss_per_batch/N_mini_batches
      train_acc_per_batch=train_acc_per_batch/N_mini_batches
      toc=time.clock()
      print ('Epoch %d Data_Batch (Time) %d (%0.03f s) Learning_Rate %0.04f Train Loss (Accuracy) %.03f (%.03f)'%(epoch,ind,toc-tic,np.array(plr()),train_loss_per_batch,train_acc_per_batch))
      train_loss+=train_loss_per_batch
      train_acc+=train_acc_per_batch
    train_loss=train_loss/N_train_batches
    train_acc=train_acc/N_train_batches
    
    val_loss=0
    val_acc=0
    D_val=pickle.load(open('%s/test_batch'%(data_dir)))
    data=D_val['data'].astype('float32')
    data=(data-Data_Mean)/Data_Std
    data=data.reshape(data.shape[0], 3, 32, 32)
    labels=np.array(D_val['labels'])
    val_batches=batch_gen(data,labels,mini_batch_size)
    N_val_batches=len(labels)//mini_batch_size
    for _ in range(N_val_batches):
      X,y=next(val_batches)
      loss,acc=f_val(X,y)
      val_loss+=loss
      val_acc+=acc
    val_loss=val_loss/N_val_batches
    val_acc=val_acc/N_val_batches  
    
    loss_ratio=val_loss/train_loss
    per_epoch_performance_stats.append([train_loss,val_loss,train_acc,val_acc])
    print ('Epoch %d Learning_Rate %0.04f Train (Val) %.03f (%.03f) Accuracy'%(epoch,np.array(plr()),train_acc,val_acc))
  return per_epoch_performance_stats
  
  
if __name__=="__main__":
  parser=optparse.OptionParser()
  parser.add_option("-S", "--simple-cnn",action="store_true",dest="simple_cnn",default=False,help="Use Simple CNN Network")
  parser.add_option("-C", "--cnn",action="store_true",dest="cnn",default=False,help="Use CNN Network")
  parser.add_option("-A", "--augment",action="store_true",dest="augment",default=False,help="Augment Training Data")
  parser.add_option("-B", "--batch-norm",action="store_true",dest="batch_norm",default=False,help="Invoke Batchnorm")
  parser.add_option("--batch-size",help="Batch Size",dest="batch_size",type=int,default=64)
  parser.add_option("-c","--cool",action="store_true",dest="cool",default=False,help="Cool Learning Rate")
  parser.add_option("--cool-factor",help="Cool Factor",dest="cool_factor",type=int,default=10)
  parser.add_option("--epochs",help="epochs",dest="epochs",type=int,default=10)
  parser.add_option("--learning-rate",help="learning rate",dest="learning_rate",type=float,default=0.1)
  parser.add_option("--data-dir",help="Data Path",dest="data_dir",type=str,default='/Users/sachintalathi/Work/Python/Data')
  parser.add_option("--nonlinearity",help="Nonlinearity type",dest="nonlinearity",type=str,default='RELU')
  
  (opts,args)=parser.parse_args()
  np.random.seed(42)
  
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
  X_sym=T.tensor4()
  y_sym=T.ivector()

  #Theano definition for output probability distribution and class prediction
  if opts.simple_cnn:
    net=simple_cnn_network(X_sym,batchnorm_bool=batch_norm)
  elif opts.cnn:
    net=cnn_network(X_sym,batchnorm_bool=batch_norm)
  else:
    print ('No Network Defined')
    sys.exit(0)
          
      
  output=lasagne.layers.get_output(net['l_out'],X_sym)
  pred=output.argmax(-1)

  val_output=lasagne.layers.get_output(net['l_out'],X_sym,deterministic=True)
  val_pred=val_output.argmax(-1)
  #Theano definition for loss
  #Since output labels are integers (and not one hot encoded); we use categorical cross entropy 
  
  loss=lasagne.objectives.categorical_crossentropy(output,y_sym)
  loss=loss.mean()
  val_loss=T.mean(lasagne.objectives.categorical_crossentropy(val_output,y_sym))
  ## Add regularization term
  layers={}
  for k in net.keys():
    layers[net[k]]=0.0005
  l2_penalty = regularize_layer_params_weighted(layers, l2)  
  ## Total loss
  loss=loss+l2_penalty
  acc=T.mean(T.eq(pred,y_sym))
  
  val_loss=val_loss+l2_penalty
  val_acc=T.mean(T.eq(val_pred,y_sym))
  
  #Get params theano variables
  params=lasagne.layers.get_all_params(net['l_out'],trainable=True)

  #Gradients
  grad=T.grad(loss,params)

  #Define shared variable for learning_rate and momentum
  lr = theano.shared(np.array(opts.learning_rate, dtype=theano.config.floatX))
  momentum = theano.shared(np.array(0.9, dtype=theano.config.floatX))
  
  #Define the gradient update rule
  updates=lasagne.updates.sgd(grad,params,learning_rate=lr)
  updates=lasagne.updates.apply_momentum(updates,momentum=momentum)

  #define training function
  f_train=theano.function([X_sym,y_sym],[loss,acc],updates=updates,allow_input_downcast=True)

  #define the validation function
  f_val=theano.function([X_sym,y_sym],[val_loss,val_acc],allow_input_downcast=True)

  #define test function
  f_pred=theano.function([X_sym],pred,allow_input_downcast=True)


  #Begin Training
  tic=time.clock()
  peps=batch_train(datagen,5,32,f_train,f_val,opts.cool,lr,\
  cool_factor=opts.cool_factor,epochs=opts.epochs,data_augment_bool=opts.augment,data_dir=opts.data_dir)
  toc=time.clock()
  print(toc-tic)