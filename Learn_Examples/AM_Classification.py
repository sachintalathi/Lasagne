import sys
import matplotlib
if 'MACOSX' in matplotlib.get_backend().upper():
  matplotlib.use('TKAgg')
import pylab as py
import theano
import lasagne
from lasagne.regularization import regularize_layer_params_weighted, l2, l1
from lasagne.regularization import regularize_layer_params
import theano.tensor as T
import numpy as np
import pickle,gzip
import optparse
from lasagne.image import ImageDataGenerator
import time
import AM_Networks.helperfunctions as AH
reload(AH)

if __name__=="__main__":
  parser=optparse.OptionParser()
  parser.add_option("-S", "--simple-cnn",action="store_true",dest="simple_cnn",default=False,help="Use Simple CNN Network")
  parser.add_option("-C", "--cnn",action="store_true",dest="cnn",default=False,help="Use CNN Network")
  parser.add_option("-A", "--augment",action="store_true",dest="augment",default=False,help="Augment Training Data")
  parser.add_option("--batch-size",help="Batch Size",dest="batch_size",type=int,default=64)
  parser.add_option("-c","--cool",action="store_true",dest="cool",default=False,help="Cool Learning Rate")
  parser.add_option("--cool-factor",help="Cool Factor",dest="cool_factor",type=int,default=10)
  parser.add_option("--epochs",help="epochs",dest="epochs",type=int,default=10)
  parser.add_option("--learning-rate",help="learning rate",dest="learning_rate",type=float,default=0.1)
  parser.add_option("--data-dir",help="Data Path",dest="data_dir",type=str,default='/Users/sachintalathi/Work/Python/Data')
  parser.add_option("--nonlinearity",help="Nonlinearity type",dest="nonlinearity",type=str,default='RELU')

  (opts,args)=parser.parse_args()
  np.random.seed(42)

    ######## Define theano variables and theano functions for training/validation and testing#############
  X_sym=T.tensor4()
  y_sym=T.ivector()

  #Theano definition for output probability distribution and class prediction
  if opts.simple_cnn:
    net=AH.simple_cnn_network(X_sym)
  elif opts.cnn:
    net=AH.cnn_network(X_sym)
  else:
    print ('No Network Defined')
    sys.exit(0)
          
      
  output=lasagne.layers.get_output(net['l_out'],X_sym)
  pred=output.argmax(-1)

  val_output=lasagne.layers.get_output(net['l_out'],X_sym,deterministic=True)
  val_pred=val_output.argmax(-1)
  
  #Theano definition for loss
  loss=lasagne.objectives.categorical_crossentropy(output,y_sym)
  loss=loss.mean()
  val_loss=T.mean(lasagne.objectives.categorical_crossentropy(val_output,y_sym))
  
  ## Add regularization term
  layers={}
  for k in net.keys():
    layers[net[k]]=0.0005
  l2_penalty = regularize_layer_params_weighted(layers, l2)  
  
  ## Total loss and accuracy
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

  ## Get Data
  o=open('/Users/talathi1/Work/DataSets/AM_Project/Train_Img_List.pkl')
  train_imglist=pickle.load(o);o.close()
  o=open('/Users/talathi1/Work/DataSets/AM_Project/Train_Img_List.pkl')
  test_imglist=pickle.load(o);o.close()

  #Begin Training
  tic=time.clock()
  AH.batch_train(train_imglist,test_imglist,f_train,f_val,lr,cool_bool=False,augment_bool=False,\
    mini_batch_size=32,epochs=10,cool_factor=10,data_augment_bool=0)
  toc=time.clock()
  print(toc-tic)