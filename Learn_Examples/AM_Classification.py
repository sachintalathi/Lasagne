import sys,os
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
import glob
import pickle,gzip
import optparse
from lasagne.image import ImageDataGenerator
import time
import AM_Networks.helperfunctions as AH
reload(AH)

def parse_list(option, opt, value, parser):
  setattr(parser.values, option.dest, value.split(','))

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
  parser.add_option("--data-dir",help="Data Path",dest="data_dir",type=str,default='/Users/talathi1/Work/Python/Data')
  parser.add_option("--save-dir",help="Save model path",dest="save_dir",type=str,default='')
  parser.add_option("--memo",help="memo",dest="memo",type=str,default='')
  parser.add_option("--model-file",help="pretrained model file path",dest="model_file",type=str,default='')
  parser.add_option("--nonlinearity",help="Nonlinearity type",dest="nonlinearity",type=str,default='RELU')
  parser.add_option("--train",action="store_true",dest="train",default=False,help="Train the network")
  parser.add_option("-a",action="store_true",dest="analyze",default=False,help="Analyze the network")
  parser.add_option('--crop-size',type='str',action='callback',callback=parse_list,dest="crop_size")

  (opts,args)=parser.parse_args()
  np.random.seed(100)
  crop_size=map(int,opts.crop_size)
    ######## Define theano variables and theano functions for training/validation and testing#############
  X_sym=T.tensor4()
  y_sym=T.ivector()

  #Theano definition for output probability distribution and class prediction
  if opts.simple_cnn:
    net=AH.simple_cnn_network(X_sym,img_size=crop_size,batchnorm_bool=True)
  elif opts.cnn:
    #net=AH.cnn_network(X_sym,img_size=crop_size,batchnorm_bool=True)
    net=AH.cnn_network(X_sym,img_size=crop_size,batchnorm_bool=True)
  else:
    print ('No Network Defined')
    sys.exit(0)
  
  ### Load pretrained model
  if len(opts.model_file)!=0:
    [peps,values]=pickle.load(open(opts.model_file))
    lasagne.layers.set_all_param_values(net['l_out'],values)
    
      
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
  home_dir=os.environ['HOME']
  imglist=glob.glob('%s/Work/DataSets/AM_Project/Resized_256x256/*'%home_dir)
  np.random.shuffle(imglist)
  train_imglist=imglist[0:32*256]
  test_imglist=imglist[32*256:]

  #Begin Training
  if opts.train:
    tic=time.clock()
    #peps=AH.batch_train(train_imglist,test_imglist,f_train,f_val,lr,cool_bool=opts.cool,\
      #mini_batch_size=128,epochs=opts.epochs,cool_factor=10,data_augment_bool=opts.augment,img_size=crop_size)
    peps=AH.batch_train_with_ListImageGenerator(train_imglist,test_imglist,f_train,f_val,lr,cool_bool=opts.cool,img_size=crop_size,\
      mini_batch_size=128,epochs=opts.epochs,cool_factor=10,shuffle=False)
    toc=time.clock()
    print 'Training Time:', (toc-tic), 's'  
    if len(opts.save_dir)!=0:
      save_file='%s/%s.pkl'%(opts.save_dir,opts.memo)
      values=lasagne.layers.get_all_param_values(net['l_out'])
      o=open(save_file,'wb')
      pickle.dump([peps,values],o)
      o.close()

  ## Analyze:
  if opts.analyze:
    layers=lasagne.layers.get_all_layers(net['l_out'])
    input_var=layers[0].input_var
    feat={};fval={}
    for k in net:
      feat[k]=lasagne.layers.get_output(net[k],deterministic=True)
      fval[k]=theano.function([input_var],feat[k])

