import sys
if '/Users/sachintalathi/Work/Python/Virtualenvs/DL_Lasagne/lib/python2.7/site-packages' in sys.path:
  del(sys.path[sys.path.index('/Users/sachintalathi/Work/Python/Virtualenvs/DL_Lasagne/lib/python2.7/site-packages')])
sys.path.append('/usr/local/lib/python2.7/site-packages/')
import matplotlib
import pylab as py
import theano
import lasagne ## current quirk of my install because CUDA is not supported
import lasagne
from lasagne.regularization import regularize_layer_params_weighted, l2, l1
from lasagne.regularization import regularize_layer_params
from lasagne.layers import batch_norm
import theano.tensor as T
import numpy as np
import pickle,gzip
import optparse


#Nice way to read random set of mini-batch data
#Define a Batch Generator Function

def batch_gen(X,y,N):
  while True:
    idx=np.random.choice(len(y),N)
    yield X[idx].astype('float32'),y[idx].astype('float32')  

#Simple Logistic Network    
def logistic_network(dim=784):
  #dim is the input data dimension in vector form
  net={}
  net['l_in']=lasagne.layers.InputLayer((None,dim))
  net['l_d']=lasagne.layers.DropoutLayer(net['l_in'],p=.1)
  net['l_out']=lasagne.layers.DenseLayer(net['l_d'],num_units=10,nonlinearity=lasagne.nonlinearities.softmax)
  return net

#Simple MLP Network
def mlp_network(dim=784,num_hidden_layers=1,num_hidden_nodes_per_layer=10,dropout=0.5):
  net={}
  net['l_in']=lasagne.layers.InputLayer((None,dim))
  net['l_d']=lasagne.layers.DropoutLayer(net['l_in'],p=.1)
  for i in range(num_hidden_layers):
    if i==0:
      net[i]=batch_norm(lasagne.layers.DenseLayer(net['l_d'],num_units=num_hidden_nodes_per_layer,nonlinearity=lasagne.nonlinearities.rectify))
    else:
      net[i]=batch_norm(lasagne.layers.DenseLayer(net[i-1],num_units=num_hidden_nodes_per_layer,nonlinearity=lasagne.nonlinearities.rectify))
    net[i+1]=lasagne.layers.DropoutLayer(net[i],p=dropout)
  net['l_out']=lasagne.layers.DenseLayer(net[2*num_hidden_layers-1],num_units=10,nonlinearity=lasagne.nonlinearities.softmax)
  return net

def cnn_network(dim=784,num_filters=3,filter_size=3):
  #Defining a simple one layer CNN network
  net={}
  net['l_in']=lasagne.layers.InputLayer((None,dim))
  net[0]=lasagne.layers.ReshapeLayer(net['l_in'],(-1,1,28,28))
  net[1]=batch_norm(lasagne.layers.Conv2DLayer(net[0],num_filters,filter_size,pad=1,nonlinearity=lasagne.nonlinearities.rectify))
  net['l_out']=lasagne.layers.DenseLayer(net[1],num_units=10,nonlinearity=lasagne.nonlinearities.softmax)
  return net
  
def batch_train(train_batches,val_batches,N_batches,N_val_batches,f_train,f_val,cool_bool,lr,cool_factor=10,epochs=10):
  per_epoch_performance_stats=[]
  count=0
  epoch_list=[epochs/3,2*epochs/3,epochs]
  plr=theano.function([],lr)
  flr=theano.function([],lr,updates={lr:lr/cool_factor})
  
  for epoch in range(epochs):
    if cool_bool:
      if epoch>epoch_list[count] and epoch<=epoch_list[count]+1:
        print ('Cool the learning rate')
        flr()
        count+=1
      
    train_loss=0
    train_acc=0
    for _ in range(N_batches):
      X,y=next(train_batches)
      loss,acc=f_train(X,y)
      train_loss+=loss
      train_acc+=acc
    train_loss=train_loss/N_batches
    train_acc=train_acc/N_batches
    
    val_loss=0
    val_acc=0
    for _ in range(N_val_batches):
      X,y=next(val_batches)
      loss,acc=f_val(X,y)
      val_loss+=loss
      val_acc+=acc
    val_loss=val_loss/N_val_batches
    val_acc=val_acc/N_val_batches  
    
    loss_ratio=val_loss/train_loss
    per_epoch_performance_stats.append([train_loss,val_loss,train_acc,val_acc])
    print ('Epoch %d Learning_Rate %0.04f Train (Val) %.03f (%.03f) Accuracy'%(epoch,plr(),train_acc,val_acc))
  return per_epoch_performance_stats



## Main Program
if __name__=="__main__":
  parser=optparse.OptionParser()
  parser.add_option("-L", "--logistic",action="store_true",dest="logistic",default=False,help="Use Logistic Network")
  parser.add_option("-M", "--mlp",action="store_true",dest="mlp",default=False,help="Use MLP Network")
  parser.add_option("-C", "--cnn",action="store_true",dest="cnn",default=False,help="Use CNN Network")
  parser.add_option("--batch-size",help="Batch Size",dest="batch_size",type=int,default=64)
  parser.add_option("-c","--cool",action="store_true",dest="cool",default=False,help="Cool Learning Rate")
  parser.add_option("--cool-factor",help="Cool Factor",dest="cool_factor",type=int,default=10)
  parser.add_option("--epochs",help="epochs",dest="epochs",type=int,default=10)
  parser.add_option("--learning-rate",help="learning rate",dest="learning_rate",type=float,default=0.1)
  parser.add_option("--data-dir",help="Data Path",dest="data_dir",type=str,default='/Users/sachintalathi/Work/Python/Data')
  (opts,args)=parser.parse_args()
  np.random.seed(42)
  
  ####### Read data and Generate Data Batch Generator ############
  train,val,test=pickle.load(gzip.open('%s/mnist.pkl.gz'%opts.data_dir))
  X_train,y_train=train
  X_val,y_val=val
  X_test,y_test=test

  #Define batchsize
  batch_size=opts.batch_size
  N_Batches=len(X_train)//batch_size
  N_Val_Batches=len(X_val)//batch_size

  ##Minibatch generator for training and validation set
  train_batches=batch_gen(X_train,y_train,batch_size)
  val_batches=batch_gen(X_val,y_val,batch_size)

  ######## Define theano variables and theano functions for training/validation and testing#############
  X_sym=T.matrix()
  y_sym=T.ivector()

  #Theano definition for output probability distribution and class prediction
  if opts.logistic:
    net=logistic_network()
  elif opts.mlp:
    net=mlp_network()
  elif opts.cnn:
    net=cnn_network()
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
  #updates=lasagne.updates.apply_momentum(updates,momentum=momentum)

  #define training function
  f_train=theano.function([X_sym,y_sym],[loss,acc],updates=updates,allow_input_downcast=True)

  #define the validation function
  f_val=theano.function([X_sym,y_sym],[val_loss,val_acc],allow_input_downcast=True)

  #define test function
  f_pred=theano.function([X_sym],pred,allow_input_downcast=True)


  #Begin Training
  peps=batch_train(train_batches,val_batches,N_Batches,N_Val_Batches,f_train,f_val,opts.cool,lr,cool_factor=opts.cool_factor,epochs=opts.epochs)