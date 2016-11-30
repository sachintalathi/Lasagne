import glob
import numpy as np
import cv2
import os,sys
import itertools as it
import theano
import theano.tensor as T
import time
np.random.seed(100)
@async_prefetch
def batch_gen(X,y,batch_size=32):
	if len(y)<batch_size:
		batch_size=len(y)
	idx=0
	idx_end=batch_size

	while True:
		ind=np.arange(idx,idx_end)
		yield X[ind],y[ind]
		idx+=batch_size
		idx_end+=batch_size
		if idx_end>len(y):
			idx_end=len(y)
			break

def Get_Data_Mean(Img_List,img_size=[224,224]):
  ## Get mean image stats
  datagen=AM_Data_Generator(Img_List,img_size=img_size)
  sum_img=np.zeros((3,img_size[0],img_size[1]))
  for _ in it.count():
    try:
    	X,y=next(datagen)
    except StopIteration:
    	return sum_img/len(Img_List)
    tmp_X=X.sum(axis=0)
    sum_img+=tmp_X
  #return mean_img/len(Img_List)

####################  Fast data generator using threading and decorators #######################
import functools
import Queue
import threading
 
def async_prefetch_wrapper(iterable, buffer=1):
	"""
	wraps an iterater such that it produces items in the background
	uses a bounded queue to limit memory consumption
	"""
	done = object()
	def worker(q,it):
		for item in it:
			q.put(item)
		q.put(done)
	# launch a thread to fetch the items in the background
	queue = Queue.Queue(buffer)
	it = iter(iterable)
	thread = threading.Thread(target=worker, args=(queue, it))
	thread.daemon = True
	thread.start()
	# pull the items of the queue as requested
	while True:
		item = queue.get()
		if item == done:
			return
		else:
			yield item
 
def async_prefetch(func):
	"""
	decorator to make generator functions fetch items in the background
	"""
	@functools.wraps(func)
	def wrapper(*args, **kwds):
		return async_prefetch_wrapper( func(*args, **kwds) )
	return wrapper


import threading
class LockedIterator(object):
    def __init__(self, it):
        self.lock = threading.Lock()
        self.it = it.__iter__()

    def __iter__(self): return self

    def next(self):
        self.lock.acquire()
        try:
            return self.it.next()
        finally:
            self.lock.release()

def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return LockedIterator(f(*a, **kw))
    return g

@async_prefetch
def AM_Data_Generator(Img_List,img_size=[224,224],batch_size=32):
	if not os.path.isfile(Img_List[0]):
		print 'Data path does not exit'
		sys.exit(0)
	idx=0
	if len(Img_List)<batch_size:
		batch_size=len(Img_List)
	idx_end=batch_size
	
	Img_Array=np.array(Img_List)
	#X=np.zeros((batch_size,224,224,3));y=np.zeros(batch_size)
	while True:
		X=[];y=[];
		ind=np.arange(idx,idx_end)
		subset_Img_Array=Img_Array[ind]
		for i in range(len(subset_Img_Array)):
			img=cv2.imread(subset_Img_Array[i])
			label_str=subset_Img_Array[i].split('/')[-1].split('_')[0]
			label=0 if 'Bad' in label_str else (1 if 'Good' in label_str else 2)
			#print label_str,label,subset_Img_Array[i]
			img_resize=cv2.resize(img,(img_size[0],img_size[1]),interpolation = cv2.INTER_LINEAR)
			img_resize_rescale=1.0*img_resize/img_resize.max()	
			#X[i,:,:,:]=img_resize_rescale
			X.append(img_resize_rescale)
			y.append(label)
		yield np.array(X).swapaxes(3,1).swapaxes(3,2).astype('float32'),np.array(y).astype('float32')
		if idx_end==len(Img_List):
			return
		idx=idx+batch_size
		idx_end=idx_end+batch_size
		if idx_end>len(Img_List):
			idx_end=len(Img_List)

import Queue
class BackgroundGenerator(threading.Thread):
    def __init__(self, generator):
        threading.Thread.__init__(self)
        self.queue = Queue.Queue(1)
        self.generator = generator
        self.daemon = True
        self.start()

    def run(self):
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
            next_item = self.queue.get()
            if next_item is None:
                 raise StopIteration
            return next_item



		
import lasagne
from lasagne.layers import InputLayer, DropoutLayer,FlattenLayer
from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import BatchNormLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import ElemwiseSumLayer
from lasagne.layers import DenseLayer
from lasagne.nonlinearities import rectify, softmax
from lasagne.image import ImageDataGenerator
from lasagne.layers import batch_norm

def simple_cnn_network(input_var,num_units=3,img_channels=3, img_size=(224,224),batchnorm_bool=False, dropout_bool=False, node_type='ReLU'):
  #Simple cnn network for prototyping
	if node_type.upper()=='RELU':
		nonlinearity=lasagne.nonlinearities.rectify
	if node_type.upper()=='ELU':
		nonlinearity=lasagne.nonlinearities.elu
  
	net={}
	net['l_in']=lasagne.layers.InputLayer((None,img_channels,img_size[0],img_size[1]))

	if batchnorm_bool:
		net[0]=batch_norm(lasagne.layers.Conv2DLayer(net['l_in'],16,3,W=lasagne.init.HeUniform(),pad=1,nonlinearity=nonlinearity))
  	else:
  		net[0]=lasagne.layers.Conv2DLayer(net['l_in'],16,3,W=lasagne.init.HeUniform(),pad=1,nonlinearity=nonlinearity)
  	net['l_out']=lasagne.layers.DenseLayer(net[0],num_units=num_units,nonlinearity=lasagne.nonlinearities.softmax)
  	return net

def cnn_network(input_var,num_units=3,img_channels=3, img_size=(224,224),batchnorm_bool=False, dropout_bool=False, node_type='ReLU'):
	if node_type.upper()=='RELU':
		nonlinearity=lasagne.nonlinearities.rectify
	if node_type.upper()=='ELU':
  		nonlinearity=lasagne.nonlinearities.elu
	net={}
	net['l_in']=lasagne.layers.InputLayer(shape=(None,img_channels,img_size[0],img_size[1]),input_var=input_var)
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
	net['l_out']=lasagne.layers.DenseLayer(net[5],num_units=num_units,nonlinearity=lasagne.nonlinearities.softmax)
	return net

def nin_net(num_units=3,num_channels=3,img_size=[32,32]):
	### Trained model weights available for cifar10
	net = {}
	net['input'] = InputLayer((None, num_channels, img_size[0], img_size[1]))
	net['conv1'] = ConvLayer(net['input'], num_filters=192, filter_size=5, pad=2, flip_filters=False)
	net['cccp1'] = ConvLayer(net['conv1'], num_filters=160, filter_size=1, flip_filters=False)
	net['cccp2'] = ConvLayer(net['cccp1'], num_filters=96, filter_size=1, flip_filters=False)
	net['pool1'] = PoolLayer(net['cccp2'], pool_size=3, stride=2, mode='max', ignore_border=False)
	net['drop3'] = DropoutLayer(net['pool1'], p=0.5)
	net['conv2'] = ConvLayer(net['drop3'], num_filters=192, filter_size=5, pad=2, flip_filters=False)
	net['cccp3'] = ConvLayer(net['conv2'], num_filters=192, filter_size=1, flip_filters=False)
	net['cccp4'] = ConvLayer(net['cccp3'], num_filters=192, filter_size=1, flip_filters=False)
	net['pool2'] = PoolLayer(net['cccp4'], pool_size=3, stride=2, mode='average_exc_pad', ignore_border=False)
	net['drop6'] = DropoutLayer(net['pool2'], p=0.5)
	net['conv3'] = ConvLayer(net['drop6'], num_filters=192, filter_size=3, pad=1, flip_filters=False)
	net['cccp5'] = ConvLayer(net['conv3'], num_filters=192, filter_size=1, flip_filters=False)
	net['cccp6'] = ConvLayer(net['cccp5'], num_filters=num_units, filter_size=1, flip_filters=False)
	net['pool3'] = PoolLayer(net['cccp6'], pool_size=8, mode='average_exc_pad', ignore_border=False)
	net['l_out'] = lasagne.layers.FlattenLayer(net['pool3'])
	#net['l_out']=lasagne.layers.DenseLayer(net['flat'],num_units=num_units,nonlinearity=lasagne.nonlinearities.softmax)
	return net

def batch_train(train_imglist,test_imglist,f_train,f_val,lr,cool_bool=False,img_size=[224,224],\
	mini_batch_size=32,epochs=10,cool_factor=10,data_augment_bool=False):

	batch_size=32
	## Data Augment Generator
	augmentgen = ImageDataGenerator(
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

	def batcheval(train_batches,f_train,N_mini_batches):
		train_loss_per_batch=0
		train_acc_per_batch=0
		for mini_batch in range(N_mini_batches):
			try:
				X,y=next(train_batches)
			except StopIteration:
				break
			#ttic=time.clock()
			loss,acc=f_train(X,y)
			#ttoc=time.clock()
			#print 'Time for training per mini-batch:',ttoc-ttic
			train_loss_per_batch+=loss
			train_acc_per_batch+=acc
		return train_loss_per_batch/N_mini_batches,train_acc_per_batch/N_mini_batches

	per_epoch_performance_stats=[]
	count=0
	epoch_list=[epochs/3,2*epochs/3,epochs]
	plr=theano.function([],lr)
	flr=theano.function([],lr,updates={lr:lr/cool_factor})

	print('Running Epochs')
	for epoch in range(epochs):
		## Cooling protocol
		if cool_bool:
		  if epoch>epoch_list[count] and epoch<=epoch_list[count]+1:
			print ('Cool the learning rate')
			flr()
			count+=1

		## Data generator
		train_datagen=AM_Data_Generator(train_imglist,batch_size=batch_size,img_size=img_size)
		
		## Data mean
		if epoch==0:
			mean_X=Get_Data_Mean(train_imglist,img_size=img_size)
	
		train_loss=0
		train_acc=0
		tic=time.clock()
		for count_iter in it.count():
			#print epoch, count_iter
			data_tic=time.clock()
			try:
				data,labels=next(train_datagen)
			except StopIteration:
				break
			data_toc=time.clock()
			print 'Time to read batch data from AM Generator:',data_toc-data_tic
			data=data[:]-mean_X

			if data_augment_bool:
				N_mini_batches=len(labels)//mini_batch_size
				train_batches=augmentgen.flow(data,labels,mini_batch_size) ### Generates data augmentation on the fly
				tlpb,tapb=batcheval(train_batches,f_train,N_mini_batches)
			else:
				loss,acc=f_train(data,labels)
				tlpb=loss
				tapb=acc
				#train_batches=batch_gen(data,labels,mini_batch_size) ### No data augmentation applied
		  
			
			
			# ttic=time.clock()
			# tlpb,tapb=batcheval(train_batches,f_train,N_mini_batches)
			# ttoc=time.clock()
			# print '           Time to train on data:',ttoc-ttic
			train_loss+=tlpb
			train_acc+=tapb

		#print count_iter
		toc=time.clock()	
		train_loss=train_loss/count_iter
		train_acc=train_acc/count_iter
		print ('Epoch %d (%0.05f s) Learning_Rate %0.04f Train Loss (Accuracy) %.03f (%.03f)'%(epoch,toc-tic,np.array(plr()),train_loss,train_acc))
		
		### Computing validation loss per epoch
		val_loss=0
		val_acc=0
		test_datagen=AM_Data_Generator(test_imglist,batch_size=batch_size,img_size=img_size)
		for count_iter in it.count():
			try:
				data,labels=next(test_datagen)
			except StopIteration:
				break
			data=data[:]-mean_X
			test_batches=batch_gen(data,labels,mini_batch_size)
			N_mini_batches=len(labels)//mini_batch_size
			#vlpb,vapb=batcheval(test_batches,f_val,N_mini_batches)
			#val_loss+=vlpb
			#val_acc+=vapb
		val_loss=val_loss/count_iter
		val_acc=val_acc/count_iter  

		#loss_ratio=val_loss/train_loss
		per_epoch_performance_stats.append([epoch,train_loss,val_loss,train_acc,val_acc])
		print ('Epoch %d Learning_Rate %0.04f Train (Val) %.03f (%.03f) Accuracy'%(epoch,np.array(plr()),train_acc,val_acc))
	return per_epoch_performance_stats
