import glob
import numpy as np
import cv2
import os,sys
import itertools as it
import theano
import theano.tensor as T
import time
np.random.seed(100)

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

def Get_Data_Mean(Img_List):
  ## Get mean image stats
  datagen=AM_Data_Generator(Img_List)
  sum_img=np.zeros((3,224,224))
  for _ in it.count():
    try:
    	X,y=next(datagen)
    except StopIteration:
    	return sum_img/len(Img_List)
    tmp_X=X.sum(axis=0)
    sum_img+=tmp_X
  #return mean_img/len(Img_List)

def AM_Data_Generator(Img_List,batch_size=32):
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
			img_resize=cv2.resize(img,(224,224),interpolation = cv2.INTER_LINEAR)
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
			
import lasagne
from lasagne.layers import InputLayer
from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import BatchNormLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import ElemwiseSumLayer
from lasagne.layers import DenseLayer
from lasagne.nonlinearities import rectify, softmax
from lasagne.image import ImageDataGenerator
from lasagne.layers import batch_norm

def simple_cnn_network(input_var,num_units=3,img_channels=3, img_size=(224,224),batchnorm_bool=0, dropout_bool=0, node_type='ReLU'):
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

def cnn_network(input_var,num_units=3,img_channels=3, img_size=(224,224),batchnorm_bool=0, dropout_bool=0, node_type='ReLU'):
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


## Residual Net pre-trained on Imagenet
def build_simple_block(incoming_layer, names,
                       num_filters, filter_size, stride, pad,
                       use_bias=False, nonlin=rectify):
    net = []
    net.append((
            names[0],
            ConvLayer(incoming_layer, num_filters, filter_size, pad, stride,
                      flip_filters=False, nonlinearity=None) if use_bias
            else ConvLayer(incoming_layer, num_filters, filter_size, stride, pad, b=None,
                           flip_filters=False, nonlinearity=None)
        ))

    net.append((
            names[1],
            BatchNormLayer(net[-1][1])
        ))
    if nonlin is not None:
        net.append((
            names[2],
            NonlinearityLayer(net[-1][1], nonlinearity=nonlin)
        ))

    return dict(net), net[-1][0]


def build_residual_block(incoming_layer, ratio_n_filter=1.0, ratio_size=1.0, has_left_branch=False,
                         upscale_factor=4, ix=''):
    simple_block_name_pattern = ['res%s_branch%i%s', 'bn%s_branch%i%s', 'res%s_branch%i%s_relu']

    net = {}

    # right branch
    net_tmp, last_layer_name = build_simple_block(
        incoming_layer, map(lambda s: s % (ix, 2, 'a'), simple_block_name_pattern),
        int(lasagne.layers.get_output_shape(incoming_layer)[1]*ratio_n_filter), 1, int(1.0/ratio_size), 0)
    net.update(net_tmp)

    net_tmp, last_layer_name = build_simple_block(
        net[last_layer_name], map(lambda s: s % (ix, 2, 'b'), simple_block_name_pattern),
        lasagne.layers.get_output_shape(net[last_layer_name])[1], 3, 1, 1)
    net.update(net_tmp)

    net_tmp, last_layer_name = build_simple_block(
        net[last_layer_name], map(lambda s: s % (ix, 2, 'c'), simple_block_name_pattern),
        lasagne.layers.get_output_shape(net[last_layer_name])[1]*upscale_factor, 1, 1, 0,
        nonlin=None)
    net.update(net_tmp)

    right_tail = net[last_layer_name]
    left_tail = incoming_layer

    # left branch
    if has_left_branch:
        net_tmp, last_layer_name = build_simple_block(
            incoming_layer, map(lambda s: s % (ix, 1, ''), simple_block_name_pattern),
            int(lasagne.layers.get_output_shape(incoming_layer)[1]*4*ratio_n_filter), 1, int(1.0/ratio_size), 0,
            nonlin=None)
        net.update(net_tmp)
        left_tail = net[last_layer_name]

    net['res%s' % ix] = ElemwiseSumLayer([left_tail, right_tail], coeffs=1)
    net['res%s_relu' % ix] = NonlinearityLayer(net['res%s' % ix], nonlinearity=rectify)

    return net, 'res%s_relu' % ix


def build_model():
    net = {}
    net['input'] = InputLayer((None, 3, 224, 224))
    sub_net, parent_layer_name = build_simple_block(
        net['input'], ['conv1', 'bn_conv1', 'conv1_relu'],
        64, 7, 3, 2, use_bias=True)
    net.update(sub_net)
    net['pool1'] = PoolLayer(net[parent_layer_name], pool_size=3, stride=2, pad=0, mode='max', ignore_border=False)
    block_size = list('abc')
    parent_layer_name = 'pool1'
    for c in block_size:
        if c == 'a':
            sub_net, parent_layer_name = build_residual_block(net[parent_layer_name], 1, 1, True, 4, ix='2%s' % c)
        else:
            sub_net, parent_layer_name = build_residual_block(net[parent_layer_name], 1.0/4, 1, False, 4, ix='2%s' % c)
        net.update(sub_net)

    block_size = list('abcd')
    for c in block_size:
        if c == 'a':
            sub_net, parent_layer_name = build_residual_block(
                net[parent_layer_name], 1.0/2, 1.0/2, True, 4, ix='3%s' % c)
        else:
            sub_net, parent_layer_name = build_residual_block(net[parent_layer_name], 1.0/4, 1, False, 4, ix='3%s' % c)
        net.update(sub_net)

    block_size = list('abcdef')
    for c in block_size:
        if c == 'a':
            sub_net, parent_layer_name = build_residual_block(
                net[parent_layer_name], 1.0/2, 1.0/2, True, 4, ix='4%s' % c)
        else:
            sub_net, parent_layer_name = build_residual_block(net[parent_layer_name], 1.0/4, 1, False, 4, ix='4%s' % c)
        net.update(sub_net)

    block_size = list('abc')
    for c in block_size:
        if c == 'a':
            sub_net, parent_layer_name = build_residual_block(
                net[parent_layer_name], 1.0/2, 1.0/2, True, 4, ix='5%s' % c)
        else:
            sub_net, parent_layer_name = build_residual_block(net[parent_layer_name], 1.0/4, 1, False, 4, ix='5%s' % c)
        net.update(sub_net)
    net['pool5'] = PoolLayer(net[parent_layer_name], pool_size=7, stride=1, pad=0,
                             mode='average_exc_pad', ignore_border=False)
    net['fc1000'] = DenseLayer(net['pool5'], num_units=1000, nonlinearity=None)
    net['prob'] = NonlinearityLayer(net['fc1000'], nonlinearity=softmax)

    return net

def batch_train(train_imglist,test_imglist,f_train,f_val,lr,cool_bool=False,augment_bool=False,\
	mini_batch_size=32,epochs=10,cool_factor=10,data_augment_bool=0):

	batch_size=256
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
			loss,acc=f_train(X,y)
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
		train_datagen=AM_Data_Generator(train_imglist,batch_size=batch_size)
		
		## Data mean
		if epoch==0:
			mean_X=Get_Data_Mean(train_imglist)
	
		train_loss=0
		train_acc=0
		tic=time.clock()
		for count_iter in it.count():
			print epoch, count_iter
			try:
				data,labels=next(train_datagen)
			except StopIteration:
				break
			data=data[:]-mean_X

			if data_augment_bool:
				train_batches=datagen.flow(data,labels,mini_batch_size) ### Generates data augmentation on the fly
			else:
				train_batches=batch_gen(data,labels,mini_batch_size) ### No data augmentation applied
		  
			N_mini_batches=len(labels)//mini_batch_size
			tlpb,tapb=batcheval(train_batches,f_train,N_mini_batches)
			train_loss+=tlpb
			train_acc+=tapb

		print count_iter
		toc=time.clock()	
		train_loss=train_loss/count_iter
		train_acc=train_acc/count_iter
		print ('Epoch %d (%0.05f s) Learning_Rate %0.04f Train Loss (Accuracy) %.03f (%.03f)'%(epoch,toc-tic,np.array(plr()),train_loss,train_acc))
		
		### Computing validation loss per epoch
		val_loss=0
		val_acc=0
		test_datagen=AM_Data_Generator(test_imglist,batch_size)
		for count_iter in it.count():
			try:
				data,labels=next(test_datagen)
			except StopIteration:
				break
			data=data[:]-mean_X
			test_batches=batch_gen(data,labels,mini_batch_size)
			N_mini_batches=len(labels)//mini_batch_size
			vlpb,vapb=batcheval(test_batches,f_val,N_mini_batches)
			val_loss+=vlpb
			val_acc+=vapb
		val_loss=val_loss/count_iter
		val_acc=val_acc/count_iter  

		loss_ratio=val_loss/train_loss
		per_epoch_performance_stats.append([epoch,train_loss,val_loss,train_acc,val_acc])
		print ('Epoch %d Learning_Rate %0.04f Train (Val) %.03f (%.03f) Accuracy'%(epoch,np.array(plr()),train_acc,val_acc))
	return per_epoch_performance_stats
