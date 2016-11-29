!wget https://www.dropbox.com/s/blrajqirr1p31v0/cifar10_nin.caffemodel 
!wget https://gist.githubusercontent.com/ebenolson/91e2cfa51fdb58782c26/raw/b015b7403d87b21c6d2e00b7ec4c0880bbeb1f7e/model.prototxt

import caffe
net_caffe = caffe.Net('model.prototxt', 'cifar10_nin.caffemodel', caffe.TEST)

import lasagne
from lasagne.layers import InputLayer, DropoutLayer, FlattenLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.utils import floatX

def nin_net():
	net = {}
	net['input'] = InputLayer((None, 3, 32, 32))
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
	net['cccp6'] = ConvLayer(net['cccp5'], num_filters=10, filter_size=1, flip_filters=False)
	net['pool3'] = PoolLayer(net['cccp6'], pool_size=8, mode='average_exc_pad', ignore_border=False)
	net['output'] = lasagne.layers.FlattenLayer(net['pool3'])
	return net

net=nin_net()
layers_caffe = dict(zip(list(net_caffe._layer_names), net_caffe.layers)

for name, layer in net.items():
    try:
        layer.W.set_value(layers_caffe[name].blobs[0].data)
        layer.b.set_value(layers_caffe[name].blobs[1].data)       
    except AttributeError:
        continue

## Try it out
import numpy as np
import pickle
import pylab as plt
plt.ion()
!wget https://s3.amazonaws.com/lasagne/recipes/pretrained/cifar10/cifar10.npz
data = np.load('cifar10.npz')
prob = np.array(lasagne.layers.get_output(net['output'], floatX(data['whitened']), deterministic=True).eval())
predicted = np.argmax(prob, 1)

## save model
values=lasagne.layers.get_all_param_values(net['output'])
o=open('~/Work/Python/Models/nin_cifar10.pkl','wb')
pickle.dump(values,o)
o.close()
