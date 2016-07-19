import theano
import theano.tensor as T
import numpy as np

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import lasagne

def hard_sigmoid(x):
    return T.clip((x+1.)/2.,0,1)

def prob_sigmoid(x):
  return 0.5*(x/T.max(x)+1)
  
# The binarization function
def binarization(W,H,binary=True,deterministic=False,stochastic=False,srng=None):
    # (deterministic == True) <-> test-time <-> inference-time
    if not binary or (deterministic and stochastic):
        # print("not binary")
        Wb = W
    else:
        # [-1,1] -> [0,1]
        Wb = hard_sigmoid(W/H)
        # Stochastic BinaryConnect
        if stochastic:
            # print("stoch")
            Wb = T.cast(srng.binomial(n=1, p=Wb, size=T.shape(Wb)), theano.config.floatX)
        # Deterministic BinaryConnect (round to nearest)
        else:
            # print("det")
            Wb = T.round(Wb)
        # 0 or 1 -> -1 or 1
        Wb = T.cast(T.switch(Wb,H,-H), theano.config.floatX)
    return Wb


### To test how to create a function of shared variable
## and then again return back the shared variable
## Important for implementing the binary connect paper idea
class Test(object):
  def __init__(self,W):
    self.W=theano.shared(W,name='W')
  
  def convert(self):
    srng = RandomStreams(lasagne.random.get_rng().randint(1, 2147462579))
    H=np.array(T.max(self.W).eval(),dtype='float32')
    Wr=self.W
    self.Wb=binarization(self.W,H,stochastic=True,srng=srng)
    self.W=self.Wb
    f=self.Wb
    self.W=Wr
    return f

w=np.array(np.random.randn(2,2),dtype='float32')
test=Test(w)

### Test init method in lasagne
import theano
import theano.tensor as T
import lasagne as la
reload(la)
input_var=T.tensor4()
net={}
net['l_in']=la.layers.InputLayer(shape=(None,3,32,32),input_var=input_var)
c1=la.layers.Conv2DLayer(net['l_in'],256,3,pad=0,W=la.init.Normal())
c2=la.layers.Conv2DLayer(c1,128,3,stride=(2,2),pad=1,W=la.init.GlorotUniform())
c3=la.layers.qConv2DLayer(c1,128,3,stride=(2,2),pad=1,W=la.init.GlorotUniform(),binary=False)
l_in = la.layers.InputLayer((100, 20))
l1=la.layers.DenseLayer(l_in, num_units=10)
l2=la.layers.qDenseLayer(l_in,num_units=10)
out=la.layers.DenseLayer(l2,num_units=10,nonlinearity=la.nonlinearities.softmax)

## Compare method in original binaryconnect code
import theano
import theano.tensor as T
import lasagne as la
reload(la)
import lasagne.binary_connect as binary_connect

#MLP Network
#using my mods to lasagne layer def
l_in = la.layers.InputLayer((100, 20))
l1=la.layers.DenseLayer(l_in, num_units=10)
l2=la.layers.qDenseLayer(l_in,num_units=10)
out=la.layers.DenseLayer(l2,num_units=10,nonlinearity=la.nonlinearities.softmax)

## Binary Connect paper layer def
mlp_in = lasagne.layers.InputLayer((100,20))
mlp_l1=la.layers.DenseLayer(mlp_in, num_units=10)
mlp_l2 = binary_connect.DenseLayer(mlp_in, binary=True,stochastic=True,nonlinearity=lasagne.nonlinearities.identity,num_units=10)
mlp_out=la.layers.DenseLayer(mlp_l2,num_units=10,nonlinearity=la.nonlinearities.softmax)


#CNN network
#My implementation
l_in=lasagne.layers.InputLayer((None,3,32,32))
c1=la.layers.Conv2DLayer(l_in,16,3,W=lasagne.init.HeUniform(),pad=1,nonlinearity=la.nonlinearities.rectify)
c2=la.layers.qConv2DLayer(l_in,num_filters=16,filter_size=3,pad=1,nonlinearity=la.nonlinearities.rectify,binary=True,stochastic=True)
cout=la.layers.DenseLayer(c2,num_units=10,nonlinearity=la.nonlinearities.softmax)

#Binary conect paper implementation
