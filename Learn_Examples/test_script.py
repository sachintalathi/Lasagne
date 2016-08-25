import theano
import theano.tensor as T
import numpy as np
import lasagne

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
srng = RandomStreams(lasagne.random.get_rng().randint(1, 2147462579))

def hard_sigmoid(x):
    return T.clip((x+1.)/2.,0,1)

def prob_sigmoid(x):
  return 0.5*(x/T.max(x)+1)
  
# The binarization function
def binarization(W,H,deterministic=False,stochastic=False,srng=None,quantization=None):
    if quantization==None:
        print("no quantization")
        Wb = W
    elif quantization.upper()=='BINARY':
        Wb = hard_sigmoid(W/H)
        # # Stochastic BinaryConnect
        if stochastic and not deterministic:
            print("stoch")
            Wb = T.cast(srng.binomial(n=1, p=Wb, size=T.shape(Wb)), theano.config.floatX)
        else:
            print("det")
            Wb = T.round(Wb)
        # # 0 or 1 -> -1 or 1
        # print H,alpha.eval()
        Wb = T.cast(T.switch(Wb,H,-H), theano.config.floatX)
    
    elif quantization.upper()=='ROUND':
        Wb=T.round(W)
    
    elif quantization.upper()=='SCALE_ROUND':
        alpha=T.max(W)
        Wb=alpha*T.round(W/alpha)
    
    elif quantization.upper()=='SIGN':
        alpha=T.max(W)
        Wb=alpha*T.sgn(W)
    
    elif quantization.upper()=='POW':
        alpha=T.max(W)
        beta=np.array(2.0,dtype='float32')
        Wb=alpha*(W/alpha)**beta*T.sgn(W)
    
    elif quantization.upper()=='STOCM':
        stocW=W*srng.uniform(low=0.5,high=2.,size=T.shape(W))
        print stocW.eval()
        Wb = hard_sigmoid(W/H)
        # # Stochastic BinaryConnect
        if stochastic and not deterministic:
            print("stoch")
            prob_Wb = T.cast(srng.binomial(n=1, p=Wb, size=T.shape(Wb)), theano.config.floatX)
        else:
            print("det")
            prob_Wb = T.round(Wb)

        Wb = T.cast(T.switch(prob_Wb,1,-1), theano.config.floatX)    
        Wb=stocW*Wb
        print Wb.eval()
    else:
        print 'Error in specifying quantizatoin type'
        print 'Allowed Values: {Round, Binary, None}'
        sys.exit(0)
    return Wb,prob_Wb,stocW

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


## Draw Figure 3 Layer MLP network for MNIST
from lasagne.layers.quantize import binarization
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
srng = RandomStreams(lasagne.random.get_rng().randint(1, 2147462579))
def Get_Hist(V,i,srng,quantize):
    val=V[i].flat[0:]
    if quantize==None:
        [N,I]=np.histogram(val,bins=1000,range=(-val.max()-0.5,val.max()+0.5))
    else:
        wb=binarization(val,1.,deterministic=True,stochastic=False,srng=srng,quantization=quantize)
        [N,I]=np.histogram(wb.eval(),bins=1000,range=(-wb.eval().max()-0.5,wb.eval().max()+0.5))
    return N,I


#Assume V is weights and R is Results


quantize='Binary'

def Plot_Stats_Quantize_MNist(V,pt,pv,srng,quantize):
    py.figure();

    py.subplot(521);
    py.plot(pt[:,1]);
    py.hold('on');py.plot(pv[:,1])
    py.legend(['Train-Loss: %.2f'%pt[:,1].min(),'Val-Loss: %.2f'%pv[:,1].min()])

    py.subplot(522);
    py.plot(pt[:,2]);
    py.hold('on');py.plot(pv[:,2])
    py.legend(['Train-Error: %.2f'%pt[:,2].min(),'Val-Error: %.2f'%pv[:,2].min()])

    py.subplot(523)
    [N,I]=Get_Hist(V,0,srng,None)
    py.plot(I[0:-1],10.**3*N/sum(N))
    py.ylabel('$p(w^{1})$')

    py.subplot(524)
    [N,I]=Get_Hist(V,0,srng,quantize)
    py.plot(I[0:-1],10.**3*N/sum(N))
    py.ylabel('$p(Q(w)^{1})$')

    py.subplot(525)
    [N,I]=Get_Hist(V,5,srng,None)
    py.plot(I[0:-1],10.**3*N/sum(N))
    py.ylabel('$p(w^{2})$')

    py.subplot(526)
    [N,I]=Get_Hist(V,5,srng,quantize)
    py.plot(I[0:-1],10.**3*N/sum(N))
    py.ylabel('$p(Q(w)^{2})$')

    py.subplot(527)
    [N,I]=Get_Hist(V,10,srng,None)
    py.plot(I[0:-1],10.**3*N/sum(N))
    py.ylabel('$p(w^{3})$')

    py.subplot(528)
    [N,I]=Get_Hist(V,10,srng,quantize)
    py.plot(I[0:-1],10.**3*N/sum(N))
    py.ylabel('$p(Q(w)^{3})$')

    py.subplot(529)
    [N,I]=Get_Hist(V,15,srng,None)
    py.plot(I[0:-1],10.**3*N/sum(N))
    py.ylabel('$p(w^{4})$')
    py.xlabel('w')

    py.subplot(5,2,10)
    [N,I]=Get_Hist(V,15,srng,quantize)
    py.plot(I[0:-1],10.**3*N/sum(N))
    py.ylabel('$p(Q(w)^{4})$')
    py.xlabel('Q(w)')


## Read From Training Log
LogFile='/prj/neo_lv/user/stalathi/Lasagne_Models/LogFiles/Log_Cifar10_Pow_Again.log'
txt=open(LogFile)
Data=txt.readlines()
Scores=[]
for i in range(len(Data)):
    if 'Error' in Data[i]:
        train_error=float32(Data[i].split('Error)')[1].split('(')[0])
        test_error=float32(Data[i].split('Error)')[1].split('(')[1].split(')')[0])
        Scores.append([train_error,test_error])

txt.close()