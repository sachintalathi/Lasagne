import numpy as np
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from .. import init
from .. import random
from .. import nonlinearities

from .base import Layer
from .quantize import hard_sigmoid,binarization

__all__ = [
    "DenseLayer",
    "NINLayer",
    "qDenseLayer",
]


class DenseLayer(Layer):
    """
    lasagne.layers.DenseLayer(incoming, num_units,
    W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.),
    nonlinearity=lasagne.nonlinearities.rectify, **kwargs)

    A fully connected layer.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape

    num_units : int
        The number of units of the layer

    W : Theano shared variable, expression, numpy array or callable
        Initial value, expression or initializer for the weights.
        These should be a matrix with shape ``(num_inputs, num_units)``.
        See :func:`lasagne.utils.create_param` for more information.

    b : Theano shared variable, expression, numpy array, callable or ``None``
        Initial value, expression or initializer for the biases. If set to
        ``None``, the layer will have no biases. Otherwise, biases should be
        a 1D array with shape ``(num_units,)``.
        See :func:`lasagne.utils.create_param` for more information.

    nonlinearity : callable or None
        The nonlinearity that is applied to the layer activations. If None
        is provided, the layer will be linear.

    Examples
    --------
    >>> from lasagne.layers import InputLayer, DenseLayer
    >>> l_in = InputLayer((100, 20))
    >>> l1 = DenseLayer(l_in, num_units=50)

    Notes
    -----
    If the input to this layer has more than two axes, it will flatten the
    trailing axes. This is useful for when a dense layer follows a
    convolutional layer, for example. It is not necessary to insert a
    :class:`FlattenLayer` in this case.
    """
    def __init__(self, incoming, num_units, W=init.GlorotUniform(),
                 b=init.Constant(0.), nonlinearity=nonlinearities.rectify,
                 **kwargs):
        super(DenseLayer, self).__init__(incoming, **kwargs)
        self.nonlinearity = (nonlinearities.identity if nonlinearity is None
                             else nonlinearity)

        self.num_units = num_units

        num_inputs = int(np.prod(self.input_shape[1:]))

        self.W = self.add_param(W, (num_inputs, num_units), name="W")
        if b is None:
            self.b = None
        else:
            self.b = self.add_param(b, (num_units,), name="b",
                                    regularizable=False)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units)

    def get_output_for(self, input, **kwargs):
        if input.ndim > 2:
            # if the input has more than two dimensions, flatten it into a
            # batch of feature vectors.
            input = input.flatten(2)

        activation = T.dot(input, self.W)
        if self.b is not None:
            activation = activation + self.b.dimshuffle('x', 0)
        return self.nonlinearity(activation)

#######Try alternative way to define qDense Layer ########
class qDenseLayer(DenseLayer):
    
    def __init__(self, incoming, num_units,stochastic = True,H=1,W_LR_scale="Glorot",quantization=None, **kwargs):
        
        self.stochastic=stochastic
        self.quantization=quantization
        self._srng = RandomStreams(random.get_rng().randint(1, 2147462579))
        self.num_units = num_units
        num_inputs = int(np.prod(np.shape(incoming)[1:]))
        
        if H=="Glorot":
            self.H=np.float32(np.sqrt(1.5/ (num_inputs + num_units)))
        elif H=="He":
            self.H=np.float32(np.sqrt(2./num_inputs))
        else:
            self.H=np.float32(1.0)
        
        if W_LR_scale=="Glorot":
            self.W_LR_scale=np.float32(1./(np.sqrt(1.5/ (num_inputs + num_units))))
        elif W_LR_scale=="He":
            self.W_LR_scale=np.float32(1./(np.sqrt(2./num_inputs)))
        else:
            self.W_LR_scale=np.float32(1.)

        self._srng = RandomStreams(random.get_rng().randint(1, 2147462579)) ## setting the seed
        
        if self.quantization:
            super(qDenseLayer, self).__init__(incoming, num_units, W=init.Uniform((-self.H,self.H)), **kwargs)
            self.params[self.W]=set(['quantize'])
        else:
            super(qDenseLayer, self).__init__(incoming, num_units, **kwargs)
        
    def get_output_for(self, input, deterministic=False, **kwargs):
        self.Wb = binarization(self.W,self.H,deterministic,self.stochastic,self._srng,self.quantization)
        Wr = self.W
        self.W = self.Wb
        rvalue = super(qDenseLayer, self).get_output_for(input, **kwargs)
        self.W = Wr
        return rvalue
        
    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units)



###################################
class NINLayer(Layer):
    """
    lasagne.layers.NINLayer(incoming, num_units, untie_biases=False,
    W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.),
    nonlinearity=lasagne.nonlinearities.rectify, **kwargs)

    Network-in-network layer.
    Like DenseLayer, but broadcasting across all trailing dimensions beyond the
    2nd.  This results in a convolution operation with filter size 1 on all
    trailing dimensions.  Any number of trailing dimensions is supported,
    so NINLayer can be used to implement 1D, 2D, 3D, ... convolutions.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape

    num_units : int
        The number of units of the layer

    untie_biases : bool
        If false the network has a single bias vector similar to a dense
        layer. If true a separate bias vector is used for each trailing
        dimension beyond the 2nd.

    W : Theano shared variable, expression, numpy array or callable
        Initial value, expression or initializer for the weights.
        These should be a matrix with shape ``(num_inputs, num_units)``,
        where ``num_inputs`` is the size of the second dimension of the input.
        See :func:`lasagne.utils.create_param` for more information.

    b : Theano shared variable, expression, numpy array, callable or ``None``
        Initial value, expression or initializer for the biases. If set to
        ``None``, the layer will have no biases. Otherwise, biases should be
        a 1D array with shape ``(num_units,)`` for ``untie_biases=False``, and
        a tensor of shape ``(num_units, input_shape[2], ..., input_shape[-1])``
        for ``untie_biases=True``.
        See :func:`lasagne.utils.create_param` for more information.

    nonlinearity : callable or None
        The nonlinearity that is applied to the layer activations. If None
        is provided, the layer will be linear.

    Examples
    --------
    >>> from lasagne.layers import InputLayer, NINLayer
    >>> l_in = InputLayer((100, 20, 10, 3))
    >>> l1 = NINLayer(l_in, num_units=5)

    References
    ----------
    .. [1] Lin, Min, Qiang Chen, and Shuicheng Yan (2013):
           Network in network. arXiv preprint arXiv:1312.4400.
    """
    def __init__(self, incoming, num_units, untie_biases=False,
                 W=init.GlorotUniform(), b=init.Constant(0.),
                 nonlinearity=nonlinearities.rectify, **kwargs):
        super(NINLayer, self).__init__(incoming, **kwargs)
        self.nonlinearity = (nonlinearities.identity if nonlinearity is None
                             else nonlinearity)

        self.num_units = num_units
        self.untie_biases = untie_biases

        num_input_channels = self.input_shape[1]

        self.W = self.add_param(W, (num_input_channels, num_units), name="W")
        if b is None:
            self.b = None
        else:
            if self.untie_biases:
                biases_shape = (num_units,) + self.output_shape[2:]
            else:
                biases_shape = (num_units,)
            self.b = self.add_param(b, biases_shape, name="b",
                                    regularizable=False)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units) + input_shape[2:]

    def get_output_for(self, input, **kwargs):
        # cf * bc01... = fb01...
        out_r = T.tensordot(self.W, input, axes=[[0], [1]])
        # input dims to broadcast over
        remaining_dims = range(2, input.ndim)
        # bf01...
        out = out_r.dimshuffle(1, 0, *remaining_dims)

        if self.b is None:
            activation = out
        else:
            if self.untie_biases:
                # no broadcast
                remaining_dims_biases = range(1, input.ndim - 1)
            else:
                remaining_dims_biases = ['x'] * (input.ndim - 2)  # broadcast
            b_shuffled = self.b.dimshuffle('x', 0, *remaining_dims_biases)
            activation = out + b_shuffled

        return self.nonlinearity(activation)
