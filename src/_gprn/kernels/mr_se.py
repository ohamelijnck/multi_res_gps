import numpy as np
import tensorflow as tf

from . import Kernel
from .. import util
from .. import Parameters

class MR_SE(Kernel):
    _id = -1
    def __init__(self, num_dimensions=1, sigma=0.0, length_scale=1.0,ARD=True, mask=None, train=True):
        super(MR_SE, self).__init__(mask)
        MR_SE._id += 1
        self.id = MR_SE._id

        self.ARD = ARD
        self.length_scale = length_scale
        self._sigma = sigma
        self.num_dimensions = num_dimensions
        self.setup_flag = False
        self.train = train

    def setup(self, context):
        self.setup_flag = True
        self.context = context
        self.parameters = self.context.parameters

        trainable_flag=self.train
        if self.ARD:
            self.length_scales = self.parameters.create(name='se_length_scale_'+str(self.id), init=self.length_scale*tf.ones(shape=[self.num_dimensions]), trainable=trainable_flag, scope=Parameters.HYPER_SCOPE)
        else:
            self.length_scales = self.parameters.create(name='se_length_scale_'+str(self.id), init=[self.length_scale], trainable=trainable_flag, scope=Parameters.HYPER_SCOPE)

        self.sigma = self.parameters.create(name='se_sigma_'+str(self.id), init=self._sigma, trainable=trainable_flag, scope=Parameters.HYPER_SCOPE)

        self.parameters = [self.sigma, self.length_scales]

    def _kernel_non_vectorised(self, _X1, _X2, jitter=False, debug=False, include_dimensions=None):
        """
        inputs:
            X1 \in R^{N \cross D}
            X2 \in R^{M \cross D}
        output:
            K \in R^{N \cross M}
        """
        sigma = util.var_postive(self.sigma)
        #sigma = 1.0
        ls = util.var_postive(self.length_scales)

        if include_dimensions is not None:
            ls = ls[0]

            _X1 = tf.expand_dims(_X1[:, include_dimensions[0]], -1) #NS x 1
            _X2 = tf.expand_dims(_X2[:, include_dimensions[0]], -1) #NS x x 1
        else:
            include_dimensions = []

        _X1 = tf.Print(_X1, [tf.shape(_X1)], '_X1: ')
        _X1 = tf.Print(_X1, [tf.shape(_X2)], '_X2: ')


        X1 = tf.transpose(tf.expand_dims(_X1, -1), perm=[1, 0, 2])  # D x N1 x 1
        X2 = tf.expand_dims(tf.transpose(_X2, perm=[1, 0]), -2)  # D x N2 x 1
        T = tf.transpose(tf.subtract(X1, X2), perm=[0, 1, 2])  # D x N1 x N2


        val = tf.exp(-tf.square(T)/(2*tf.expand_dims(tf.expand_dims(tf.square(ls), -1), -1)))
        val = tf.reduce_prod(val, axis=0)
        val = sigma*val

        
        if jitter is True:
            val =  util.add_jitter(val, self.context.jitter) 

        return val

    def _kernel(self, _X1, _X2, jitter=False, debug=False, include_dimensions=None):
        """
        inputs:
            X1 \in R^{N \cross D}
            X2 \in R^{M \cross D}
        output:
            K \in R^{N \cross M}
        """
        sigma = util.var_postive(self.sigma)
        #sigma = 1.0
        ls = util.var_postive(self.length_scales)

        if include_dimensions is not None:
            #ls = self.s(ls, include_dimensions)
            #print('include_dimensions: ', include_dimensions[0])
            _X1 = tf.expand_dims(_X1[:, :, include_dimensions[0]], -1) #N x S x 1
            _X2 = tf.expand_dims(_X2[:, :, include_dimensions[0]], -1) #N x S x 1
        else:
            include_dimensions = []

        X1 = _X1 # 
        X2 = tf.transpose(_X2, perm=[0, 2, 1]) #D x 1 x N2
        T = tf.transpose(tf.subtract(X1, X2), perm=[0, 1, 2])  # D x N1 x N2

        val = tf.exp(-tf.square(T)/(2*tf.expand_dims(tf.expand_dims(tf.square(ls), -1), -1)))
        val = sigma*val

        
        if jitter is True:
            #val =  util.add_jitter(val, self.context.jitter) 

            val =  val+(self.context.jitter*tf.eye(tf.shape(val)[1]))[None, :, :]


        return val

    def get_parameters(self):
        return self.parameters
