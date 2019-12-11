import numpy as np
import tensorflow as tf

from . import Kernel
from .. import util
from .. import Parameters

class NN(Kernel):
    _id = -1
    def __init__(self, num_dimensions=1, mask=None):
        super(SE, self).__init__(mask)
        SE._id += 1
        self.id = SE._id

        self.num_dimensions = num_dimensions
        self.setup_flag = False

    def setup(self, context):
        self.setup_flag = True
        self.context = context
        self.parameters = self.context.parameters

        self.sigmas = self.parameters.create(name='nn_sigmas_'+str(self.id), init=self._sigma, trainable=True, scope=Parameters.HYPER_SCOPE)

        self.parameters = [self.sigmas]

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
            ls = self.s(ls, include_dimensions)
        else:
            include_dimensions = []

        X1 = tf.transpose(tf.expand_dims(_X1, -1), perm=[1, 0, 2])  # D x N1 x 1
        X2 = tf.expand_dims(tf.transpose(_X2, perm=[1, 0]), -2)  # D x N2 x 1
        T = tf.transpose(tf.subtract(X1, X2), perm=[0, 1, 2])  # D x N1 x N2


        val = tf.exp(-tf.square(T)/(2*tf.expand_dims(tf.expand_dims(tf.square(ls), -1), -1)))
        val = tf.reduce_prod(val, axis=0)
        val = sigma*val

        
        if jitter is True:
            val =  util.add_jitter(val, self.context.jitter) 

        val = tf.Print(val, [sigma], 'sigma: ')
        return val

    def get_parameters(self):
        return self.parameters

