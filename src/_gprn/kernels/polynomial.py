import numpy as np
import tensorflow as tf

from . import Kernel
from .. import util

class Polynomial(Kernel):
    _id = 0
    def __init__(self, num_dimensions=1, a=1.0, d = 1.0,  mask=None):
        super(Polynomial, self).__init__(mask)
        Polynomial._id += 1
        self.num_dimensions = num_dimensions
        self._a = a
        self._d = d
        #self.sigma = tf.Variable(sigma, name='Polynomial_sigma_'+str(polynomial._id), dtype=tf.float32, trainable=True)

    def setup(self, context):
        self.context = context
        self.a = tf.Variable(self._a, name='polynomial_a_'+str(Polynomial._id), dtype=tf.float32, trainable=True)
        self.d = tf.constant(self._d, name='polynomial_d_'+str(Polynomial._id), dtype=tf.float32)
        self.parameters = [self.a]



    def _kernel(self, X1, X2, jitter=False, debug=False, include_dimensions=None):
        """
        inputs:
            X1 \in R^{N \cross D}
            X2 \in R^{M \cross D}
        output:
            K \in R^{N \cross M}
        """

        val = tf.pow(tf.matmul(X1, X2, transpose_b = True) + util.var_postive(self.a), self.d)

        if jitter is True:
            val =  util.add_jitter(val, self.context.jitter)  
        return val

    def get_parameters(self):
        return self.parameters


