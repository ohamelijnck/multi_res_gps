import numpy as np
import tensorflow as tf

from . import Kernel
from .. import util

class Constant(Kernel):
    _id = 0
    def __init__(self, sigma=1.0, mask=None):
        super(Constant, self).__init__(mask)
        Constant._id += 1
        self.sigma_init = sigma

    def setup(self, context):
        self.sigma = tf.Variable(self.sigma_init, name='constant_sigma_'+str(Constant._id), dtype=tf.float32, trainable=False)
        self.white = context.jitter

    def _kernel(self, X1, X2, jitter=False, debug=False):
        """
        inputs:
            X1 \in R^{N \cross D}
            X2 \in R^{M \cross D}
        output:
            K \in R^{N \cross M}
        """
        sig = util.var_postive(self.sigma)
        val =  sig*tf.matmul(tf.expand_dims(tf.ones([tf.shape(X1)[0]]), -1), tf.expand_dims(tf.ones([tf.shape(X2)[0]]), -1), transpose_b=True)
        #val =  (self.sigma**2)*tf.matmul(tf.expand_dims(tf.ones([tf.shape(X1)[0]]), -1), tf.expand_dims(tf.ones([tf.shape(X2)[0]]), -1), transpose_b=True)

        if jitter is True:
            return val + self.white * tf.eye(tf.shape(X1)[0])
        return val

    def get_parameters(self):
        return [self.sigma]

