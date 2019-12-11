import numpy as np
import tensorflow as tf

from . import Kernel
from .. import util

class Periodic(Kernel):
    _id = 0
    def __init__(self, num_dimensions=1, sigma=1.0, period=1, length_scale=1.0,ARD=True, jitter = 0.1, mask=None):
        super(Periodic, self).__init__(mask)
        Periodic._id += 1

        self.length_scales = tf.Variable(length_scale*tf.ones(shape=[num_dimensions]), name='periodic_length_scale_'+str(Periodic._id), dtype=tf.float32, trainable=True)  
        self.sigma = tf.Variable(sigma, name='periodic_sigma_'+str(Periodic._id), dtype=tf.float32, trainable=True)
        self.period = tf.Variable(period, name='periodic_period_'+str(Periodic._id), dtype=tf.float32, trainable=True)

        self.parameters = [self.sigma, self.length_scales]
        self.white = jitter

    def _kernel(self, X1, X2, jitter=False, debug=False):
        """
        inputs:
            X1 \in R^{N \cross D}
            X2 \in R^{M \cross D}
        output:
            K \in R^{N \cross M}
        """

        sigma = self.sigma
        period = self.period
        length_scales = self.length_scales


        p=0
        x1_p = tf.matmul(tf.expand_dims(X1[:, p], 1), tf.expand_dims(tf.ones([tf.shape(X2[:,p])[0]]), -1), transpose_b=True)
        x2_p = tf.matmul(tf.expand_dims(tf.ones([tf.shape(X1[:, p])[0]]), -1), tf.expand_dims(X2[:, p], -1), transpose_b=True)

        x1 = x1_p
        x2 = x2_p
        
        r = np.pi*tf.abs(x1-x2)/period
        r = tf.square(tf.sin(r)/length_scales)
        val = (sigma**2)*util.safe_exp(-2*r)

        if jitter is True:
            return val + self.white * tf.eye(tf.shape(X1)[0])
        return val


    def get_parameters(self):
        return self.parameters

