import numpy as np
import tensorflow as tf

from . import Kernel
from .. import util

class Matern52(Kernel):
    _id = 0
    def __init__(self, num_dimensions=1, num_components = 1, length_scale = 0.5, means = None, variances=None , var_scale=1.0, mean_scale=1.0, jitter = 0.01, init=False, mask=None):
        super(Matern52, self).__init__(mask)
        Matern52._id += 1
        ARD=False

        if ARD:
            self.length_scales = tf.Variable(length_scale*tf.ones(shape=[num_dimensions]), name='matern32_length_scale_'+str(Matern52._id), dtype=tf.float32, trainable=True)  
        else:
            self.length_scales = tf.Variable([length_scale], name='matern32_length_scale_'+str(Matern52._id), dtype=tf.float32, trainable=True)

        self.white = jitter
   
    def _kernel(self, _X1, _X2, jitter=False, debug=False):
        X1 = tf.transpose(tf.expand_dims(_X1, -1), perm=[1, 0, 2])  # D x N1 x 1
        X2 = tf.expand_dims(tf.transpose(_X2, perm=[1, 0]), -2)  # D x N2 x 1
        r = tf.transpose(tf.abs(tf.subtract(X1, X2)), perm=[0, 1, 2])  # D x N1 x N2
        r = tf.clip_by_value(r, 0, 1e8)

        k =  (1+(tf.scalar_mul(np.sqrt(5), r)/self.length_scales) + tf.scalar_mul(5, tf.square(r))/(3*tf.square(self.length_scales)))*util.safe_exp(-tf.scalar_mul(np.sqrt(5), r)/self.length_scales)

        k = tf.reduce_prod(k, axis=0)

        if jitter:
            k =  k + self.white * tf.eye(tf.shape(_X1)[0])

        return k


 
    def get_parameters(self):
        return self.parameters


