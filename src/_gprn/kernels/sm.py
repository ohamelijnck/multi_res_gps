import numpy as np
import tensorflow as tf

from . import Kernel
from .. import util

class SM(Kernel):
    _id = 0
    def __init__(self, num_dimensions=1, num_components = 1, weights = [[1.0]], means = None, variances=None , var_scale=1.0, mean_scale=1.0, jitter = 0.01, init=False, mask=None):
        super(SM, self).__init__(mask)
        SM._id += 1

        self.num_dimensions = num_dimensions
        self.num_components = num_components
        self.means  = means
        self.variances = variances
        self.mean_scale = mean_scale
        self.var_scale = var_scale


    def setup(self, context):
        self.context = context
        if self.means:
            self.raw_variances = tf.Variable(self.variances, name='sm_variances_'+str(SM._id), dtype=tf.float32, trainable=True)
            self.raw_means = tf.Variable(self.means, name='sm_means_'+str(SM._id), dtype=tf.float32, trainable=True)
            self.raw_weights = tf.Variable(-tf.ones([self.num_components]), name='sm_raw_weights_'+str(SM._id), dtype=tf.float32, trainable=True)
        else:
            tf.set_random_seed(0.0)
            self.raw_variances = tf.Variable(self.var_scale*tf.random_uniform([self.num_components, self.num_dimensions], 0.1, 1, dtype=tf.float32), name='sm_variances_'+str(SM._id), dtype=tf.float32, trainable=True)
            self.raw_means = tf.Variable(self.mean_scale*tf.random_uniform([self.num_components, self.num_dimensions], 0.1, 1, dtype=tf.float32), name='sm_means_'+str(SM._id), dtype=tf.float32, trainable=True)
            self.raw_weights = tf.Variable(tf.ones([self.num_components]), name='sm_raw_weights_'+str(SM._id), dtype=tf.float32, trainable=True)

        self.parameters = [self.raw_variances, self.raw_means, self.raw_weights]
 
    @property
    def sigma(self):
        #effective sigma is the sum of weights as exp(0)*cos(0)=1
        weights = self.raw_weights
        weights = util.safe_exp(weights) 
        #weights = util.safe_exp(weights) / tf.reduce_sum(util.safe_exp(weights))
        weights = tf.Print(weights, [tf.reduce_sum(weights)], 'weights: ')
        return util.safe_log(tf.reduce_sum(weights))

    def _kernel(self, _X1, _X2, jitter=False, debug=False, include_dimensions=None):
        #1/mean = period
        #1/variance = lengthscale
        variances = util.safe_exp(self.raw_variances)
        means = util.safe_exp(self.raw_means)
        weights = self.raw_weights

        weights = util.safe_exp(weights) 

        variances = tf.Print(variances, [variances, means, weights], 'sm params: ')
        #weights = util.safe_exp(weights) / tf.reduce_sum(util.safe_exp(weights))

        #[1, N1] - [N2, 1] = [N1, N2]
        X1 = tf.transpose(tf.expand_dims(_X1, -1), perm=[1, 0, 2])  # D x N1 x 1
        X2 = tf.expand_dims(tf.transpose(_X2, perm=[1, 0]), -2)  # D x N2 x 1
        T = tf.transpose(tf.abs(tf.subtract(X1, X2)), perm=[0, 1, 2])  # D x N1 x N2
        T = tf.clip_by_value(T, 0, 1e8)
        T2 = tf.square(T)

        cos_term = tf.tensordot(a=means, b=T, axes=1)
        exp_term = tf.tensordot(a=variances, b=T2, axes=1)
    
        res = tf.multiply(
            util.safe_exp(tf.scalar_mul(-2*np.pi*np.pi, exp_term)),
            tf.cos(tf.scalar_mul(2*np.pi, cos_term))
        )

        k = tf.tensordot(a=weights, b=res, axes=1)

        k = tf.Print(k, [exp_term], 'exp_term: ', summarize=300)
        k = tf.Print(k, [cos_term], 'cos_term: ', summarize=300)
        k = tf.Print(k, [T2], 'T2: ', summarize=300)
        k = tf.Print(k, [res, k], 'res: k: ')

        if jitter is True:
            k =  util.add_jitter(k, self.context.jitter)  


        return k

 
    def get_parameters(self):
        return self.parameters

