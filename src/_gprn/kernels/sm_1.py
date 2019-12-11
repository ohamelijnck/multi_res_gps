import numpy as np
import tensorflow as tf

from . import Kernel
from .. import util

class SM(Kernel):
    _id = 0
    def __init__(self, num_dimensions=1, num_components = 1, weights = [[1.0]], means = [[2.0]], variances=[[1.0]] , white = 0.01, init=False):
        SM._id += 1

        self.num_dimensions = num_dimensions
        self.num_components = num_components

        self.means = tf.Variable(tf.random_uniform([self.num_components, self.num_dimensions], 0, 2, tf.float32, seed=1), name='sm_means_'+str(SM._id), dtype=tf.float32, trainable=True)
        self.variances = tf.Variable(tf.random_uniform([self.num_components, self.num_dimensions], 0, 2, tf.float32, seed=1), name='sm_variances_'+str(SM._id), dtype=tf.float32, trainable=True)
        self.raw_weights = tf.Variable(tf.random_uniform([self.num_components], 0, 1, tf.float32, seed=1), name='sm_weights_'+str(SM._id), dtype=tf.float32, trainable=True)

        #self.means = tf.Variable(means, name='sm_means_'+str(SM._id), dtype=tf.float32, trainable=True)
        #self.variances = tf.Variable(variances, name='sm_variances_'+str(SM._id), dtype=tf.float32, trainable=True)
        #self.raw_weights = tf.Variable(weights, name='sm_raw_weights_'+str(SM._id), dtype=tf.float32, trainable=True)

        self.weights = util.safe_exp(self.raw_weights) / tf.reduce_sum(util.safe_exp(self.raw_weights))

        self.parameters = [self.means, self.variances, self.raw_weights]
        self.white = white

    def kernel(self, X1, X2, jitter=False):
        x1 = [0.0 for p in range(self.num_dimensions)]
        x2 = [0.0 for p in range(self.num_dimensions)]
        for p in range(self.num_dimensions):
            x1_p = tf.matmul(tf.expand_dims(X1[:, p], 1), tf.expand_dims(tf.ones([tf.shape(X2[:,p])[0]]), -1), transpose_b=True)
            x2_p = tf.matmul(tf.expand_dims(tf.ones([tf.shape(X1[:, p])[0]]), -1), tf.expand_dims(X2[:, p], -1), transpose_b=True)
            x1[p] = x1_p
            x2[p] = x2_p

        total_sum = 0.0
        for q in range(self.num_components):
            total_prod = 1.0

            for p in range(self.num_dimensions):
                T = x1[p]-x2[p]
                total_prod *= util.safe_exp(-2*np.pi*tf.pow(T, 2)*tf.pow(self.variances[q,p], 2))

            total_sum += self.weights[q]*total_prod*tf.cos(2*np.pi*T*self.means[q,:])
        k = total_sum

        if jitter is True:
            k = k + self.white * tf.eye(tf.shape(X1)[0])

        return k
    def get_parameters(self):
        return self.parameters

