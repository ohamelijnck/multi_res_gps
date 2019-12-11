import tensorflow as tf
import numpy as np

import math
from .. import util

class Parameters(object):
    _ID_ = 0
    VARIATIONAL_SCOPE = 'variational_params'
    HYPER_SCOPE = 'hyper_params'
    def __init__(self, context):
        Parameters._ID_ += 1
        self.id = Parameters._ID_
        self.params = {}
        self.context = context

    def get(self, name):
        return self.params[name]

    def save(self, name, var):
        self.params[name] = var
        return var

    def create(self, name, init, trainable=True, scope='parameters'):
        with tf.variable_scope(scope, reuse=None):
            var = tf.get_variable(initializer=init, dtype=tf.float32, name=name, trainable=trainable)

        self.save(name, var)
        return var

    def load_parameters(self):
        self.load_posterior_component_weights()

    def load_posterior_component_weights(self):
        q_weights_raw = self.get('q_raw_weights')

        q_weights = util.safe_exp(q_weights_raw) / tf.reduce_sum(util.safe_exp(q_weights_raw))
        self.save('q_weights', q_weights)

    #goes through each element in var, and stacks in the same order
    def load_from_array(self, var, shape, n, fn):
        if shape.shape[0] == 1:
            return fn(n, var)

        total_covs = []
        for i in range(shape[0]):
            cov = self.load_from_array(var[i, :], shape[1:], n, fn)
            total_covs.append(cov)
        return total_covs


    def load_posterior_covariance(self, name, from_name, shape, n):
        raw_cov = self.get(from_name)
        covs = self.load_from_array(raw_cov, np.array(shape), n, lambda n, var: util.covar_to_mat(n, var, self.context.use_diag_covar, self.context.jitter))
        covs = tf.stack(covs)
        self.save(name, covs)
        return covs

    def load_posterior_cholesky(self, name, from_name, shape, n):
        raw_cov = self.get(from_name)
        covs = self.load_from_array(raw_cov, np.array(shape), n, lambda n, var: util.vec_to_lower_triangle_matrix(n, var))
        covs = tf.stack(covs)
        self.save(name, covs)
        return covs




