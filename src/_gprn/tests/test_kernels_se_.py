#To run tests: python -m unittest *

import sys
sys.path.insert(0,'../../')

import unittest
from unittest.mock import MagicMock
from unittest.mock import Mock

import tensorflow as tf
import numpy as np
import math
import scipy as sp

import gprn

import GPy

class Test(tf.test.TestCase):
    #called by unittest before each test is ran
    def setUp(self):
        print("\n In method", self._testMethodName)
        tf.reset_default_graph()
        self.decimal = 5

    def get_context(self):
        #setup context
        context = gprn.context.ContextFactory().create()
        context.num_latent = 1
        context.num_outputs = 1
        context.num_weights = 1
        context.num_components = 1
        context.jitter = 1e-5
        context.num_latent_process = 1
        context.num_latent_process = 1
        context.use_diag_covar_flag= False
        context.use_diag_covar= False
        return context

    def get_data(self, N):
        x = np.expand_dims(np.random.random(N), -1).astype(np.float32)
        y = np.expand_dims(np.random.random(N), -1).astype(np.float32)

        data = gprn.Dataset()
        data.add_source_dict({
            'x': x,
            'y': y,
            'batch_size': None
        })
        return data

    def get_parameters(self, context, data):
        parameters = gprn.Parameters(context)
        num_inducing = data.get_num_inducing(0)
        sigma = np.int(num_inducing*(num_inducing+1)/2)

        np.random.seed(10)

        self.mean_u = np.random.random([context.num_components, context.num_latent, num_inducing]).astype(np.float32)
        self.mean_v = np.random.random([context.num_components, context.num_outputs, context.num_latent, num_inducing]).astype(np.float32)
        self.covar_u = np.random.random([context.num_components, context.num_latent, sigma]).astype(np.float32)
        self.covar_v = np.random.random([context.num_components, context.num_outputs, context.num_latent, sigma]).astype(np.float32)

        weights = np.array([1]).astype(np.float32)


        parameters.save(name='inducing_locations_0', var=tf.constant(data.get_inducing_points_from_source(0)))

        parameters.save(name='q_means_u_0', var=tf.constant(self.mean_u))
        parameters.save(name='q_means_v_0', var=tf.constant(self.mean_v))
        parameters.save(name='q_covars_u_0_raw', var=tf.constant(self.covar_u))
        parameters.save(name='q_covars_v_0_raw', var=tf.constant(self.covar_v))

        parameters.load_posterior_covariance(name='q_covars_u_0', from_name='q_covars_u_0_raw', shape=[context.num_components, context.num_latent, sigma], n=num_inducing)
        parameters.load_posterior_covariance(name='q_covars_v_0', from_name='q_covars_v_0_raw', shape=[context.num_components, context.num_outputs, context.num_latent, sigma], n=num_inducing)

        parameters.load_posterior_cholesky(name='q_cholesky_u_0', from_name='q_covars_u_0_raw', shape=[context.num_components, context.num_latent, sigma], n=num_inducing)
        parameters.load_posterior_cholesky(name='q_cholesky_v_0', from_name='q_covars_v_0_raw', shape=[context.num_components, context.num_outputs, context.num_latent, sigma], n=num_inducing)


        parameters.create(name='q_raw_weights', init=tf.constant(weights), trainable=False)
        parameters.load_posterior_component_weights()

        return parameters

    def random_psd_matrix(self, n):
        jit = 1e-5
        chol_k = np.tril(np.random.random([n, n]), -1)
        k = np.matmul(chol_k, chol_k.T)+np.eye(n)*jit
        return k.astype(np.float32)


    def test__kernel__same_x__same_as_gpy(self):
        n = 50
        ls = 2.0
        sig = 0.3
        #Arrange
        X = np.expand_dims(np.linspace(-10, 10, 100), -1).astype(np.float32)
        ans = GPy.kern.RBF(1, lengthscale=ls, variance=sig).K(X, X)

        context = self.get_context()
        parameters = gprn.Parameters(context)
        context.parameters = parameters


        K = gprn.kernels.SE(num_dimensions=1, sigma=np.log(sig).astype(np.float32), length_scale=np.log(ls).astype(np.float32))
        K.setup(context)

        with tf.Session() as session:
            session.run(tf.global_variables_initializer())

            #Act
            res = K._kernel(tf.constant(X), tf.constant(X), jitter=False)


            #Assert
            np.testing.assert_almost_equal(res.eval(),ans, decimal=self.decimal)

