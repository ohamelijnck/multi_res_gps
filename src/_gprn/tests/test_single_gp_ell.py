#To run tests: python -m unittest *

import sys
sys.path.insert(0,'../../')

import unittest
from unittest.mock import MagicMock

import tensorflow as tf
import numpy as np
import math
import scipy as sp

import gprn
from gprn import sparsity

class Test(tf.test.TestCase):
    #called by unittest before each test is ran
    def setUp(self):
        print("\n In method", self._testMethodName)
        tf.reset_default_graph()
        self.decimal = 3

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
        context.seed= 0
        return context

    def get_data(self, N):
        np.random.seed(0)
        self.x = np.expand_dims(np.random.random(N), -1).astype(np.float32)
        self.y = np.expand_dims(np.random.random(N), -1).astype(np.float32)

        data = gprn.Dataset()
        data.add_source_dict({
            'x': self.x,
            'y': self.y,
            'batch_size': None
        })
        return data

    def setup_vars(self, context, data, n):
        num_inducing = data.get_num_inducing(0)
        sigma = np.int(num_inducing*(num_inducing+1)/2)

        np.random.seed(10)

        self.mean_u = np.random.random([context.num_components, context.num_latent, num_inducing]).astype(np.float32)
        self.mean_v = np.random.random([context.num_components, context.num_outputs, context.num_latent, num_inducing]).astype(np.float32)
        self.covar_u = np.random.random([context.num_components, context.num_latent, sigma]).astype(np.float32)
        self.covar_v = np.random.random([context.num_components, context.num_outputs, context.num_latent, sigma]).astype(np.float32)
        self.sigma_y = 0.001
        self.weights = np.array([1]).astype(np.float32)


    def get_parameters(self, context, data, n):
        np.random.seed(10)

        def get_sig_mat(sig, n):
            idx = np.tril_indices(n)
            matrix = np.zeros((n,n))
            matrix[idx] = sig
            sig_mat = np.matmul(matrix, matrix.T)
            return sig_mat

        num_inducing = data.get_num_inducing(0)
        sigma = np.int(num_inducing*(num_inducing+1)/2)

        self.covar_u_mat = get_sig_mat(self.covar_u[0, 0, :], n)
        self.covar_v_mat = get_sig_mat(self.covar_v[0, 0, 0, :], n)
        parameters = gprn.Parameters(context)

        parameters.save(name='inducing_locations_0', var=tf.constant(data.get_inducing_points_from_source(0)))

        parameters.save(name='q_means_u_0', var=tf.constant(self.mean_u))
        parameters.save(name='q_means_v_0', var=tf.constant(self.mean_v))
        parameters.save(name='q_covars_u_0_raw', var=tf.constant(self.covar_u))
        parameters.save(name='q_covars_v_0_raw', var=tf.constant(self.covar_v))

        parameters.load_posterior_covariance(name='q_covars_u_0', from_name='q_covars_u_0_raw', shape=[context.num_components, context.num_latent, sigma], n=num_inducing)
        parameters.load_posterior_covariance(name='q_covars_v_0', from_name='q_covars_v_0_raw', shape=[context.num_components, context.num_outputs, context.num_latent, sigma], n=num_inducing)

        parameters.load_posterior_cholesky(name='q_cholesky_u_0', from_name='q_covars_u_0_raw', shape=[context.num_components, context.num_latent, sigma], n=num_inducing)
        parameters.load_posterior_cholesky(name='q_cholesky_v_0', from_name='q_covars_v_0_raw', shape=[context.num_components, context.num_outputs, context.num_latent, sigma], n=num_inducing)


        parameters.create(name='q_raw_weights', init=tf.constant(self.weights), trainable=False)

        parameters.create(name='noise_sigma_0', init=tf.constant([np.log(self.sigma_y)]), trainable=False)
        parameters.load_posterior_component_weights()

        return parameters

    def random_psd_matrix(self, n):
        jit = 1e-5
        chol_k = np.tril(np.random.random([n, n]), -1)
        k = np.matmul(chol_k, chol_k.T)+np.eye(n)*jit
        return k.astype(np.float32)

    def get_actual_ell_with_no_inducing(self, n):
        total = 0.0
        
        c1 = -(n/2)*np.log(2*np.pi*self.sigma_y*self.sigma_y)
        c2 = -1.0/(2.0*self.sigma_y*self.sigma_y)

        err = (self.y[:, 0]-self.mean_u[0, 0, :])
        total += np.dot(err, err)
        total = 0.0

        total += np.trace(self.covar_u_mat)
        print('tr[f]', np.trace(self.covar_u_mat))
        print('self.sigma_y*self.sigma_y', self.sigma_y*self.sigma_y)
        print('c1', c1)
        print('c2', c2)
        print('err', np.dot(err, err))
        print('c2*total', c2*total)


        return c1+c2*total


    def test__build_ell__no_inducing(self):
        #Arrange
        n = 50
        context = self.get_context()
        data = self.get_data(n)
        self.setup_vars(context, data, n)
        parameters = self.get_parameters(context, data, n)
        context.parameters = parameters

        sparsity = gprn.sparsity.StandardSparsity(data, context)

        mock_elbo  = lambda: None
        mock_elbo.context = context
        mock_elbo.data = data
        mock_elbo.num_outputs = 1
        mock_elbo.q_num_components = 1
        mock_elbo.q_weights = self.weights
        mock_elbo.sparsity = sparsity
        ans=self.get_actual_ell_with_no_inducing(n)

        k_se = gprn.kernels.SE(1)
        k_se.setup(context)

        context.kernels = [{
            'f': [k_se]
        }]


        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            
            mu_ans = self.mean_u[0, 0, :]

            #Act
            data.setup(context)
            data.create_placeholders()
            f_dict = data.next_batch(0, force_all=True)

            ell = gprn.elbos.ell.GP_ELL(context=context, r=0)
            ell.setup(mock_elbo)
            res = ell._build_ell()

            #Assert
            np.testing.assert_almost_equal(res.eval(feed_dict=f_dict),ans, decimal=self.decimal)


    def _test__build_ell__no_inducing__u__is_y(self):
        #Arrange
        n = 50
        context = self.get_context()
        data = self.get_data(n)
        self.setup_vars(context, data, n)
        self.mean_u[0, 0, :] = np.squeeze(self.y)
        self.covar_u[0, 0, :] = 0.0

        parameters = self.get_parameters(context, data, n)
        context.parameters = parameters

        sparsity = gprn.sparsity.StandardSparsity(data, context)

        mock_elbo  = lambda: None
        mock_elbo.context = context
        mock_elbo.data = data
        mock_elbo.num_outputs = 1
        mock_elbo.q_num_components = 1
        mock_elbo.q_weights = self.weights
        mock_elbo.sparsity = sparsity
        ans=self.get_actual_ell_with_no_inducing(n)

        k_se = gprn.kernels.SE(1)
        k_se.setup(context)

        context.kernels = [{
            'f': [k_se]
        }]


        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            
            mu_ans = self.mean_u[0, 0, :]

            #Act
            data.setup(context)
            data.create_placeholders()
            f_dict = data.next_batch(0, force_all=True)

            ell = gprn.elbos.ell.GP_ELL(context=context, r=0)
            ell.setup(mock_elbo)
            res = ell._build_ell()

            print(ans)

            #Assert
            np.testing.assert_almost_equal(res.eval(feed_dict=f_dict),ans, decimal=self.decimal)
