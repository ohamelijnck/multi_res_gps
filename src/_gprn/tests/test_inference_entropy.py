import sys
sys.path.insert(0,'../../')

import unittest

import tensorflow as tf
import numpy as np
from scipy.stats import multivariate_normal
import math

import gprn
from gprn.kernels import SE

from gprn import Inference

class Test(tf.test.TestCase):
    #called by unittest before each test is ran
    def _se(self, x1, x2, sigma=1.0, ls=1.0):
        return (sigma**2) * np.exp(-((x1-x2)**2)/(2*(ls**2)))

    def setUp(self):
        self.n = 1
        self.x_train = [[1.0]]
        self.y_train = [[1.0]]
        self.num_latent = 1
        self.num_outputs = 1
        self.num_weights = 1
        self.num_components = 1
        self.num_inducing = 1
        self.inducing_locations = [[1.0]]
        self.kern_f = [SE(num_dimensions=1, ARD=True, length_scale=1.0, sigma=1.0)]
        self.kern_w = [[SE(num_dimensions=1, ARD=True, length_scale=1.0, sigma=1.0)]]
        self.q_means_u = [[[1.0]]]
        self.q_covars_u = [[[2.0]]]
        self.q_means_v = [[[[3.0]]]]
        self.q_covars_v = [[[[4.0]]]]
        self.q_weights = [1.0]
        self.sigma_y = 2.0

        self.inference = Inference(
            tf.constant(self.x_train),
            tf.constant(self.y_train),
            self.num_latent,
            self.num_outputs,
            self.num_weights,
            self.num_components,
            self.num_inducing,
            tf.constant(self.inducing_locations),
            self.kern_f,
            self.kern_w,
            tf.constant(self.q_means_u),
            tf.constant(self.q_covars_u),
            tf.constant(self.q_means_v),
            tf.constant(self.q_covars_v),
            tf.constant(self.q_weights),
            self.sigma_y
        )
        print("\n In method", self._testMethodName)

    def test_build_entropy__1d__is_correct(self):
        #Arrange
        
        total_sum = 0.0
        for k in range(self.num_components):
            pi_k = self.q_weights[k]

            sum_u = 0.0
            for l in range(self.num_components):
                m_k = self.q_means_u[k][0][0]
                S_k = self.q_covars_u[k][0][0]**2
                m_l = self.q_means_u[l][0][0]
                S_l = self.q_covars_u[l][0][0]**2
                sum_u += multivariate_normal.pdf(m_k, mean=m_l, cov=(S_k+S_l))


            sum_v = 0.0
            for l in range(self.num_components):
                m_k = self.q_means_v[k][0][0][0]
                S_k = self.q_covars_v[k][0][0][0]**2
                m_l = self.q_means_v[l][0][0][0]
                S_l = self.q_covars_v[l][0][0][0]**2
                sum_v += multivariate_normal.pdf(m_k, mean=m_l, cov=(S_k+S_l))

            total_sum += pi_k * (np.log(sum_u)+np.log(sum_v))
        desired = -total_sum

        #Act
        graph = self.inference._build_entropy()
        init = tf.global_variables_initializer()
        with tf.Session() as session:
            session.run(init)
            result = graph.eval()

            #Assert
            np.testing.assert_almost_equal(result, desired, decimal=5)

    def test_build_ell__1d__is_correct(self):
        #Arrange
        
        c1 = -self.num_outputs*np.log(2*np.pi*self.sigma_y**2)/2
        c2 = -1/(2*self.sigma_y**2)
        total_sum = 0.0
        for n in range(self.n):
            for k in range(self.num_components):
                for p in range(self.num_outputs):
                    y_ni = self.y_train[n][p]
                    k_xx = self._se(self.x_train[0][0], self.x_train[0][0])
                    k_xz = self._se(self.x_train[0][0], self.inducing_locations[0][0])
                    k_zx = k_xz
                    k_zz = self._se(self.inducing_locations[0][0], self.inducing_locations[0][0])
                    k_zz_inv = 1/k_zz

                    mu_w_k_ni = k_xz*k_zz_inv*self.q_means_v[k][0][0][0]
                    mu_f_k_n = k_xz*k_zz_inv*self.q_means_u[k][0][0]

                    sigma_w_k_ni = k_xx-k_xz*k_zz_inv*k_zx + k_xz*k_zz_inv*(self.q_covars_v[k][0][0][0]**2)*k_zz_inv*k_zx
                    sigma_f_k_n = k_xx-k_xz*k_zz_inv*k_zx + k_xz*k_zz_inv*(self.q_covars_u[k][0][0]**2)*k_zz_inv*k_zx

                    err = (y_ni-mu_w_k_ni*mu_f_k_n)**2
                    total_sum += err + mu_f_k_n*sigma_w_k_ni*mu_f_k_n + mu_w_k_ni*sigma_f_k_n*mu_w_k_ni + sigma_w_k_ni*sigma_f_k_n

        desired = c1+c2*total_sum

        #Act
        graph = self.inference._build_expected_log_likelihood()
        init = tf.global_variables_initializer()
        with tf.Session() as session:
            session.run(init)
            result = graph.eval()[0][0]

            #Assert
            np.testing.assert_almost_equal(result, desired, decimal=5)

    def test_build_cross_entropy__1d__is_correct(self):
        #Arrange
        
        c1 = -self.num_outputs*np.log(2*np.pi*self.sigma_y**2)/2
        c2 = -1/(2*self.sigma_y**2)
        total_sum = 0.0
        for n in range(self.n):
            for k in range(self.num_components):
                for p in range(self.num_outputs):
                    y_ni = self.y_train[n][p]
                    k_xx = self._se(self.x_train[0][0], self.x_train[0][0])
                    k_xz = self._se(self.x_train[0][0], self.inducing_locations[0][0])
                    k_zx = k_xz
                    k_zz = self._se(self.inducing_locations[0][0], self.inducing_locations[0][0])
                    k_zz_inv = 1/k_zz

                    mu_w_k_ni = k_xz*k_zz_inv*self.q_means_v[k][0][0][0]
                    mu_f_k_n = k_xz*k_zz_inv*self.q_means_u[k][0][0]

                    sigma_w_k_ni = k_xx-k_xz*k_zz_inv*k_zx + k_xz*k_zz_inv*(self.q_covars_v[k][0][0][0]**2)*k_zz_inv*k_zx
                    sigma_f_k_n = k_xx-k_xz*k_zz_inv*k_zx + k_xz*k_zz_inv*(self.q_covars_u[k][0][0]**2)*k_zz_inv*k_zx

                    err = (y_ni-mu_w_k_ni*mu_f_k_n)**2
                    total_sum += err + mu_f_k_n*sigma_w_k_ni*mu_f_k_n + mu_w_k_ni*sigma_f_k_n*mu_w_k_ni + sigma_w_k_ni*sigma_f_k_n

        desired = c1+c2*total_sum

        #Act
        graph = self.inference._build_cross_entropy()
        init = tf.global_variables_initializer()
        with tf.Session() as session:
            session.run(init)
            result = graph.eval()

            #Assert
            np.testing.assert_almost_equal(result, desired, decimal=5)






