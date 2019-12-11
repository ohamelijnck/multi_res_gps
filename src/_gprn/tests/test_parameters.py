#To run tests: python -m unittest *

import sys
sys.path.insert(0,'../../')

import unittest

import tensorflow as tf
import numpy as np
import math
import scipy as sp

import gprn
from gprn import util

class Test(tf.test.TestCase):
    #called by unittest before each test is ran
    def setUp(self):
        print("\n In method", self._testMethodName)
        self.decimal = 6

    def get_context(self):
        context = gprn.context.ContextFactory().create()
        context.jitter = 1e-5
        context.use_diag_covar = False
        return context

    def test__load_posterior_covariance(self):
        init = tf.global_variables_initializer()
        with tf.Session() as session:
            session.run(init)
            #Arrange
            context = self.get_context()
            n = 3
            sig = int(n*(n+1)/2)

            x = np.random.random([1, 1, sig]).astype(np.float32)

            idx = np.tril_indices(n)
            matrix = np.zeros((n,n))
            matrix[idx] = x[0, 0, :]
            ans = np.matmul(matrix, matrix.T)

            parameters = gprn.parameters.Parameters(context)
            parameters.save('sig_raw', var=tf.constant(x))

            #Act
            parameters.load_posterior_covariance(name='sig', from_name='sig_raw', shape=[1, 1, sig], n=n)
            res = parameters.get('sig')[0, 0, :].eval()

            #Assert
            np.testing.assert_almost_equal(res,ans, decimal=self.decimal)


    def test__load_posterior_cholesky(self):
        init = tf.global_variables_initializer()
        with tf.Session() as session:
            session.run(init)
            #Arrange
            context = self.get_context()
            n = 3
            sig = int(n*(n+1)/2)

            x = np.random.random([1, 1, sig]).astype(np.float32)

            idx = np.tril_indices(n)
            matrix = np.zeros((n,n))
            matrix[idx] = x[0, 0, :]
            ans = matrix

            parameters = gprn.parameters.Parameters(context)
            parameters.save('sig_raw', var=tf.constant(x))

            #Act
            parameters.load_posterior_cholesky(name='sig', from_name='sig_raw', shape=[1, 1, sig], n=n)
            res = parameters.get('sig')[0, 0, :].eval()

            #Assert
            np.testing.assert_almost_equal(res,ans, decimal=self.decimal)







