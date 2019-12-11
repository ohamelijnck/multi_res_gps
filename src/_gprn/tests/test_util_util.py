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
from gprn.util import vec_cholesky_to_mat

class Test(tf.test.TestCase):
    #called by unittest before each test is ran
    def setUp(self):
        print("\n In method", self._testMethodName)
        self.decimal = 6

    def test_vec_cholesky_to_mat__is_correct(self):
        #Arrange
        n = 2
        vec = tf.Variable([1.0, 2.0, 3.0])
        #Act
        init = tf.global_variables_initializer()
        with tf.Session() as session:
            session.run(init)

            mat = util.vec_cholesky_to_mat(n, vec, jitter=0.0)

            #Assert
            disired = [[1.0, 2.0], [2.0, 13.0]]
            np.testing.assert_almost_equal(mat.eval(),disired)



    def test_vec_to_lower_triangle_matrix__single(self):
        #Arrange
        vec = tf.Variable([1.0])
        n = 1

        #Act
        init = tf.global_variables_initializer()
        with tf.Session() as session:
            session.run(init)

            mat = util.vec_to_lower_triangle_matrix(n, vec)

            #Assert
            disired = [[1.0]]
            np.testing.assert_almost_equal(mat.eval(),disired)

    def test_vec_to_lower_triangle_matrix__small(self):
        #Arrange
        vec = tf.Variable([2.0, 3.0, 4.0])
        n = 2

        #Act
        init = tf.global_variables_initializer()
        with tf.Session() as session:
            session.run(init)

            mat = util.vec_to_lower_triangle_matrix(n, vec)

            #Assert
            disired = [[2.0, 0.0], [3.0, 4.0]]
            np.testing.assert_almost_equal(mat.eval(),disired)

    def test_vec_to_lower_triangle_matrix__bigger(self):
        #Arrange
        vec = tf.Variable([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        n = 3

        #Act
        init = tf.global_variables_initializer()
        with tf.Session() as session:
            session.run(init)

            mat = util.vec_to_lower_triangle_matrix(n, vec)

            #Assert
            disired = [[1.0, 0.0, 0.0], [2.0, 3.0, 0.0], [4.0, 5.0, 6.0]]
            np.testing.assert_almost_equal(mat.eval(),disired)

    def test_add_jitter(self):
        #Arrange
        n = 5
        jit=0.2
        input_mat = tf.Variable(np.ones(n).astype(np.float32))

        res_mat = np.ones(n)+np.eye(n)*jit

        #Act
        init = tf.global_variables_initializer()
        with tf.Session() as session:
            session.run(init)

            mat = util.add_jitter(input_mat, jit)

            #Assert
            disired = res_mat
            np.testing.assert_almost_equal(mat.eval(),disired)

    def test_log_chol_matrix_det(self):
        #Arrange
        n = 10
        input_mat = np.ones(n)+np.eye(n)*0.2

        ans = np.log(np.linalg.det(input_mat))

        #Act
        init = tf.global_variables_initializer()
        with tf.Session() as session:
            session.run(init)

            res = util.log_chol_matrix_det(tf.cholesky(tf.constant(input_mat)))

            #Assert
            np.testing.assert_almost_equal(res.eval(),ans, decimal=self.decimal)

    def test_log_normal_chol(self):
        #Arrange
        np.random.seed(1)

        n = 10
        jit=0.1
        x = np.expand_dims(np.random.random(n), -1)
        mu = np.expand_dims(np.random.random(n), -1)
        chol_k = np.tril(np.random.random([n, n]), -1)
        k = np.matmul(chol_k, chol_k.T)+np.eye(n)*jit

        ans = sp.stats.multivariate_normal.logpdf(x[:, 0], mu[:, 0], k)

        #Act
        init = tf.global_variables_initializer()
        with tf.Session() as session:
            session.run(init)

            res = util.log_normal_chol(tf.constant(x.astype(np.float32)), tf.constant(mu.astype(np.float32)), tf.cholesky(tf.constant(k.astype(np.float32))), n)[0, 0]

            #Assert
            np.testing.assert_almost_equal(res.eval(),ans, decimal=self.decimal)

    def test__log_normal_chol__zero_x(self):
        #Arrange
        np.random.seed(1)

        n = 10
        jit=1e-5
        x = np.expand_dims(np.zeros(n), -1)
        mu = np.expand_dims(np.random.random(n), -1)
        chol_k = np.tril(np.random.random([n, n]), -1)
        k = np.matmul(chol_k, chol_k.T)+np.eye(n)*jit

        ans = sp.stats.multivariate_normal.logpdf(x[:, 0], mu[:, 0], k)
        log_det = 2*np.sum(np.log(np.diag(np.linalg.cholesky(k))))
        ans = -0.5*(n*np.log(2*np.pi)+log_det+np.matmul(mu.T, sp.linalg.cho_solve((np.linalg.cholesky(k), True), mu))[0, 0])

        #Act
        init = tf.global_variables_initializer()
        with tf.Session() as session:
            session.run(init)

            res = util.log_normal_chol(tf.constant(x.astype(np.float32)), tf.constant(mu.astype(np.float32)), tf.cholesky(tf.constant(k.astype(np.float32))), n)[0, 0]

            #Assert
            np.testing.assert_almost_equal(res.eval(),ans, decimal=self.decimal)

    def test__log_normal_chol__x_is_mean(self):
        #Arrange
        np.random.seed(1)

        n = 10
        jit=0.1
        mu = np.expand_dims(np.random.random(n), -1)
        x = mu
        chol_k = np.tril(np.random.random([n, n]), -1)
        k = np.matmul(chol_k, chol_k.T)+np.eye(n)*jit

        ans = sp.stats.multivariate_normal.logpdf(x[:, 0], mu[:, 0], k)

        #Act
        init = tf.global_variables_initializer()
        with tf.Session() as session:
            session.run(init)

            res = util.log_normal_chol(tf.constant(x.astype(np.float32)), tf.constant(mu.astype(np.float32)), tf.cholesky(tf.constant(k.astype(np.float32))), n)[0, 0]

            #Assert
            np.testing.assert_almost_equal(res.eval(),ans, decimal=self.decimal)


    def test__safe_log(self):
        x = 0.1
        ans = np.log(x)

        #Act
        init = tf.global_variables_initializer()
        with tf.Session() as session:
            session.run(init)

            res = util.safe_log(tf.constant(x))

            #Assert
            np.testing.assert_almost_equal(res.eval(),ans, decimal=self.decimal)

    def test__safe_exp(self):
        x = 0.1
        ans = np.exp(x)

        #Act
        init = tf.global_variables_initializer()
        with tf.Session() as session:
            session.run(init)

            res = util.safe_exp(tf.constant(x))

            #Assert
            np.testing.assert_almost_equal(res.eval(),ans, decimal=self.decimal)






