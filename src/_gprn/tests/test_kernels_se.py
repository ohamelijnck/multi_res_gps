#To run tests: python -m unittest *

import sys
sys.path.insert(0,'../../')

import unittest

import tensorflow as tf
import numpy as np
import math

import gprn
from gprn.kernels import SE

class Test(tf.test.TestCase):
    #called by unittest before each test is ran
    def setUp(self):
        print("\n In method", self._testMethodName)

    def test_is_correct__1_d(self):
        #Arrange
        ls = 1.0
        sig = 1.0
        num_dim = 1 
        se = SE(num_dimensions=num_dim, ARD=False, length_scale=ls, sigma=sig)
        #x1 = tf.constant([[ 0.        ], [ 0.12822827], [ 0.25645654], [ 0.38468481], [ 0.51291309], [ 0.64114136], [ 0.76936963], [ 0.8975979 ], [ 1.02582617], [ 1.15405444], [ 1.28228272], [ 1.41051099], [ 1.53873926], [ 1.66696753], [ 1.7951958 ], [ 1.92342407], [ 2.05165235], [ 2.17988062], [ 2.30810889], [ 2.43633716], [ 2.56456543], [ 2.6927937 ], [ 2.82102197], [ 2.94925025], [ 3.07747852], [ 3.20570679], [ 3.33393506], [ 3.46216333], [ 3.5903916 ], [ 3.71861988], [ 3.84684815], [ 3.97507642], [ 4.10330469], [ 4.23153296], [ 4.35976123], [ 4.48798951], [ 4.61621778], [ 4.74444605], [ 4.87267432], [ 5.00090259], [ 5.12913086], [ 5.25735913], [ 5.38558741], [ 5.51381568], [ 5.64204395], [ 5.77027222], [ 5.89850049], [ 6.02672876], [ 6.15495704], [ 6.28318531]])
        x1_val = 2.1
        x2_val = 3.2
        x1 = tf.constant([[x1_val]])
        x2 = tf.constant([[x2_val]])

        #Act
        init = tf.global_variables_initializer()
        with tf.Session() as session:
            session.run(init)
            k = se.kernel(x1, x2)

            #Assert
            result = k.eval().flatten()
            desired = (sig**2) * np.exp(-((x1_val-x2_val)**2)/(2*(ls**2)))

            np.testing.assert_almost_equal(result[0], desired)


    def test_psd__1_d(self):
        #Arrange
        ls = 3.0
        num_dim = 2 
        se = SE(num_dimensions=num_dim, ARD=False, length_scale=ls, sigma=1.0)
        #x1 = tf.constant([[ 0.        ], [ 0.12822827], [ 0.25645654], [ 0.38468481], [ 0.51291309], [ 0.64114136], [ 0.76936963], [ 0.8975979 ], [ 1.02582617], [ 1.15405444], [ 1.28228272], [ 1.41051099], [ 1.53873926], [ 1.66696753], [ 1.7951958 ], [ 1.92342407], [ 2.05165235], [ 2.17988062], [ 2.30810889], [ 2.43633716], [ 2.56456543], [ 2.6927937 ], [ 2.82102197], [ 2.94925025], [ 3.07747852], [ 3.20570679], [ 3.33393506], [ 3.46216333], [ 3.5903916 ], [ 3.71861988], [ 3.84684815], [ 3.97507642], [ 4.10330469], [ 4.23153296], [ 4.35976123], [ 4.48798951], [ 4.61621778], [ 4.74444605], [ 4.87267432], [ 5.00090259], [ 5.12913086], [ 5.25735913], [ 5.38558741], [ 5.51381568], [ 5.64204395], [ 5.77027222], [ 5.89850049], [ 6.02672876], [ 6.15495704], [ 6.28318531]])
        x1 = tf.constant([[float(i)] for i in range(0, 10)])
        x2 = x1

        #Act
        init = tf.global_variables_initializer()
        with tf.Session() as session:
            session.run(init)
            k = se.kernel(x1, x2, jitter=True)

            #Assert
            result = tf.self_adjoint_eig(k)[0]
            result = result.eval().flatten()

            assert(all(i >= 0.0 for i in result))
            

    def test_length_scales__multi_dim(self):
        #Arrange
        ls = 3.0
        num_dim = 2.0 
        se = SE(num_dimensions=num_dim, ARD=False, length_scale=ls, sigma=1.0)
        x1 = tf.constant([[1.0, 2.0]])
        x2 = tf.constant([[2.0, 3.0]])

        #Act
        init = tf.global_variables_initializer()
        with tf.Session() as session:
            session.run(init)
            k = se.kernel(x1, x2)

            #Assert
            np.testing.assert_almost_equal(k.eval(),np.exp(-num_dim/(2*ls**2)))

    def test_length_scales(self):
        #Arrange
        ls = 3.0
        se = SE(num_dimensions=1, ARD=True, length_scale=ls, sigma=1.0)
        x1 = tf.constant([[1.0]])
        x2 = tf.constant([[2.0]])

        #Act
        init = tf.global_variables_initializer()
        with tf.Session() as session:
            session.run(init)
            k = se.kernel(x1, x2)

            #Assert
            np.testing.assert_almost_equal(k.eval(),np.exp(-1/(2*ls**2)))


    def test_covar_is_symmetric__multi_dim(self):
        #Arrange
        se = SE(num_dimensions=3, ARD=True, length_scale=1.0, sigma=1.0)
        x1 = tf.constant([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
        x2 = tf.constant([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])

        #Act
        init = tf.global_variables_initializer()
        with tf.Session() as session:
            session.run(init)
            k = se.kernel(x1, x2)

            k_lower = tf.matrix_band_part(k, -1, 0)
            k_upper = tf.matrix_band_part(k, 0, -1)
            k_upper_T = tf.transpose(k_upper)

            #Assert
            assert(tf.equal(k_lower, k_upper_T).eval().all())

    def test_covar_is_symmetric__scalar(self):
        #Arrange
        se = SE(num_dimensions=1, ARD=True, length_scale=1.0, sigma=1.0)
        x1 = tf.constant([[1.0], [2.0], [3.0]])
        x2 = tf.constant([[1.0], [2.0], [3.0]])

        #Act
        init = tf.global_variables_initializer()
        with tf.Session() as session:
            session.run(init)
            k = se.kernel(x1, x2)

            k_lower = tf.matrix_band_part(k, -1, 0)
            k_upper = tf.matrix_band_part(k, 0, -1)
            k_upper_T = tf.transpose(k_upper)

            #Assert
            assert(tf.equal(k_lower, k_upper_T).eval().all())

    def test_covar_is_sigma_for_same_value__multi_dim(self):
        #Arrange
        se = SE(num_dimensions=1, ARD=True, length_scale=1.0, sigma=3.0)
        x1 = tf.constant([[1.0, 2.0, 5.0]])
        x2 = tf.constant([[1.0, 2.0, 5.0]])

        #Act
        init = tf.global_variables_initializer()
        with tf.Session() as session:
            session.run(init)
            v = se.kernel(x1, x2)
            sigma = se.get_parameters()['sigma']
            #Assert
            assert(v.eval() == sigma.eval()**2)

    def test_covar_is_sigma_for_same_value__scalar(self):
        #Arrange
        se = SE(num_dimensions=1, ARD=True, length_scale=1.0, sigma=2.0)
        x1 = tf.constant([[1.0]])
        x2 = tf.constant([[1.0]])

        #Act
        init = tf.global_variables_initializer()
        with tf.Session() as session:
            session.run(init)
            v = se.kernel(x1, x2)
            sigma = se.get_parameters()['sigma']
            #Assert
            assert(v.eval() == sigma.eval() ** 2)
        
