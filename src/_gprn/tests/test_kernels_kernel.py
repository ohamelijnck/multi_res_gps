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

    def test_kernel_symmetrix(self):
        #Arrange
        se = SE()
        x1 = tf.constant([1.0, 2.0])
        x2 = tf.constant([1.0, 2.0])

        #Act
        init = tf.global_variables_initializer()
        with tf.Session() as session:
            session.run(init)
            v = se.kernel(x1, x2)
            #Assert
            print(v.eval())
        

