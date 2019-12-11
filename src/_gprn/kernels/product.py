import numpy as np
import tensorflow as tf

from . import Kernel
from .. import util

class Product(Kernel):
    _id = 0
    def __init__(self, k_arr, include_arr = [], mask=None):
        super(Product, self).__init__(mask)
        self.k_arr = k_arr
        self.include_arr = include_arr

    def setup(self, context):
        self.context = context
        for k in self.k_arr:
            k.setup(context)

    def _kernel_non_vectorised(self, X1, X2, jitter=False, debug=False, include_dimensions=None):
        K = None
        for i in range(len(self.k_arr)):
            k = self.k_arr[i]
            include_dimensions = self.include_arr[i]
            val = k._kernel_non_vectorised(X1, X2, jitter=False, include_dimensions=include_dimensions)
            #val = k._kernel(X1, X2, jitter=False)

            if K is None:
                K = val
            else:
                K = tf.multiply(K, val)
        
        if jitter is True:
            K =  K+(self.context.jitter*tf.eye(tf.shape(K)[1]))

        return K


    def _kernel(self, X1, X2, jitter=False, include_dimensions=None):
        K = None
        for i in range(len(self.k_arr)):
            k = self.k_arr[i]
            include_dimensions = self.include_arr[i]
            val = k._kernel(X1, X2, jitter=False, include_dimensions=include_dimensions)
            #val = k._kernel(X1, X2, jitter=False)

            if K is None:
                K = val
            else:
                K = tf.multiply(K, val)
        
        if jitter is True:
            K =  K+(self.context.jitter*tf.eye(tf.shape(K)[1]))[None, :, :]

        return K


    def get_parameters(self):
        return []

