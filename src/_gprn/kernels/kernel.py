import numpy as np
import tensorflow as tf

from .. import util

class Kernel(object):
    def __init__(self, bool_mask):
        self.bool_mask = bool_mask
        self.mask = None
        if (self.bool_mask):
            mask =[]
            for i in range(len(bool_mask)):
                if bool_mask[i] is 1:
                    mask.append(i)
            self.mask = mask

    def _np_mask(self, input1, input2):
        b = np.array([bool(x) for x in self.bool_mask])
        return input1[:,b], input2[:, b]
        
    def s(self, arr, idx):
        total_arr = []
        for i in idx:
            if len(arr.shape) == 1:
                total_arr.append(arr[i])
            else:
                total_arr.append(arr[:, i])

        if len(arr.shape) == 1:
            total_arr = tf.stack(total_arr, axis=0)
        else:
            total_arr = tf.stack(total_arr, axis=1)

        return total_arr

    def _mask(self, input1,input2):
        arr_1 = []
        arr_2 = []
        for i in range(len(self.mask)):
            col_1 = input1[:,self.mask[i]]
            col_2 = input2[:,self.mask[i]]
            arr_1.append(col_1)
            arr_2.append(col_2)
        arr_1 = tf.stack(arr_1, axis=1)
        arr_2 = tf.stack(arr_2, axis=1)
        return arr_1, arr_2

    def normalise(self, k):
        rows = tf.matmul(tf.diag(tf.sqrt(tf.diag_part(k))), 0.0*k+1.0)
        cols = tf.transpose(rows)
        return tf.divide(tf.divide(k, rows), cols)

    def kernel_non_vectorised(self,input1,input2, jitter=False, diag=False, debug=False, include_dimensions=None):
        if diag:
            jit = 0
            if jitter:
                jit = self.context.jitter

            sigma = util.var_postive(self.sigma)

            return sigma*tf.ones(tf.shape(input1)[0])+jit

        if self.mask:
            arr_1, arr_2 = self._mask(input1, input2)
            K = self._kernel_non_vectorised(arr_1, arr_2, jitter=jitter, debug=debug, include_dimensions=include_dimensions)
        else:
            K =  self._kernel_non_vectorised(input1, input2, jitter)

        return K

    def kernel(self,input1,input2, jitter=False, diag=False, debug=False, include_dimensions=None):
        if diag:
            jit = 0
            if jitter:
                jit = self.context.jitter

            sigma = util.var_postive(self.sigma)

            return sigma*tf.ones(tf.shape(input1)[0])+jit

        if self.mask:
            arr_1, arr_2 = self._mask(input1, input2)
            K = self._kernel(arr_1, arr_2, jitter=jitter, debug=debug, include_dimensions=include_dimensions)
        else:
            K =  self._kernel(input1, input2, jitter)

        return K

    def get_params(self):
        pass


