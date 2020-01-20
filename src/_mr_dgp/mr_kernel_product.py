import numpy as np
import tensorflow as tf
import gpflow
from gpflow import settings
from gpflow import params_as_tensors

class MR_KERNEL_PRODUCT(gpflow.kernels.Kernel):
    def __init__(self, kernels):
        self.kernels = kernels
        pass
    def fK(self, kernels):
        return tf.reduce_prod(kernels, axis=1)

    def K(self, x_arr, kernels):
        return tf.reduce_prod([kernels[i].K(x_arr[i][0], x_arr[i][1]) for i in range(len(self.kernels))], axis=1)

    def Kdiag(self, x_arr, kernels):
        return tf.reduce_prod([kernels[i].Kdiag(x_arr[i][0], x_arr[i][1]) for i in range(len(self.kernels))], axis=1)

