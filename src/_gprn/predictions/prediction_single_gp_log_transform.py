import tensorflow as tf
import numpy as np
import math
from .. import util
from .. import debugger
from .. sparsity import StandardSparsity
from . import PredictionSingleGP

from tensorflow.contrib.distributions import MultivariateNormalTriL
from tensorflow.contrib.distributions import MultivariateNormalDiag 

class PredictionSingleGPLogTransform(PredictionSingleGP):
    def transformed_mean(self, mean, var):
        noise_sigma = tf.square(util.var_postive(self.sigma_y[0]))

        return util.safe_exp(0.5*noise_sigma)*util.safe_exp(mean+0.5*var)

    def transformed_var(self, mean, var, transformed_mean):
        noise_sigma = tf.square(util.var_postive(self.sigma_y[0]))

        return util.safe_exp(2*noise_sigma)*util.safe_exp(2*(mean+0.5*var))-tf.square(transformed_mean)

    def build_sample_graph(self, r=0):
         sample = self.build_sample_single_gp()
         return util.safe_exp(sample)

    def build_graph(self, r=0):
        print('build single gp log transform')
        self.r = r
        expected = self.build_expected_value()
        var = self.build_variance()

        transformed_mean = self.transformed_mean(expected, var)
        transformed_var = self.transformed_var(expected, var, transformed_mean)

        return transformed_mean, transformed_var
