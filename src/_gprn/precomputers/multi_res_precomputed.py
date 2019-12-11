import numpy as np
import tensorflow as tf
import math
from .. import util
from . import Precomputed

class MultiResPrecomputed(Precomputed):
    def __init__(self, data, context):
        super(MultiResPrecomputed, self).__init__(data, context)
        self.data = data
        self.context = context
        self.load_variables_from_context()
        self.get_multi_res_variables()
        self.build_multi_res_covars()

    def get_multi_res_variables(self):
        with tf.variable_scope("parameters", reuse=True):
            self.q_covars_uh_raw = tf.get_variable(name='q_covars_uh_raw')
            self.q_covars_ug_raw = tf.get_variable(name='q_covars_ug_raw')
            self.q_covars_ugg_raw = tf.get_variable(name='q_covars_ugg_raw')

    def build_kernels(self):
        pass

    def get_kernels(self):
        pass

    def build_multi_res_covars(self):        
        q_covars_uh_arr = [[0]]
        q_covars_ug_arr = [[0]]
        q_covars_ugg_arr =[[0]]
        jit = 0

        for k in range(self.num_components):
            s = util.covar_to_mat(self.data.get_num_inducing(source=1), self.q_covars_uh_raw[k, 0, :], self.use_diag_covar, jit)
            q_covars_uh_arr[k][0]=s

            s = util.covar_to_mat(self.data.get_num_inducing(source=1), self.q_covars_ug_raw[k, 0, :], self.use_diag_covar, jit)
            q_covars_ug_arr[k][0]=s

            s = util.covar_to_mat(self.data.get_num_inducing(source=1), self.q_covars_ugg_raw[k, 0, :], self.use_diag_covar, jit)
            q_covars_ugg_arr[k][0]=s

        #These variables are not trainable but the 'raw' versions are
        self.q_covars_uh = tf.stack(q_covars_uh_arr)
        self.q_covars_ug = tf.stack(q_covars_ug_arr)
        self.q_covars_ugg = tf.stack(q_covars_ugg_arr)


    def get_multi_res_covars(self):
        return self.q_covars_ugg,self.q_covars_ug,self.q_covars_uh

    def build_conditionals(self):
        pass

    def get_conditionals(self):
        pass



