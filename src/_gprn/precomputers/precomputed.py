import numpy as np
import tensorflow as tf
import math
from .. import util

class Precomputed(object):
    def __init__(self, data, context):
        self.data = data
        self.context = context
        self.load_variables_from_context()
        self.get_variables()
        self.build_covars()

    def get_variables(self):
        self.q_covars_u_raw_arr = []
        self.q_covars_v_raw_arr = []

        with tf.variable_scope("parameters", reuse=True):
            for r in range(self.num_resolutions):
                self.q_covars_u_raw_arr.append(tf.get_variable(name='q_covars_u_{r}_raw'.format(r=r)))
                self.q_covars_v_raw_arr.append(tf.get_variable(name='q_covars_v_{r}_raw'.format(r=r)))

                self.q_raw_weights = tf.get_variable(name='q_raw_weights')
                self.q_weights = util.safe_exp(self.q_raw_weights) / tf.reduce_sum(util.safe_exp(self.q_raw_weights))

    def load_variables_from_context(self):
        self.num_resolutions = self.context.num_resolutions
        self.num_latent = self.context.num_latent
        self.num_outputs = self.context.num_outputs
        self.num_components = self.context.num_components
        self.use_diag_covar = self.context.use_diag_covar_flag
        self.train_inducing_points_flag = self.context.train_inducing_points_flag
        self.sigma_y_init = self.context.sigma_y_init
        self.sigma_y_train_flag = self.context.sigma_y_train_flag
        self.jitter = self.context.jitter
        self.num_sigma = self.context.num_sigma

    def build_kernels(self):
        pass

    def get_kernels(self):
        pass

    def build_covars(self):
        self.q_covar_u_arr = []
        self.q_covar_v_arr = []
        for r in range(self.num_resolutions):
            num_inducing = self.data.get_num_inducing(source=r)

            q_covars_u_arr = [[0 for j in range(self.num_latent)] for k in range(self.num_components)]
            q_covars_v_arr = [[[0 for j in range(self.num_latent)] for i in range(self.num_outputs)] for k in range(self.num_components)]
            jit = self.jitter
            jit = 0.0
            for k in range(self.num_components):
                for j in range(self.num_latent):
                    s = util.covar_to_mat(num_inducing, self.q_covars_u_raw_arr[r][k, j, :], self.use_diag_covar, jit)
                    
                    q_covars_u_arr[k][j]=s

                    for i in range(self.num_outputs):
                        s = util.covar_to_mat(num_inducing, self.q_covars_v_raw_arr[r][k, i, j, :], self.use_diag_covar, jit)
                        q_covars_v_arr[k][i][j]=s

            self.q_covar_u_arr.append(tf.stack(q_covars_u_arr))
            self.q_covar_v_arr.append(tf.stack(q_covars_v_arr))

    def get_covars(self, r):
        return self.q_covar_u_arr[r],self.q_covar_v_arr[r]

    def get_chol_covars(self, r):
        num_inducing = self.data.get_num_inducing(source=r)
        return util.vec_to_lower_triangle_matrix(num_inducing, self.q_covars_u_raw_arr[r][0, 0, :]),self.q_covars_v_raw_arr[r]


    def build_conditionals(self):
        pass

    def get_conditionals(self):
        pass


