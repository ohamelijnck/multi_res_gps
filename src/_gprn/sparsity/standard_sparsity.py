from sklearn.cluster import KMeans

import numpy as np
import tensorflow as tf
import math
from .. import util
from . import Sparsity


class StandardSparsity(Sparsity):
    def __init__(self, data, context):
        self.context = context
        self.data = data
        self.batch_size = self.data.get_batch_size(source=0)
        self.num_latent = self.context.num_latent
        self.num_outputs = self.context.num_outputs
        self.num_weights = self.context.num_weights
        self.num_components = self.context.num_components
        self.num_inducing = self.data.get_num_inducing(source=0)

        self.use_diag_covar_flag = self.context.use_diag_covar_flag
        self.jitter=self.context.jitter
        self.num_latent_process = self.context.num_latent_process

        self.parameters = self.context.parameters

        self.get_variables()

    def get_standard_variables(self):
        self.q_means_u_arr = []
        self.q_means_v_arr = []
        self.inducing_locations_arr = []
        self.q_covars_u_arr = []
        self.q_covars_v_arr = []
        self.q_chol_covars_u_arr = []
        self.q_chol_covars_v_arr = []

        self.q_weights = self.parameters.get(name='q_weights')

        for r in range(self.num_latent_process):
            self.q_means_u_arr.append(self.parameters.get(name='q_means_u_{r}'.format(r=r)))
            self.q_means_v_arr.append(self.parameters.get(name='q_means_v_{r}'.format(r=r)))

            self.inducing_locations_arr.append(self.parameters.get(name='inducing_locations_{r}'.format(r=r)))

            self.q_covars_u_arr.append(self.parameters.get(name='q_covars_u_{r}'.format(r=r)))
            self.q_chol_covars_u_arr.append(self.parameters.get(name='q_cholesky_u_{r}'.format(r=r)))
            self.q_covars_v_arr.append(self.parameters.get(name='q_covars_v_{r}'.format(r=r)))
            self.q_chol_covars_v_arr.append(self.parameters.get(name='q_cholesky_v_{r}'.format(r=r)))


    def get_variables(self):
        self.get_standard_variables()

    def _build_marginal(self, m, s_chol, k_zz,  k_xz, k_xx, predict=False):
        k_zz_chol = tf.cholesky(tf.cast(k_zz, tf.float64))

        if self.context.whiten:
            mu = tf.matmul(k_xz, util.tri_mat_solve(tf.transpose(k_zz_chol), m, lower=False))
        else:
            mu = tf.matmul(k_xz, util.tri_mat_solve(tf.transpose(k_zz_chol), util.tri_mat_solve(k_zz_chol, m, lower=True), lower=False))

        A = util.tri_mat_solve(k_zz_chol, tf.transpose(k_xz), lower=True)

        if predict:
            sig = k_xx - tf.reduce_sum(tf.square(A), axis=0)
        else:
            sig = k_xx - tf.matmul(A, A, transpose_a=True)



        #s_chol = tf.cholesky(tf.cast(s, tf.float64))
        if self.context.whiten:
            A = tf.matmul(k_xz, util.tri_mat_solve(tf.transpose(k_zz_chol), s_chol, lower=False))
        else:
            A = tf.matmul(k_xz, util.mat_solve(k_zz, s_chol))

        if predict:
            sig = sig + tf.reduce_sum(tf.square(A), axis=1)
        else:
            sig = sig + tf.matmul(A, A, transpose_b=True)


        return mu, sig


    def _build_f_intermediate(self, k, r, X, predict=False, use_non_vectorised=False):
        mu_f_k_arr = [0.0 for j in range(self.num_latent)]
        sigma_f_k_arr = [[[0.0 for n2 in range(self.batch_size)] for n1 in range(self.batch_size)] for j in range(self.num_latent)]
        kern_f = self.context.kernels[r]['f']

        inducing_locations = self.inducing_locations_arr[r]

        q_means_u = self.q_means_u_arr[r]
        q_covars_u = self.q_covars_u_arr[r]
        q_chol_covars_u = self.q_chol_covars_u_arr[r]

        x_is_z_flag = (not predict) and (not self.data.inducing_added_flag)
        x_is_z_flag = False

        for j in range(self.num_latent):
            mu_f_jk = None
            mu_w_jik = None

            #Compute
            if not  use_non_vectorised:
                K_xx_f = kern_f[j].kernel(X, X, jitter=False, diag=predict)
                #K_xx_f = util.add_jitter(K_xx_f, 1e-6)
                #jit = (self.inducing_locations.shape[1]==X.shape[1] and self.inducing_locations.shape[0]==X.shape[0])
                K_xz_f = kern_f[j].kernel(X, inducing_locations, jitter=x_is_z_flag)
                #K_xz_f = util.add_jitter(K_xz_f, 1e-6)
                K_zz_f = kern_f[j].kernel(inducing_locations, inducing_locations, jitter=True)
            else:
                K_xx_f = kern_f[j].kernel_non_vectorised(X, X, jitter=False, diag=predict)
                #K_xx_f = util.add_jitter(K_xx_f, 1e-6)
                #jit = (self.inducing_locations.shape[1]==X.shape[1] and self.inducing_locations.shape[0]==X.shape[0])
                K_xz_f = kern_f[j].kernel_non_vectorised(X, inducing_locations, jitter=x_is_z_flag)
                #K_xz_f = util.add_jitter(K_xz_f, 1e-6)
                K_zz_f = kern_f[j].kernel_non_vectorised(inducing_locations, inducing_locations, jitter=True)
                

            mu_f_jk, sigma_f_jk = self._build_marginal(tf.expand_dims(q_means_u[k,j,:],  1), q_chol_covars_u[k,j,:,:], K_zz_f, K_xz_f, K_xx_f, predict)

            if False:
                mu_f_jk = tf.expand_dims(q_means_u[k,j,:],  1)
                if predict:
                    sigma_f_jk = tf.diag_part(q_covars_u[k, j, :, :])
                else:
                    sigma_f_jk = q_covars_u[k, j, :, :]

            mu_f_k_arr[j] = mu_f_jk

            #Compute Sigma
            #sigma_f_k_arr[j] = sigma_f_jk + tf.pow(self.sigma_f[j],2) * tf.eye(X.get_shape().as_list()[0])
            #sigma_f_k_arr[j] = sigma_f_jk + util.var_postive(self.sigma_f[j]) * tf.eye(self.batch_size)
            sigma_f_k_arr[j] = sigma_f_jk 
        return tf.stack(mu_f_k_arr), tf.stack(sigma_f_k_arr)

    def _build_w_intermediate(self, k, r, X, predict=False, use_non_vectorised=False):
        mu_w_k_arr = [[0.0 for j in range(self.num_latent)] for i in range(self.num_outputs)]
        sigma_w_k_arr = [[[[0.0 for n2 in range(self.batch_size)] for n1 in range(self.batch_size)] for j in range(self.num_latent)] for i in range(self.num_outputs)]
        kern_w = self.context.kernels[r]['w']

        inducing_locations = self.inducing_locations_arr[r]

        q_means_v = self.q_means_v_arr[r]
        q_covars_v = self.q_covars_v_arr[r]
        q_chol_covars_v = self.q_chol_covars_v_arr[r]

        x_is_z_flag = (not predict) and (not self.data.inducing_added_flag)
        x_is_z_flag = False

        for j in range(self.num_latent):
            for i in range(self.num_outputs):
                #compute kernels
                if not use_non_vectorised:
                    K_xx_w = kern_w[i][j].kernel(X, X, jitter=False, diag=predict)
                    K_xz_w = kern_w[i][j].kernel(X, inducing_locations, jitter=x_is_z_flag)
                    K_zz_w = kern_w[i][j].kernel(inducing_locations, inducing_locations, jitter=True)
                else:
                    K_xx_w = kern_w[i][j].kernel_non_vectorised(X, X, jitter=False, diag=predict)
                    K_xz_w = kern_w[i][j].kernel_non_vectorised(X, inducing_locations, jitter=x_is_z_flag)
                    K_zz_w = kern_w[i][j].kernel_non_vectorised(inducing_locations, inducing_locations, jitter=True)


                mu_w_jik, sigma_w_jik = self._build_marginal(tf.expand_dims(q_means_v[k,i,j,:], 1), q_chol_covars_v[k,i,j,:,:], K_zz_w, K_xz_w, K_xx_w, predict)

                mu_w_k_arr[i][j] = mu_w_jik
                sigma_w_k_arr[i][j] = sigma_w_jik

        return tf.stack(mu_w_k_arr), tf.stack(sigma_w_k_arr)


    def _build_standard_intermediate_conditionals(self, k, r, X, predict=False, use_non_vectorised=False):
        print('_build_intermediate_conditionals')

        mu_w, sig_w = None, None
        mu_f, sig_f = self._build_f_intermediate(k, r, X, predict, use_non_vectorised=use_non_vectorised)
        if 'w' in self.context.kernels[r]:
            mu_w, sig_w = self._build_w_intermediate(k, r, X, predict, use_non_vectorised=use_non_vectorised)

        return mu_f, sig_f, mu_w, sig_w

    def _build_intermediate_conditionals(self, k, r, X, predict=False, use_non_vectorised=False):
        return self._build_standard_intermediate_conditionals(k, r, X, predict, use_non_vectorised=use_non_vectorised)



