from sklearn.cluster import KMeans

import numpy as np
import tensorflow as tf
import math
from .. import util
from . import Sparsity


class MRSparsity(Sparsity):
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
        #k_zz = tf.tile(k_zz, [tf.shape(k_xz)[0], 1, 1]) # N x M x M
        
        _s_chol = s_chol
        s_chol = tf.tile(tf.expand_dims(s_chol, 0), [tf.shape(k_xz)[0], 1, 1]) # N x M x M

        #k_zz in 1 x M x M 
        k_zz_chol = tf.cholesky(tf.cast(k_zz, tf.float64)) #  M x M
        _k_zz_chol = k_zz_chol
        k_zz_chol = tf.tile(k_zz_chol, [tf.shape(k_xz)[0], 1, 1]) # N x M x M

        #TODO: make work white is off
            

        if self.context.whiten:
            mu = tf.matmul(k_xz, util.tri_mat_solve(tf.transpose(k_zz_chol, [0, 2 ,1]), m, lower=False, name='m_'))
        else:
            mu = tf.matmul(k_xz, util.tri_mat_solve(tf.transpose(k_zz_chol), util.tri_mat_solve(k_zz_chol, m, lower=True), lower=False))

        if False:
            a_shp = tf.shape(k_xz)
            k = tf.reshape(k_xz, [a_shp[0]*a_shp[1], a_shp[2]]) #NS x M
            k_t = tf.transpose(k)
            a = util.tri_mat_solve(_k_zz_chol[0, :, :], k_t, lower=True, name='A1_') # M x NS
            a_t = tf.transpose(a)
            A = tf.reshape(a_t, a_shp) # N x S x M
            A = tf.transpose(A, [0, 2, 1])
            print('k: ', k)
            print('k_t: ', k)
            print('a: ', a)
            print('a_t: ', a_t)
            print('A: ', A)

            #a = tf.transpose(k_xz, [0, 2, 1]) # N x M x S
            #a = tf.reshape(k_xz, []) 
        else:
            A = util.tri_mat_solve(k_zz_chol, tf.transpose(k_xz, [0, 2, 1]), lower=True, name='A1_') # N x M x S
        sig = k_xx - tf.matmul(tf.transpose(A, [0, 2, 1]), A) # N x S x S

        if self.context.whiten:
            a = util.tri_mat_solve(tf.transpose(_k_zz_chol[0, :, :]), _s_chol, lower=False, name='A2_')
            a = tf.expand_dims(a, 0)
            a = tf.tile(a, [tf.shape(k_xz)[0], 1, 1])
            A = tf.matmul(k_xz, a)
            #A = tf.matmul(k_xz, util.tri_mat_solve(tf.transpose(k_zz_chol, [0, 2, 1]), s_chol, lower=False, name='A2_'))
        else:
            A = tf.matmul(k_xz, util.mat_solve(k_zz, s_chol))

        sig = sig + tf.matmul(A, tf.transpose(A, [0, 2, 1]))

        return mu, sig


    def _build_f_intermediate(self, k, r, X, predict=False):
        mu_f_k_arr = [0.0 for j in range(self.num_latent)]
        sigma_f_k_arr = [[[0.0 for n2 in range(self.batch_size)] for n1 in range(self.batch_size)] for j in range(self.num_latent)]
        kern_f = self.context.kernels[r]['f']

        #X : N x S x D
        inducing_locations = self.inducing_locations_arr[r]
        inducing_locations = tf.expand_dims(inducing_locations, 0) #1 x M x D

        q_means_u = self.q_means_u_arr[r] #K, Q, M
        q_covars_u = self.q_covars_u_arr[r] #K, Q, M, M
        q_chol_covars_u = self.q_chol_covars_u_arr[r] #K, Q, M, M

        #when the inducing points lie exactly at the training points adding jitter to K_xz can sometimes help numerically
        x_is_z_flag = False 

        for j in range(self.num_latent):
            mu_f_jk = None
            mu_w_jik = None

            #Compute
            K_xx_f = kern_f[j].kernel(X, X, jitter=False) # N x S x S

            K_xz_f = kern_f[j].kernel(X, inducing_locations, jitter=x_is_z_flag) # N x S x M
            K_zz_f = kern_f[j].kernel(inducing_locations, inducing_locations, jitter=True) # 1 x M x M
            #TODO: is there a way to remove the tile?
            #K_zz_f = tf.tile(K_zz_f, [tf.shape(K_xz_f)[0], 1, 1]) # N x M x M

            m = tf.expand_dims(q_means_u[k,j,:],  1) # M x 1
            m = tf.tile(tf.expand_dims(m, 0), [tf.shape(K_xz_f)[0], 1, 1]) # N x M x 1

            S =  q_chol_covars_u[k,j,:,:] # M x M
            #S = tf.tile(tf.expand_dims(S, 0), [tf.shape(K_xz_f)[0], 1, 1]) # N x M x M


            #Compute $q(f_q) = \int p(f_q|u_q) q(u_q) \mathop{du}$
            #return shapes:
                #mu_f_jk = N x S x 1
                #sigma_f_jk = N x S x S
            mu_f_jk, sigma_f_jk = self._build_marginal(m, S, K_zz_f, K_xz_f, K_xx_f, predict)

            mu_f_k_arr[j] = mu_f_jk
            sigma_f_k_arr[j] = sigma_f_jk 
            #Compute Sigma
            #sigma_f_k_arr[j] = sigma_f_jk + tf.pow(self.sigma_f[j],2) * tf.eye(X.get_shape().as_list()[0])
            #sigma_f_k_arr[j] = sigma_f_jk + util.var_postive(self.sigma_f[j]) * tf.eye(self.batch_size)
  
        return tf.stack(mu_f_k_arr), tf.stack(sigma_f_k_arr)

    def _build_w_intermediate(self, k, r, X, predict=False):
        mu_w_k_arr = [[0.0 for j in range(self.num_latent)] for i in range(self.num_outputs)]
        sigma_w_k_arr = [[[[0.0 for n2 in range(self.batch_size)] for n1 in range(self.batch_size)] for j in range(self.num_latent)] for i in range(self.num_outputs)]
        kern_w = self.context.kernels[r]['w']

        inducing_locations = self.inducing_locations_arr[r] #M x D
        inducing_locations = tf.expand_dims(inducing_locations, 0) #1 x M x D


        q_means_v = self.q_means_v_arr[r] #K x P x Q x M
        q_covars_v = self.q_covars_v_arr[r] #K x P x Q x M x M
        q_chol_covars_v = self.q_chol_covars_v_arr[r] #K x P x Q x M x M

        x_is_z_flag = False

        for j in range(self.num_latent):
            for i in range(self.num_outputs):
                #compute kernels
                K_xx_w = kern_w[i][j].kernel(X, X, jitter=False) # N x S x S
                K_xz_w = kern_w[i][j].kernel(X, inducing_locations, jitter=x_is_z_flag) # N x S x M
                K_zz_w = kern_w[i][j].kernel(inducing_locations, inducing_locations, jitter=True) # 1 x M x M
                #K_zz_w = tf.tile(K_zz_w, [tf.shape(K_xz_w)[0], 1, 1]) # N x M x M


                m = tf.expand_dims(q_means_v[k,i,j,:], 1) # M x 1
                m = tf.tile(tf.expand_dims(m, 0), [tf.shape(K_xz_w)[0], 1, 1]) # N x M x 1

                S =  q_chol_covars_v[k,i,j,:,:]
                #S = tf.tile(tf.expand_dims(S, 0), [tf.shape(K_xz_w)[0], 1, 1]) # N x M x M

                #Compute $q(W_{pq}) = \int p(W_{pq}|v_{pq}) q(v_{pq}) \mathop{du}$
                #return shapes:
                    #mu_f_jk = N x S x 1
                    #sigma_f_jk = N x S x S
                mu_w_jik, sigma_w_jik = self._build_marginal(m, S, K_zz_w, K_xz_w, K_xx_w, predict)

                mu_w_k_arr[i][j] = mu_w_jik
                sigma_w_k_arr[i][j] = sigma_w_jik

        return tf.stack(mu_w_k_arr), tf.stack(sigma_w_k_arr)


    def _build_standard_intermediate_conditionals(self, k, r, X, predict=False):
        print('_build_intermediate_conditionals')

        mu_w, sig_w = None, None
        mu_f, sig_f = self._build_f_intermediate(k, r, X, predict)
        if 'w' in self.context.kernels[r]:
            mu_w, sig_w = self._build_w_intermediate(k, r, X, predict)

        return mu_f, sig_f, mu_w, sig_w

    def _build_intermediate_conditionals(self, k, r, X, predict=False):
        return self._build_standard_intermediate_conditionals(k, r, X, predict)



