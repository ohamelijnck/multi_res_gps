import numpy as np
import tensorflow as tf
import math
from .. import util
from tensorflow.contrib.distributions import MultivariateNormalTriL
from tensorflow.contrib.distributions import MultivariateNormalDiag
from ..sparsity import StandardSparsity
from ..precomputed import Precomputed

class Prediction(object):
    def __init__(self, model, x_test, num_test, data, context):
        self.data = data

        self.model = model
        self.context = context
        self.x_test = x_test
        self.num_test = num_test
        self.num_latent = self.context.num_latent
        self.num_outputs = self.context.num_outputs
        self.num_weights = self.context.num_weights
        self.num_components = self.context.num_components
        self.num_inducing = self.data.get_num_inducing(source=0)
        self.kern_f = self.context.kern_f
        self.kern_w = self.context.kern_w
        self.use_diag_covar_flag = self.context.use_diag_covar
        self.jitter = self.context.jitter

        self.precomputed = Precomputed(self.data, self.context)
        self.get_variables()
        self.sparsity = StandardSparsity(self.data, self.context, self.precomputed)


    def get_variables(self):
        with tf.variable_scope("parameters", reuse=True):
            self.q_means_u = tf.get_variable(name='q_means_u')
            self.q_means_v = tf.get_variable(name='q_means_v')

            self.q_raw_weights = tf.get_variable(name='q_raw_weights')
            self.q_weights = util.safe_exp(self.q_raw_weights) / tf.reduce_sum(util.safe_exp(self.q_raw_weights))

            self.sigma_y = tf.get_variable(name='y_sigma')
            self.sigma_f = tf.get_variable(name='f_sigma')

            self.inducing_locations = tf.get_variable(name='inducing_locations')


        self.q_covars_u, self.q_covars_v = self.precomputed.get_covars()

    def build_variance(self):
        print('build variance')
        num_test = self.x_test.get_shape().as_list()[0]
        total_sum = [0.0 for y in range(self.num_outputs)]

        precomp_intermediate = [[] for x in range(self.num_components)]
        for l in range(self.num_components):
            mu_f, sigma_f, mu_w, sigma_w = self.sparsity._build_intermediate_conditionals(l, self.x_test)
            mu_f, sigma_f, mu_w, sigma_w = self.model.get_expected_values(mu_f, sigma_f, mu_w, sigma_w)
            pi_l = self.q_weights[l]
            precomp_intermediate[l].append(pi_l)
            precomp_intermediate[l].append(mu_f)
            precomp_intermediate[l].append(sigma_f)
            precomp_intermediate[l].append(mu_w)
            precomp_intermediate[l].append(sigma_w)


        for k in range(self.num_components):
            #mu_f = [Q, N, 1]
            #mu_w = [Q, P, N, 1]
            mu_f, sigma_f, mu_w, sigma_w = self.sparsity._build_intermediate_conditionals(k, self.x_test)
            mu_f, sigma_f, mu_w, sigma_w = self.model.get_expected_values(mu_f, sigma_f, mu_w, sigma_w)
            pi_k = self.q_weights[k]
            for i in range(self.num_outputs):
                mu_wi = mu_w[i,:,:,:] # Q x N x 1
                sigma_wik = sigma_w[i,:,:,:] # Q x N x N

                s = tf.reduce_sum(tf.multiply(tf.matrix_diag_part(sigma_f), tf.matrix_diag_part(sigma_wik)), axis=0)
                s += util.var_postive(self.sigma_y[i])
                s += tf.reduce_sum(tf.multiply(tf.multiply(mu_f[:,:,0], mu_f[:,:,0]), tf.matrix_diag_part(sigma_wik)), axis=0)
                s += tf.reduce_sum(tf.multiply(tf.multiply(mu_wi[:, :, 0], mu_wi[:, :, 0]), tf.matrix_diag_part(sigma_f)), axis=0)

                a =  tf.diag_part(tf.matmul(mu_wi[:, :, 0], mu_f[:,:,0], transpose_a=True))
                a_sum = 0
                for l in range(self.num_components):
                    pi_l = precomp_intermediate[l][0]
                    mu_f_l = precomp_intermediate[l][1][:,:,0]
                    mu_w_l = precomp_intermediate[l][3][i,:,:,0]

                    a_sum += pi_l*tf.diag_part(tf.matmul(mu_w_l, mu_f_l, transpose_a=True))

                    #a_sum += pi_l*tf.matmul(mu_w_l, mu_f_l)
                s += tf.multiply(a, a_sum) - tf.multiply(a, a_sum)

                total_sum[i] += pi_k*s

        total_sum = tf.stack(total_sum, axis=1)
        return total_sum



    def build_w_f_separate(self):
        mu_f, sigma_f, mu_w, sigma_w = self.sparsity._build_intermediate_conditionals(0, self.x_test)
        mu_f, sigma_f, mu_w, sigma_w = self.model.get_expected_values(mu_f, sigma_f, mu_w, sigma_w)
        i = 0
        j = 0
        mu_fj = tf.squeeze(mu_f[j,:, :])
        mu_wij = tf.squeeze(mu_w[i,j,:,:])
        return mu_fj, mu_wij

    def build_expected_value(self):
        total_sum = [0.0 for y in range(self.num_outputs)]
        self.f = None
        self.w = None
        for k in range(self.num_components):
            mu_f, sigma_f, mu_w, sigma_w = self.sparsity._build_intermediate_conditionals(k, self.x_test)
            mu_f, sigma_f, mu_w, sigma_w = self.model.get_expected_values(mu_f, sigma_f, mu_w, sigma_w)
            pi_k = self.q_weights[k]
            for i in range(self.num_outputs):
                for j in range(self.num_latent):
                    mu_fj = tf.squeeze(mu_f[j,:, :])
                    mu_wij = tf.squeeze(mu_w[i,j,:,:])
                    sigma_fk = sigma_f[j,:,:]
                    sigma_wik = sigma_w[i,j,:,:]

                    self.f = mu_fj
                    self.w = mu_wij

                    w_kij_arr = tf.diag(tf.squeeze(mu_wij))
                    wf = tf.matmul(w_kij_arr, tf.expand_dims(mu_fj, 1))

                    total_sum[i] += pi_k*tf.squeeze(wf)

        total_sum = tf.stack(total_sum, axis=1)

        return total_sum

    def build_graph(self):
        expected = self.build_expected_value()
        var = self.build_variance()

        return expected, var

    def build_graph_monte_carlo(self):
        f_k_arr = [0.0 for j in range(self.num_latent)]
        w_k_arr = [[0.0 for j in range(self.num_latent)] for i in range(self.num_outputs)]
        total_sum = [0.0 for y in range(self.num_outputs)]
        num_test = self.x_test.get_shape().as_list()[0]
        print('build_graph')
        for i in range(self.num_outputs):
            wf_sum = 0

            k = util.sample_index_with_prob_weights(self.q_weights, self.num_outputs)                
            mu_f, sigma_f, mu_w, sigma_w = self.sparsity._build_intermediate_conditionals(k, self.x_test)
            mu_f, sigma_f, mu_w, sigma_w = self.model.get_expected_values(mu_f, sigma_f, mu_w, sigma_w)
            pi_k = self.q_weights[k]

            latent_sum = [0.0 for j in range(self.num_latent)]
            for j in range(self.num_latent):
                mu_fj = tf.squeeze(mu_f[j,:, :])
                mu_wij = tf.squeeze(mu_w[i,j,:,:])
                sigma_fk = sigma_f[j,:,:]
                sigma_wik = sigma_w[i,j,:,:]
                
                sigma_fk_chol = tf.cholesky(sigma_fk)
                sig_w = tf.cholesky(sigma_wik)

                f_kj_arr = MultivariateNormalTriL(loc=mu_fj, scale_tril=sigma_fk_chol).sample()
                w_kij_arr = MultivariateNormalTriL(loc=mu_wij, scale_tril=sig_w).sample()

                f_kj_arr = tf.expand_dims(tf.squeeze(f_kj_arr), 1)
                w_kij_arr = tf.diag(tf.squeeze(w_kij_arr))

                wf = tf.matmul(w_kij_arr, f_kj_arr)
                latent_sum[j] = wf

            latent_sum = tf.reduce_sum(latent_sum, axis=0)

            wf_sum += tf.squeeze(latent_sum)
                
            y = MultivariateNormalDiag(loc=wf_sum, scale_identity_multiplier=util.var_postive(self.sigma_y)).sample()
            #y = tf.expand_dims(y, 1)
            y = tf.squeeze(y)

            total_sum[i] += y

        total_sum = tf.stack(total_sum, axis=1)
        return total_sum

