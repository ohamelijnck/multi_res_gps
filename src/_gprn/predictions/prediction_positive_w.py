import tensorflow as tf
import numpy as np
import math
from .. import util
from .. import debugger
from .. sparsity import StandardSparsity
from .. precomputers import Precomputed
from . import PredictionStandard

class PredictionPositiveW(PredictionStandard):
    def __init__(self, context):
        super(PredictionPositiveW, self).__init__(context)

    def _get_expected_values(self, mu_f, sigma_f, mu_w, sigma_w):
        sigma_w = tf.expand_dims(tf.matrix_diag_part(sigma_w), -1)

        m_w = util.safe_exp(mu_w+0.5*sigma_w)
        s_w = util.safe_exp(2*(mu_w+sigma_w))-tf.multiply(m_w,mu_w) 

        return mu_f, sigma_f, m_w, s_w

    def get_expected_values(self, mu_f, sigma_f, mu_w, sigma_w, predict=False):
        if predict:
            return self._get_expected_values(mu_f, sigma_f, mu_w, sigma_w)

        _mu_w = mu_w
        sigma_w_diag = tf.expand_dims(tf.matrix_diag_part(sigma_w), -1)

        cols_repeated = tf.tile(sigma_w_diag, [1, 1, 1, tf.shape(sigma_w_diag)[2]])
        rows_repeated = tf.tile(tf.transpose(sigma_w_diag, perm=[0, 1, 3, 2]), [1, 1, tf.shape(sigma_w_diag)[2], 1])

        sigma_w = tf.Print(sigma_w, [tf.shape(sigma_w)], 'get_expected_values: sigma_w: ', summarize=100)
        sigma_w = tf.Print(sigma_w, [tf.shape(sigma_w_diag)], 'get_expected_values: sigma_w_diag: ', summarize=100)

        sigma_w_diag = tf.expand_dims(tf.matrix_diag_part(sigma_w), -1)

        m_w = util.safe_exp(mu_w+0.5*sigma_w_diag)

        m_w = tf.Print(m_w, [tf.shape(m_w)], 'get_expected_values: m_w: ', summarize=100)

        mu_w_cols_repeated = tf.tile(_mu_w, [1, 1, 1, tf.shape(m_w)[2]])
        mu_w_rows_repeated = tf.tile(tf.transpose(_mu_w, perm=[0, 1, 3, 2]), [1, 1, tf.shape(m_w)[2], 1])


        s_w = util.safe_exp(mu_w_cols_repeated+mu_w_rows_repeated+0.5*(cols_repeated+rows_repeated))*(util.safe_exp(sigma_w)-1)
        s_w = tf.Print(s_w, [sigma_w], 'get_expected_values: sigma_w: ', summarize=100)
        s_w = tf.Print(s_w, [s_w], 'get_expected_values: s_w: ', summarize=100)

        return mu_f, sigma_f, m_w, s_w

    def build_variance_positive_w(self):
        print('build positive w variance')
        num_test = self.x_test.get_shape().as_list()[0]
        total_sum = [0.0 for y in range(self.num_outputs)]


        var_only_flag = True

        precomp_intermediate = [[] for x in range(self.num_components)]
        for l in range(self.num_components):
            mu_f, sigma_f, mu_w, sigma_w = self.sparsity._build_intermediate_conditionals(l, self.a, self.x_test, predict=var_only_flag)
            pi_l = self.q_weights[l]
            precomp_intermediate[l].append(pi_l)
            precomp_intermediate[l].append(mu_f)
            precomp_intermediate[l].append(sigma_f)
            precomp_intermediate[l].append(mu_w)
            precomp_intermediate[l].append(sigma_w)

        for k in range(self.num_components):
            #mu_f = [Q, N, 1]
            #mu_w = [Q, P, N, 1]
            _mu_f, _sigma_f, _mu_w, _sigma_w = self.sparsity._build_intermediate_conditionals(k, self.a, self.x_test, predict=var_only_flag)
            if var_only_flag:
                _sigma_f = tf.expand_dims(_sigma_f, -1)
                _sigma_w = tf.expand_dims(_sigma_w, -1)


            print('_sigma_w: ', _sigma_w)

            _mu_w = tf.Print(_mu_w, [tf.shape(_mu_w)], '_mu_w: ')
            _sigma_w = tf.Print(_sigma_w, [tf.shape(_sigma_w)], '_sigma_w: ')

            pi_k = self.q_weights[k]
            for i in range(self.num_outputs):
                noise_sigma = tf.square(util.var_postive(self.sigma_y[i]))
                mu_wi = _mu_w[i,:,:,:] # Q x N x 1
                sigma_wik = _sigma_w[i,:,:,:] # Q x N x N

                print(mu_f)
                mu_f = _mu_f[:,:, 0] # Q x N
                mu_wi = mu_wi[:,:, 0] # Q x N

                sig_f = _sigma_f[:, :,0] # Q x N
                sig_wi = sigma_wik[:,:,0] # Q x N

                q = 0


                exp_from_quad = util.safe_exp(2*mu_wi+2*sig_wi) # Q x N
                exp_from_mean = util.safe_exp(mu_wi+0.5*sig_wi) # Q x N

                exp_from_quad = tf.expand_dims(exp_from_quad[q, :], -1) #N x 1
                exp_from_mean = tf.expand_dims(exp_from_mean[q, :], -1) #N x 1

                mu_f = tf.expand_dims(mu_f[q, :], -1) #Nx1
                mu_wi = tf.expand_dims(mu_wi[q, :], -1) #Nx1
                sig_f = tf.expand_dims(sig_f[q, :], -1) #Nx1
                sig_wi = tf.expand_dims(sig_wi[q, :], -1) #Nx1

                mu_f= tf.Print(mu_f, [mu_f], 'mu_f: ', summarize=100)
                mu_f= tf.Print(mu_f, [mu_wi], 'mu_wi: ', summarize=100)
                mu_f= tf.Print(mu_f, [sig_f], 'sig_f: ', summarize=100)
                mu_f= tf.Print(mu_f, [sig_wi], 'sig_wi: ', summarize=100)

                s = tf.multiply(mu_f, tf.multiply(exp_from_quad, mu_f))
                s= tf.Print(s, [s], 's2: ', summarize=100)

                s += tf.multiply(exp_from_quad, sig_f)
                s= tf.Print(s, [s], 's3: ', summarize=100)

                expected = np.multiply(exp_from_mean, mu_f)
                s -= tf.multiply(expected, expected)

                s= tf.Print(s, [s], 's4: ', summarize=100)

                if self.context.plot_posterior:
                    s += 0.0
                else:
                    s += noise_sigma

                s= tf.Print(s, [s], 's5: ', summarize=100)

                total_sum[i] += pi_k*tf.squeeze(s)

        total_sum = tf.stack(total_sum, axis=1)
        return total_sum

    def build_f_expected_value_standard(self):
        total_sum = [0.0 for y in range(self.num_outputs)]
        self.f = None
        self.w = None
        for k in range(self.num_components):
            mu_f, sigma_f, mu_w, sigma_w = self.sparsity._build_intermediate_conditionals(k, self.a, self.x_test, predict=True)
            mu_f, sigma_f, mu_w, sigma_w = self.get_expected_values(mu_f, sigma_f, mu_w, sigma_w)

            pi_k = self.q_weights[k]
            for i in range(self.num_outputs):
                for j in range(self.num_latent):
                    mu_fj = mu_f[j,:, :] #Nx1
                    mu_wij = mu_w[i,j,:, :] #Nx1
                    sigma_fk = tf.expand_dims(sigma_f[j,:], -1) #Nx1
                    sigma_wik = tf.expand_dims(sigma_w[i,j,:],-1) #Nx1

                    self.f = mu_fj
                    self.w = mu_wij


                    wf = mu_fj

                    total_sum[i] += pi_k*tf.squeeze(wf)

        total_sum = tf.stack(total_sum, axis=1)

        return total_sum

    def build_w_expected_value_standard(self):
        total_sum = [0.0 for y in range(self.num_outputs)]
        self.f = None
        self.w = None
        for k in range(self.num_components):
            mu_f, sigma_f, mu_w, sigma_w = self.sparsity._build_intermediate_conditionals(k, self.a, self.x_test, predict=True)
            mu_f, sigma_f, mu_w, sigma_w = self.get_expected_values(mu_f, sigma_f, mu_w, sigma_w)

            pi_k = self.q_weights[k]
            for i in range(self.num_outputs):
                for j in range(self.num_latent):
                    mu_fj = mu_f[j,:, :] #Nx1
                    mu_wij = mu_w[i,j,:, :] #Nx1
                    sigma_fk = tf.expand_dims(sigma_f[j,:], -1) #Nx1
                    sigma_wik = tf.expand_dims(sigma_w[i,j,:],-1) #Nx1

                    self.f = mu_fj
                    self.w = mu_wij


                    wf = mu_wij

                    total_sum[i] += pi_k*tf.squeeze(wf)

        total_sum = tf.stack(total_sum, axis=1)

        return total_sum



    def build_graph(self, seperate=False):
        if seperate:
            expected = self.build_f_expected_value_standard()
            var = self.build_w_expected_value_standard()
        else:
            expected = self.build_expected_value()
            var = self.build_variance()
        return expected, var


    def build_variance(self):
        return self.build_variance_positive_w()

