import tensorflow as tf
import numpy as np
import math
from .. import util
from .. import debugger
from .. sparsity import StandardSparsity, MRSparsity
from . import Prediction

from tensorflow.contrib.distributions import MultivariateNormalTriL
from tensorflow.contrib.distributions import MultivariateNormalFullCovariance
from tensorflow.contrib.distributions import MultivariateNormalDiag 
class PredictionMRStandard(Prediction):
    def __init__(self, context):
        super(PredictionMRStandard, self).__init__()
        self.context = context
        self.r = 0

    def setup_standard(self, x_test, num_test, data):
        self.x_test = x_test
        self.num_test = num_test
        self.data = data

        self.num_latent = self.context.num_latent
        self.num_outputs = self.context.num_outputs
        self.num_weights = self.context.num_weights
        self.num_components = self.context.num_components
        self.num_inducing = self.data.get_num_inducing(source=0)
        self.kern_f = self.context.kern_f
        self.kern_w = self.context.kern_w
        self.use_diag_covar_flag = self.context.use_diag_covar
        self.jitter = self.context.jitter

        self.parameters = self.context.parameters
        #self.sparsity = StandardSparsity(self.data, self.context)
        self.sparsity = MRSparsity(self.data, self.context)

    def setup(self, x_test, num_test, data, a=0, r=0):
        self.a=a
        self.r=r
        self.setup_standard(x_test, num_test, data)
        self.get_variables()

    def get_variables_standard(self):

        self.q_means_u = self.parameters.get(name='q_means_u_{a}'.format(a=self.a))
        self.q_means_v = self.parameters.get(name='q_means_v_{a}'.format(a=self.a))
        self.q_covars_u = self.parameters.get(name='q_covars_u_{a}'.format(a=self.a))
        self.q_covars_v = self.parameters.get(name='q_covars_v_{a}'.format(a=self.a))

        self.q_weights = self.parameters.get(name='q_weights')

        self.sigma_y = self.parameters.get(name='noise_sigma_{r}'.format(r=self.r))
        self.sigma_f = self.parameters.get(name='f_sigma')

        self.sigma_f = self.parameters.get(name='f_sigma')
        self.likelihood_weights = self.parameters.get(name='likelihood_weights')

        self.inducing_locations = self.parameters.get(name='inducing_locations_{a}'.format(a=self.a))

    def get_variables(self):
        self.get_variables_standard()

    def get_expected_values(self, mu_f, sigma_f, mu_w, sigma_w, predict=False):
        return mu_f, sigma_f, mu_w, sigma_w

    def build_variance_standard(self):
        print('build variance')
        num_test = self.x_test.get_shape().as_list()[0]
        total_sum = [0.0 for y in range(self.num_outputs)]
        full_var_flag = False

        x_test = tf.expand_dims(self.x_test, 1) #N x 1 x D

        precomp_intermediate = [[] for x in range(self.num_components)]
        for l in range(self.num_components):
            #_mu_f = Q x N x 1 x 1
            #_sigma_f = Q x N x 1 x 1
            #_mu_w = P x Q x N x 1 x 1
            #_sigma_w = P x Q x N x 1 x 1
            mu_f, sigma_f, mu_w, sigma_w = self.sparsity._build_intermediate_conditionals(l, 0, x_test, predict=True)
            mu_f, sigma_f, mu_w, sigma_w = self.get_expected_values(mu_f, sigma_f, mu_w, sigma_w)
            pi_l = self.q_weights[l]
            precomp_intermediate[l].append(pi_l)
            precomp_intermediate[l].append(mu_f)
            precomp_intermediate[l].append(sigma_f)
            precomp_intermediate[l].append(mu_w)
            precomp_intermediate[l].append(sigma_w)


        for k in range(self.num_components):



            for i in range(self.num_outputs):
                #mu_f = Q x N x 1 x 1
                #sigma_f = Q x N x 1 x 1
                #mu_w = P x Q x N x 1 x 1
                #sigma_w = P x Q x N x 1 x 1
                pi_k, mu_f, sigma_f, mu_w, sigma_w = precomp_intermediate[k]
                
                if self.context.plot_posterior:
                    noise_sigma = 0.0
                else:
                    noise_sigma = tf.square(util.var_postive(self.sigma_y[i]))

                mu_f = mu_f[:,:, 0, :] # Q x N x 1
                sigma_f = sigma_f[:,:, 0, :] # Q x N x 1
                mu_wi = mu_w[i,:,:, 0, :] # Q x N x 1
                sigma_wik = sigma_w[i,:,:, 0, :] # Q x N x 1

                s = noise_sigma

                s += tf.reduce_sum(tf.multiply(sigma_f, sigma_wik), axis=0) # N x 1
                s += tf.reduce_sum(tf.multiply(tf.multiply(mu_f, mu_f), sigma_wik), axis=0) # N x 1
                s += tf.reduce_sum(tf.multiply(tf.multiply(mu_wi, mu_wi), sigma_f), axis=0) # N x 1
                
                s = s[:, 0] # N

                if False:
                    a =  tf.reduce_sum(tf.multiply(mu_wi[:, :, 0],  mu_f[:,:,0]), axis=0)
                    
                    a_sum = 0
                    for l in range(self.num_components):
                        pi_l = precomp_intermediate[l][0]
                        mu_f_l = precomp_intermediate[l][1][:,:,0]
                        mu_w_l = precomp_intermediate[l][3][i,:,:,0]

                        a_sum += pi_l*tf.reduce_sum(tf.multiply(mu_w_l, mu_f_l), axis=0)

                    #a_sum += pi_l*tf.matmul(mu_w_l, mu_f_l)
                #s += tf.multiply(a, a_sum) - tf.multiply(a, a_sum)
                #s += tf.multiply(a, a) - tf.multiply(a, a_sum)

                total_sum[i] += pi_k*s

        total_sum = tf.stack(total_sum, axis=1)
        return total_sum

    def build_variance(self):
        return self.build_variance_standard()

    def build_expected_value_standard(self):
        total_sum = [0.0 for y in range(self.num_outputs)]

        for k in range(self.num_components):
            x_test = tf.expand_dims(self.x_test, 1) #N x 1 x D
            print('x_test: ', x_test)
            #_mu_f = Q x N x 1 x 1
            #_sigma_f = Q x N x 1 x 1
            #_mu_w = P x Q x N x 1 x 1
            #_sigma_w = P x Q x N x 1 x 1
            mu_f, sigma_f, mu_w, sigma_w = self.sparsity._build_intermediate_conditionals(k, 0, x_test, predict=True)
            mu_f, sigma_f, mu_w, sigma_w = self.get_expected_values(mu_f, sigma_f, mu_w, sigma_w, predict=True)

            pi_k = self.q_weights[k]
            for i in range(self.num_outputs):
                for j in range(self.num_latent):
                    mu_fj = mu_f[j,:, :, 0] #Nx1
                    mu_wij = mu_w[i,j,:, :, 0] #Nx1
                    
                    wf = tf.multiply(mu_fj, mu_wij)
                    total_sum[i] += pi_k*tf.squeeze(wf)

        total_sum = tf.stack(total_sum, axis=1)

        return total_sum

    def build_sample_standard(self):
        total_sum = [0.0 for y in range(self.num_outputs)]
        r=self.r
        for i in range(self.num_outputs):
            k = tf.squeeze(tf.random_uniform(shape=[1], minval=0, maxval=self.num_components, dtype=tf.int32))
            full_cov = True
            mu_f, sigma_f, mu_w, sigma_w = self.sparsity._build_intermediate_conditionals(k, r, self.x_test, predict=not full_cov)
            #mu_f, sigma_f, mu_w, sigma_w = self.get_expected_values(mu_f, sigma_f, mu_w, sigma_w)


            pi_k = self.q_weights[k]
            wf_sum = 0
            latent_sum = [0.0 for j in range(self.num_latent)]
            for j in range(self.num_latent):
                if full_cov:
                    print('mu_f: ', mu_f)
                    print('mu_w: ', mu_w)
                    print('sigma_f: ', sigma_f)
                    print('sigma_w: ', sigma_w)
                    mu_fj = tf.squeeze(mu_f[j,:, :])
                    mu_wij = tf.squeeze(mu_w[i,j,:,:])
                    sigma_fk = sigma_f[j,:,:]
                    sigma_wik = sigma_w[i,j,:,:]
                else:
                    mu_fj = tf.squeeze(mu_f[j,:, :])
                    mu_wij = tf.squeeze(mu_w[i,j,:,:])
                    sigma_fk = sigma_f[j,:,:]
                    sigma_wik = sigma_w[i,j,:,:]

                sigma_fk_chol = tf.cast(tf.cholesky(tf.cast(util.add_jitter(sigma_fk, 1e-4), tf.float64)), tf.float32)
                sig_w_chol = tf.cast(tf.cholesky(tf.cast(util.add_jitter(sigma_wik, 1e-4), tf.float64)), tf.float32)

                f = MultivariateNormalTriL(loc=mu_fj, scale_tril=sigma_fk_chol).sample()
                w = MultivariateNormalTriL(loc=mu_wij, scale_tril=sig_w_chol).sample()

                w = tf.diag(tf.squeeze(w))
                wf = tf.matmul(w, tf.expand_dims(f, 1))

                latent_sum[j] += pi_k*tf.squeeze(wf)

            latent_sum = tf.stack(latent_sum)
            latent_sum = tf.squeeze(tf.reduce_sum(latent_sum, axis=0))


            if self.context.plot_posterior:
                y = latent_sum
            else:  
                noise_sigma = tf.square(util.var_postive(self.sigma_y[i]))
                y = MultivariateNormalDiag(loc=latent_sum, scale_diag=tf.sqrt(noise_sigma)*tf.ones(tf.shape(self.x_test)[0])).sample()

            #y = latent_sum
            total_sum[i] += y
            
        total_sum = tf.stack(total_sum, axis=1)
        return total_sum

    def build_f_expected_value_standard(self):
        total_sum = [0.0 for y in range(self.num_latent)]
        self.f = None
        self.w = None
        for k in range(self.num_components):
            x_test = tf.expand_dims(self.x_test, 1) #N x 1 x D
            print('x_test: ', x_test)
            #_mu_f = Q x N x M x 1
            #_sigma_f = Q x N x M x M
            #_mu_w = P x Q x N x M x 1
            #_sigma_w = P x Q x N x M x M
            mu_f, sigma_f, mu_w, sigma_w = self.sparsity._build_intermediate_conditionals(k, 0, x_test, predict=True)

            print('mu_f: ', mu_f)
            print('sigma_f: ', sigma_f)
            print('mu_w: ', mu_w)
            print('sigma_w: ', sigma_w)
            mu_f, sigma_f, mu_w, sigma_w = self.get_expected_values(mu_f, sigma_f, mu_w, sigma_w)


            pi_k = self.q_weights[k]
            for j in range(self.num_latent):
                mu_fj = mu_f[j,:, :] #Nx1
                sigma_fk = tf.expand_dims(sigma_f[j,:], -1) #Nx1

                self.f = mu_fj

                wf = self.f

                total_sum[j] += pi_k*tf.squeeze(wf)

        total_sum = tf.stack(total_sum, axis=0)

        return total_sum

    def build_w_expected_value_standard(self):
        total_sum = [[0.0 for y in range(self.num_latent)] for i in range(self.num_outputs)]
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

                    total_sum[i][j] = tf.squeeze(wf)

        total_sum = tf.stack(total_sum, axis=0)
        return total_sum


    def build_sample_graph(self):
        return self.build_sample_standard()

    def build_expected_value(self):
        return self.build_expected_value_standard()

    def build_graph(self, seperate=False):
        if seperate:
            expected = self.build_f_expected_value_standard()
            #var = self.build_w_expected_value_standard()
        else:
            expected = self.build_expected_value()
            var = self.build_variance()

        return expected, var



