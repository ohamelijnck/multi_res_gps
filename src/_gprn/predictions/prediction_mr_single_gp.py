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

class PredictionMRSingleGP(Prediction):
    def __init__(self, context):
        super(PredictionMRSingleGP, self).__init__()
        self.context = context

    def setup_standard(self, x_test, num_test, data):
        self.x_test = x_test
        self.num_test = num_test
        self.data = data

        self.num_latent = self.context.num_latent
        self.num_outputs = self.context.num_outputs
        self.num_weights = self.context.num_weights
        self.num_components = self.context.num_components
        self.num_inducing = self.data.get_num_inducing(source=0)
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
        self.q_covars_u = self.parameters.get(name='q_covars_u_{a}'.format(a=self.a))

        self.q_weights = self.parameters.get(name='q_weights')

        self.sigma_y = self.parameters.get(name='noise_sigma_{r}'.format(r=self.r))
        self.sigma_f = self.parameters.get(name='f_sigma')
        self.likelihood_weights = self.parameters.get(name='likelihood_weights')

        self.inducing_locations = self.parameters.get(name='inducing_locations_{a}'.format(a=self.a))

    def get_variables(self):
        self.get_variables_standard()

    def get_expected_values(self, mu_f, sigma_f):
        return mu_f, sigma_f

    def build_variance_standard(self):
        print('build variance')
        num_test = self.x_test.get_shape().as_list()[0]
        total_sum = [0.0 for y in range(self.num_outputs)]
        r=self.r

        full_var_flag=False

        precomp_intermediate = [[] for x in range(self.num_components)]
        for l in range(self.num_components):
            x_test = tf.expand_dims(self.x_test, 1) #N* x 1 x D
            mu_f, sigma_f, _, _ = self.sparsity._build_intermediate_conditionals(l, self.a, x_test, predict=not full_var_flag)
            mu_f, sigma_f = self.get_expected_values(mu_f, sigma_f)
            pi_l = self.q_weights[l]
            precomp_intermediate[l].append(pi_l)
            precomp_intermediate[l].append(mu_f)
            precomp_intermediate[l].append(sigma_f)

        if self.context.plot_posterior:
            noise_sigma = 0.0
        else:
            noise_sigma = tf.square(util.var_postive(self.sigma_y[0]))

        noise_sigma = tf.Print(noise_sigma, [noise_sigma, tf.square(util.var_postive(self.sigma_y[0]))], 'noise_sigma: ')
        noise_sigma = tf.Print(noise_sigma, [noise_sigma], 'noise_sigma: ')

        for k in range(self.num_components):
            #mu_f = [Q, N, 1]
            #mu_w = [Q, P, N, 1]

            x_test = tf.expand_dims(self.x_test, 1) #N* x 1 x D
            #mu_f = #Q * N* x 1
            #sigma_f = #Q * N* x 1 x 1
            mu_f, sigma_f, _, _ = self.sparsity._build_intermediate_conditionals(k, self.a, x_test, predict=not full_var_flag)
            mu_f, sigma_f = self.get_expected_values(mu_f, sigma_f)


            
            pi_k = self.q_weights[k]
            i = 0
            j = 0
            mu_f = mu_f[j, :, 0] # N x 1
            #sigma_f = tf.matrix_diag_part(sigma_f[j, :])
            sigma_f = sigma_f[j, :, :, 0] # N x 1

            s = sigma_f[:, 0]
            s = tf.Print(s, [noise_sigma], 'noise_sigma: ')
            s = tf.Print(s, [tf.shape(sigma_f)], 'tf.shape(sigma_f): ')
            s += noise_sigma

            total_sum[i] += pi_k*s


        if full_var_flag:
            total_sum = total_sum[0]
        else:
            total_sum = tf.stack(total_sum, axis=1)

        total_sum = tf.Print(total_sum, [self.likelihood_weights[self.r]], 'self.likelihood_weights[self.r]: ')
        if full_var_flag:
            total_sum = tf.Print(total_sum, [tf.shape(total_sum)], 'total_sum: ')
            return tf.expand_dims(tf.diag_part(total_sum), -1)

        return total_sum

    def build_variance(self):
        return self.build_variance_standard()

    def build_expected_value_standard(self):
        total_sum = [0.0 for y in range(self.num_outputs)]
        full_var_flag = False

        #single gp, no latent functions or tasks
        j = 0
        i =0

        for k in range(self.num_components):
            x_test = tf.expand_dims(self.x_test, 1) #N* x 1 x D
            #mu_f = #Q * N* x 1
            #sigma_f = #Q * N* x 1 x 1
            mu_f, sigma_f, _, _ = self.sparsity._build_intermediate_conditionals(k, self.a, x_test, predict=not full_var_flag)
            mu_f, sigma_f = self.get_expected_values(mu_f, sigma_f)
            pi_k = self.q_weights[k]

            mu_fj = tf.squeeze(mu_f[j,:, :])

            total_sum[i] += pi_k*tf.squeeze(mu_fj)
            print(total_sum[i])

        print('total_sum: ', total_sum)
        total_sum = tf.stack(total_sum, axis=1)

        return total_sum

    def build_sample_single_gp(self):
        total_sum = [0.0 for y in range(self.num_outputs)]
        i = 0
        j = 0   
        k = tf.squeeze(tf.random_uniform(shape=[1], minval=0, maxval=self.num_components, dtype=tf.int32))

        noise_sigma = tf.square(util.var_postive(self.sigma_y[0]))

        mu_f, sigma_f, _, _ = self.sparsity._build_intermediate_conditionals(k, 0,  self.x_test, predict=False)
        mu_f, sigma_f = self.get_expected_values(mu_f, sigma_f)
        pi_k = self.q_weights[k]
        wf_sum = 0
        latent_sum = [0.0 for j in range(self.num_latent)]

        mu_fj = tf.squeeze(mu_f[j,:, :])
        sigma_fk = sigma_f[j,:,:]

        sigma_fk_chol = tf.cast(tf.cholesky(tf.cast(util.add_jitter(sigma_fk, 1e-4), tf.float64)), tf.float32)

        f = MultivariateNormalTriL(loc=mu_fj, scale_tril=sigma_fk_chol).sample()

        latent_sum[j] += pi_k*tf.squeeze(f)

        latent_sum = tf.stack(latent_sum)
        latent_sum = tf.squeeze(tf.reduce_sum(latent_sum, axis=0))

        y = MultivariateNormalDiag(loc=latent_sum, scale_diag=tf.sqrt(noise_sigma)*tf.ones(tf.shape(self.x_test)[0])).sample()
        total_sum[i] += y
            
        total_sum = tf.stack(total_sum, axis=1)
        return total_sum

    def build_sample_graph(self):
        return self.build_sample_single_gp()

    def build_expected_value(self):
        return self.build_expected_value_standard()

    def build_graph(self, seperate=False):

        expected = self.build_expected_value()
        var = self.build_variance()
        #var = expected

        return expected, var




