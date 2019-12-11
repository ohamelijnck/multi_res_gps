import numpy as np
import tensorflow as tf

from . import Model
from .. import util
from .. import Parameters

class Base(Model):
    def __init__(self, context):
        self.context = context
        self.mean_v_scale = 1.0
        self.covar_v_scale = 1.0

    def setup(self):
        pass

    def _load_variables_from_context(self):
        #self.model = self.context.model
        self.num_latent = self.context.num_latent
        self.num_outputs = self.context.num_outputs
        self.num_components = self.context.num_components
        self.use_diag_covar = self.context.use_diag_covar_flag
        self.train_inducing_points_flag = self.context.train_inducing_points_flag
        self.sigma_y_init = self.context.noise_sigmas[0][0]
        self.sigma_y_train_flag = self.context.noise_sigmas[0][1]
        self.jitter = self.context.jitter

        self.parameters = self.context.parameters

    def _setup_base_variables(self, a=0):
        print(a)
        self.input_dim = self.data.get_input_dim(source=a)
        self.num_train = self.data.get_num_training(source=a)
        self.num_inducing = self.data.get_num_inducing(source=a)


        self.num_weights = self.num_latent*self.num_outputs

        self.q_num_components = self.num_components
        if  self.use_diag_covar:
            self.num_sigma = self.num_inducing
        else:
            self.num_sigma = int(self.num_inducing*(self.num_inducing+1)/2)

        self.context.num_sigma = self.num_sigma

        self.parameters.create(name='inducing_locations_{a}'.format(a=a), init=self.data.get_inducing_points_from_source(a).astype(np.float32), trainable=self.train_inducing_points_flag,  scope=Parameters.VARIATIONAL_SCOPE)

        self.parameters.create(name='q_means_u_{a}'.format(a=a), init=tf.random_uniform([self.q_num_components, self.num_latent, self.num_inducing], 0, 1, tf.float32, seed=0),  trainable=True, scope=Parameters.VARIATIONAL_SCOPE)
        self.parameters.create(name='q_covars_u_{a}_raw'.format(a=a), init=0.1*tf.random_uniform([self.q_num_components, self.num_latent, self.num_sigma], 0.0, 0.5, dtype=tf.float32, seed=0),  trainable=True, scope=Parameters.VARIATIONAL_SCOPE)
        self.parameters.load_posterior_covariance(name='q_covars_u_{a}'.format(a=a), from_name='q_covars_u_{a}_raw'.format(a=a), shape=[self.q_num_components, self.num_latent, self.num_sigma], n=self.num_inducing)
        self.parameters.load_posterior_cholesky(name='q_cholesky_u_{a}'.format(a=a), from_name='q_covars_u_{a}_raw'.format(a=a), shape=[self.q_num_components, self.num_latent, self.num_sigma], n=self.num_inducing)

        if True:
            if self.context.constant_w:
                with tf.variable_scope(Parameters.VARIATIONAL_SCOPE, reuse=None):
                    w = tf.get_variable(name='q_means_v_{a}'.format(a=a), initializer=self.mean_v_scale*tf.random_uniform([self.q_num_components, self.num_outputs, self.num_latent, 1], 0, 1, tf.float32, seed=1))
                    w = tf.tile(w, [1, 1, 1, self.num_inducing])

                    w_var_raw = tf.get_variable(name='q_covars_v_{a}_raw'.format(a=a),initializer=self.covar_v_scale*tf.random_uniform([self.q_num_components, self.num_outputs, self.num_latent, 1], 0, 1, tf.float32, seed=1))
                    w_diag = tf.tile(w_var_raw, [1, 1, 1, self.num_inducing])
                    w_var_diag = tf.matrix_diag(w_diag)

                self.parameters.save(name='q_means_v_{a}'.format(a=a), var=w)
                self.parameters.save(name='q_covars_v_{a}'.format(a=a), var=w_var_diag)
                self.parameters.save(name='q_covars_v_{a}_raw'.format(a=a), var=w_var_raw)
                self.parameters.save(name='q_cholesky_v_{a}'.format(a=a), var=tf.cholesky(w_var_diag))
            else:
                self.parameters.create(name='q_means_v_{a}'.format(a=a), init=self.mean_v_scale*tf.random_uniform([self.q_num_components, self.num_outputs, self.num_latent, self.num_inducing], 0, 1, tf.float32, seed=1),  trainable=True, scope=Parameters.VARIATIONAL_SCOPE)
                self.parameters.create(name='q_covars_v_{a}_raw'.format(a=a), init=self.covar_v_scale*tf.random_uniform([self.q_num_components, self.num_outputs, self.num_latent, self.num_sigma], -0.1, 0.1, tf.float32, seed=1),  trainable=True, scope=Parameters.VARIATIONAL_SCOPE)
                self.parameters.load_posterior_covariance(name='q_covars_v_{a}'.format(a=a), from_name='q_covars_v_{a}_raw'.format(a=a), shape=[self.q_num_components, self.num_outputs, self.num_latent, self.num_sigma], n=self.num_inducing)
                self.parameters.load_posterior_cholesky(name='q_cholesky_v_{a}'.format(a=a), from_name='q_covars_v_{a}_raw'.format(a=a), shape=[self.q_num_components, self.num_outputs, self.num_latent, self.num_sigma], n=self.num_inducing)


        if a == 0:
            self.parameters.create(name='q_raw_weights'.format(a=a), init=tf.ones([self.q_num_components], tf.float32), trainable=True, scope=Parameters.HYPER_SCOPE)
            self.parameters.load_posterior_component_weights()

        self.parameters.create(name='noise_sigma_{a}'.format(a=a), init=self.context.noise_sigmas[0][0], trainable=self.context.noise_sigmas[0][1], scope=Parameters.HYPER_SCOPE)

        same_sig_f = False

        if a==0:
            if same_sig_f is False:
                self.parameters.create(name='f_sigma', init=0.0*tf.ones([self.num_latent]), trainable=False, scope=Parameters.HYPER_SCOPE)
            else:
                self.parameters.create(name='f_sigma', init=1.0, trainable=True, scope=Parameters.HYPER_SCOPE)

    def setup_variables(self):
        self._load_variables_from_context()
        self._setup_base_variables(a=0)

    def setup_base_kernels(self):
        print('==============setup kernel================')
        for r_kernels in self.context.kernels:
            for k in r_kernels['f']:
                k.setup(self.context)
            if 'w' in r_kernels:
                for row in r_kernels['w']:
                    for k in row:
                        k.setup(self.context)

    def setup_kernels(self):
        self.setup_base_kernels()

    def build_elbo_graph(self):
        self.elbo.setup(self.data)
        return self.elbo.build_graph()

    def build_likelihood_graph(self):
        print('graph')
        self.likelihood.setup(self.data)
        return self.likelihood.build_graph()

    def build_prediction_graph(self, x_test, num_test, a=0, r=0, seperate=False):
        self.predictor.setup(x_test, num_test, self.data, a=a, r=r)
        return self.predictor.build_graph(seperate=seperate)

    def build_sample_graph(self, x_test, num_test, a=0, r=0):
        self.predictor.setup(x_test, num_test, self.data, a=a, r=r)
        return self.predictor.build_sample_graph()

