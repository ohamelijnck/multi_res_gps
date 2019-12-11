from . import Likelihood
import tensorflow as tf
import numpy as np
import math
from .. import util
from .. import debugger
from .. sparsity import StandardSparsity

class SingleGPLikelihood(Likelihood):
    def __init__(self, context, r):
        self.context = context
        self.r = r

    def setup_standard(self):
        self.num_train = self.data.get_num_training(source=self.r)
        self.batch_size = self.data.get_batch_size(source=self.r)
        self.num_latent = self.context.num_latent
        self.num_outputs = self.context.num_outputs
        self.num_weights = self.context.num_weights
        self.num_inducing = self.data.get_num_inducing(source=self.r)
        self.kern_f = self.context.kern_f
        self.use_diag_covar_flag = self.context.use_diag_covar_flag
        self.jitter=self.context.jitter

        self.x_train = self.data.get_placeholder(source=self.r, var='x')
        self.y_train = self.data.get_placeholder(source=self.r, var='y')
        self.y_train_nans = self.data.get_placeholder(source=self.r, var='y_nan')

        #=====Components for q(u,v) as an MoG
        self.q_num_components = self.context.num_components
        self.num_sigma = self.num_inducing*(self.num_inducing+1)/2
        
        self.parameters = self.context.parameters

        self.get_standard_variables()
        self.sparsity = StandardSparsity(self.data, self.context)

    def setup(self, data):
        self.data = data
        self.setup_standard()

    def get_standard_variables(self):
        self.q_means_u = self.parameters.get(name='q_means_u_0')
        self.q_covars_u = self.parameters.get(name='q_covars_u_0')

        self.q_weights = self.parameters.get(name='q_weights')

        self.sigma_y = self.parameters.get(name='noise_sigma_0')
        self.sigma_f = self.parameters.get(name='f_sigma')

        self.inducing_locations = self.parameters.get(name='inducing_locations_0')


    def build_graph(self):
        lik = self._build_log_likelihood()
        return lik

    def _build_log_likelihood(self):
        total_sum = 0.0

        c1 = 0
        P = self.num_outputs

        get_sigma = lambda sig : tf.square(util.var_postive(sig))

        p = 0
        for k in range(self.q_num_components):
            pi_k = self.q_weights[k]
            n_p = tf.count_nonzero(self.y_train_nans[:,p])
            c1 -= pi_k * (tf.to_float(n_p)/2)*util.safe_log(2*np.pi*get_sigma(self.sigma_y[p]))
            total_sum = tf.Print(total_sum,  [get_sigma(self.sigma_y[p])], 'sig^2')

        for k in range(self.q_num_components):
            pi_k = self.q_weights[k]
            _mu_f, _sigma_f, _, _ = self.sparsity._build_intermediate_conditionals(k, self.r, self.x_train)

            c2 = -(1/(2*get_sigma(self.sigma_y[p])))
            #c2 = tf.clip_by_value(c2, -1e20, 1e20)

            nan_mask = tf.cast(self.y_train_nans[:,p], dtype=tf.bool)
            y_p = tf.boolean_mask(mask=nan_mask, tensor=self.y_train[:,p])

            #sample the X where we have data
            sample_diag = lambda t: tf.boolean_mask(mask=nan_mask, tensor=t, axis=1)
            #sample_diag = lambda t: t

            mu_f = sample_diag(_mu_f[:, :, 0])

            f = mu_f

            err = y_p - tf.squeeze(f)
            err = tf.reduce_sum(tf.square(err), axis=0)

            total_sum += c2*pi_k*err

        return c1+total_sum
