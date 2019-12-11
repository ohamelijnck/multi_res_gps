from . import Likelihood
import tensorflow as tf
import numpy as np
import math
from .. import util
from .. import debugger
from .. sparsity import StandardSparsity, MRSparsity

class GP_Aggr_Likelihood(Likelihood):
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
        self.kern_w = self.context.kern_w
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
        self.sparsity = MRSparsity(self.data, self.context)    

    def setup(self, data):
        self.data = data

        self.setup_standard()

    def get_standard_variables(self):

        self.q_weights = self.parameters.get(name='q_weights')

        self.noise_sigma = self.parameters.get(name='noise_sigma_{r}'.format(r=self.r))
        self.sigma_f = self.parameters.get(name='f_sigma')

    def build_graph(self):
        lik = self._build_log_likelihood()
        return lik

    def _build_log_likelihood(self):
        print('ell standard')
        total_sum = 0.0
        M = self.data.get_raw(self.r, 'M')

        c1 = 0
        P = self.num_outputs

        get_sigma = lambda sig : tf.square(util.var_postive(sig))

        for k in range(self.q_num_components):
            pi_k = self.q_weights[k]
            for p in range(1):
                n_p = tf.count_nonzero(self.y_train_nans[:,p])
                c1 -= pi_k * (tf.to_float(n_p)/2)*util.safe_log(2*np.pi*get_sigma(self.noise_sigma[p]))
        c2 = -(1/(2*get_sigma(self.noise_sigma[p])))

        for k in range(self.q_num_components):
            pi_k = self.q_weights[k]
            for p in range(self.num_outputs):
                nan_mask = tf.cast(self.y_train_nans[:,p], dtype=tf.bool)
                y_p = tf.boolean_mask(mask=nan_mask, tensor=self.y_train[:,p])

                _mu_f, _sigma_f, _, _ = self.sparsity._build_intermediate_conditionals(k, 0, self.x_train, predict=True)
                mu_f = _mu_f[0, :, :, 0] # N x S

                y_m = tf.expand_dims(y_p, -1)

                f = tf.reduce_mean(mu_f, axis=1)[:, None]
                f = tf.boolean_mask(mask=nan_mask, tensor=f, axis=0)


                err = tf.reduce_sum(tf.square(y_m-f))

                total_sum += (err)
        return c1+c2*total_sum


