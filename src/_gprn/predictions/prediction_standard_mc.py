import tensorflow as tf
import numpy as np
import math
from .. import util
from .. import debugger
from .. sparsity import StandardSparsity
from .. precomputers import Precomputed
from . import PredictionStandard

from tensorflow.contrib.distributions import MultivariateNormalTriL
from tensorflow.contrib.distributions import MultivariateNormalDiag 

class PredictionStandardMC(PredictionStandard):
    def __init__(self, context):
        super(PredictionStandardMC, self).__init__(context)
        self.context = context
  
    def build_sample_standard(self):
        total_sum = [0.0 for y in range(self.num_outputs)]
        for i in range(self.num_outputs):
            k = tf.squeeze(tf.random_uniform(shape=[1], minval=0, maxval=self.num_components, dtype=tf.int32))
            mu_f, sigma_f, mu_w, sigma_w = self.sparsity._build_intermediate_conditionals(k, self.x_test)
            mu_f, sigma_f, mu_w, sigma_w = self.get_expected_values(mu_f, sigma_f, mu_w, sigma_w)
            pi_k = self.q_weights[k]
            wf_sum = 0
            latent_sum = [0.0 for j in range(self.num_latent)]
            for j in range(self.num_latent):
                mu_fj = tf.squeeze(mu_f[j,:, :])
                mu_wij = tf.squeeze(mu_w[i,j,:,:])
                sigma_fk = sigma_f[j,:,:]
                sigma_wik = sigma_w[i,j,:,:]

                sigma_fk_chol = tf.cholesky(sigma_fk)
                sig_w_chol = tf.cholesky(sigma_wik)

                f = MultivariateNormalTriL(loc=mu_fj, scale_tril=sigma_fk_chol).sample()
                w = MultivariateNormalTriL(loc=mu_wij, scale_tril=sig_w_chol).sample()

                w = tf.diag(tf.squeeze(w))
                wf = tf.matmul(w, tf.expand_dims(f, 1))

                latent_sum[j] += pi_k*tf.squeeze(wf)

            latent_sum = tf.stack(latent_sum)
            latent_sum = tf.squeeze(tf.reduce_sum(latent_sum, axis=0))

            y = MultivariateNormalDiag(loc=latent_sum, scale_diag=tf.sqrt(util.var_postive(self.sigma_y[i]))*tf.ones(tf.shape(self.x_test)[0])).sample()
            y = tf.Print(y, [latent_sum], 'latent_sum')
            y = tf.Print(y, [y], 'y')

            #y = latent_sum
            total_sum[i] += y
            
        total_sum = tf.stack(total_sum, axis=1)
        return total_sum

    def build_graph(self):
        sample = self.build_sample_standard()
        return sample



