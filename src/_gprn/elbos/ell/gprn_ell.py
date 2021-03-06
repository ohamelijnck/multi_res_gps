import numpy as np
import tensorflow as tf

from . import ELL
from ... import util
class GPRN_ELL(ELL):
    def __init__(self, context, r=0):
        self.context = context
        self.r = r

    def setup(self, elbo):
        self.elbo = elbo
        self.parameters = self.context.parameters

        self.x_train = self.elbo.data.get_placeholder(source=self.r, var='x')
        self.y_train = self.elbo.data.get_placeholder(source=self.r, var='y')
        self.y_train_nans = self.elbo.data.get_placeholder(source=self.r, var='y_nan')

        self.noise_sigma = self.parameters.get(name='noise_sigma_{r}'.format(r=self.r))

    def _build_ell(self):
        print('BUILDING GPRN ELL')
        total_sum = 0.0

        c1 = 0
        P = self.elbo.num_outputs

        get_sigma = lambda sig : tf.square(util.var_postive(sig))

        for k in range(self.elbo.q_num_components):
            pi_k = self.elbo.q_weights[k]
            for p in range(self.elbo.num_outputs):
                n_p = tf.count_nonzero(self.y_train_nans[:,p])
                sig = get_sigma(self.noise_sigma[p])
                c1 -= pi_k * (tf.to_float(n_p)/2)*util.safe_log(2*np.pi*sig)

        for k in range(self.elbo.q_num_components):
            pi_k = self.elbo.q_weights[k]
            _mu_f, _sigma_f, _mu_w, _sigma_w = self.elbo.sparsity._build_intermediate_conditionals(k, self.r, self.x_train)
            #active_tasks = self.elbo.data.get_raw(source=self.r, var='active_tasks')
            #for p in active_tasks:
            for p in range(self.elbo.num_outputs):

                sig = get_sigma(self.noise_sigma[p])
                c2 = -(1/(2*sig))
                #c2 = tf.clip_by_value(c2, -1e20, 1e20)

                nan_mask = tf.cast(self.y_train_nans[:,p], dtype=tf.bool)
                y_p = tf.boolean_mask(mask=nan_mask, tensor=self.y_train[:,p])

                #sample the X where we have data
                sample_diag = lambda t: tf.boolean_mask(mask=nan_mask, tensor=t, axis=1)
                #sample_diag = lambda t: t

                mu_f = sample_diag(_mu_f[:, :, 0])
                sigma_f = sample_diag(tf.matrix_diag_part(_sigma_f))
                mu_wi = sample_diag(_mu_w[p,:,:,0])
                sigma_wi = sample_diag(tf.matrix_diag_part(_sigma_w[p,:,:,:]))

                #f = tf.diag_part(tf.matmul(mu_wi, mu_f, transpose_a=True))
                f = tf.multiply(mu_wi, mu_f)


                err = y_p - tf.squeeze(f)
                err = tf.reduce_sum(tf.square(err))

                        
                ell = err + tf.trace(tf.matmul(mu_f, tf.multiply(sigma_wi, mu_f), transpose_a=True))
                ell = ell + tf.trace(tf.matmul(mu_wi, tf.multiply(sigma_f, mu_wi), transpose_a=True))
                ell = ell + tf.trace(tf.matmul(sigma_f, sigma_wi, transpose_b=True))

                total_sum = tf.Print(total_sum, [p, err, sig, c2, c2*pi_k*ell], 'c2*pi_k*ell: ', summarize=110)
                total_sum += c2*pi_k*ell

        return c1+total_sum
