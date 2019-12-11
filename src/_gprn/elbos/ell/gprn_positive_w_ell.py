import numpy as np
import tensorflow as tf

from . import ELL
from ... import util
class GPRN_Positive_W_ELL(ELL):
    def __init__(self, context, r=0):
        self.context = context
        self.r = r
        print(self.r)

    def setup(self, elbo):
        self.elbo = elbo
        self.parameters = self.context.parameters

        self.x_train = self.elbo.data.get_placeholder(source=self.r, var='x')
        self.y_train = self.elbo.data.get_placeholder(source=self.r, var='y')
        self.y_train_nans = self.elbo.data.get_placeholder(source=self.r, var='y_nan')

        self.noise_sigma = self.parameters.get(name='noise_sigma_{r}'.format(r=self.r))

    def _build_ell(self):
        print('ell')
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
            for p in range(self.elbo.num_outputs):
                sig = get_sigma(self.noise_sigma[p])
                c2 = -(1/(2*sig))

                nan_mask = tf.cast(self.y_train_nans[:,p], dtype=tf.bool)

                sample_diag = lambda t: tf.boolean_mask(mask=nan_mask, tensor=t, axis=1)

                mu_f = sample_diag(_mu_f[:,:,0])
                mu_wi = sample_diag(_mu_w[p,:,:,0])
                sig_wi = sample_diag(tf.matrix_diag_part(_sigma_w[p,:,:, :]))
                sig_f = sample_diag(tf.matrix_diag_part(_sigma_f[:,:, :]))

                y_p = self.y_train[:,p]
                y_p = tf.boolean_mask(mask=nan_mask, tensor=self.y_train[:,p])

                ell = tf.reduce_sum(tf.square(y_p))

                ell -= 2*tf.reduce_sum(tf.multiply(y_p,tf.diag_part(tf.matmul(util.safe_exp(mu_wi+0.5*sig_wi), mu_f, transpose_a = True))))
                ell = tf.clip_by_value(ell, -1e10, 1e10)


                ell += tf.trace(tf.multiply(
                    tf.matmul(mu_f,util.safe_exp(mu_wi+sig_wi ), transpose_a=True),
                    tf.matmul(util.safe_exp(mu_wi+sig_wi),mu_f, transpose_a=True)
                ))

                ell += tf.reduce_sum(tf.multiply(tf.multiply(util.safe_exp(mu_wi+sig_wi   ), sig_f), util.safe_exp(mu_wi+sig_wi)))
                ell = tf.clip_by_value(ell, -1e10, 1e10)

                total_sum += c2*pi_k*ell

        return c1+total_sum

