import numpy as np
import tensorflow as tf

from . import ELL
from ... import util
class GP_ELL(ELL):
    #r = resolution index
    def __init__(self, context, r=0, a=0):
        self.context = context
        self.parameters = self.context.parameters
        self.r = r
        self.a = a

    def setup(self, elbo):
        self.elbo = elbo

        self.x_train = self.elbo.data.get_placeholder(source=self.r, var='x')
        self.y_train = self.elbo.data.get_placeholder(source=self.r, var='y')
        self.y_train_nans = self.elbo.data.get_placeholder(source=self.r, var='y_nan')

        self.noise_sigma = self.parameters.get(name='noise_sigma_{r}'.format(r=self.r))

        self.num_outputs = self.elbo.num_outputs
        self.q_num_components = self.elbo.q_num_components
        self.q_weights = self.elbo.q_weights
        self.sparsity = self.elbo.sparsity


    def _build_ell(self):
        print('BUILDING SINGLE GP ELL')
        total_sum = 0.0

        c1 = 0
        P = self.num_outputs

        p = 0
        noise_sigma = tf.cast(tf.square(util.var_postive(tf.cast(self.noise_sigma[p], tf.float64))), tf.float32)
        #noise_sigma = 1e-6
        noise_sigma = tf.Print(noise_sigma, [noise_sigma], 'noise_sigma: ')

        for k in range(self.q_num_components):
            pi_k = self.q_weights[k]
            n_p = tf.to_float(tf.count_nonzero(self.y_train_nans[:,p]))
            c1 = pi_k * 0.5*(n_p*util.safe_log(2*np.pi)+n_p*util.safe_log(noise_sigma))
            #c1 = pi_k * 0.5*(n_p*util.safe_log(2*np.pi*noise_sigma))

        c1 = -c1
        c1 = tf.Print(c1, [c1], 'c1')
        c1 = tf.Print(c1, [n_p], 'n_p')

        for k in range(self.q_num_components):
            pi_k = self.q_weights[k]
            _mu_f, _sigma_f, _, _ = self.sparsity._build_intermediate_conditionals(k, self.a, self.x_train)

            c2 = -tf.divide(1.0, 2.0*noise_sigma)
            #c2 = tf.clip_by_value(c2, -1e20, 1e20)

            nan_mask = tf.cast(self.y_train_nans[:,p], dtype=tf.bool)
            y_p = tf.boolean_mask(mask=nan_mask, tensor=self.y_train[:,p])
            #y_p = self.y_train[:,p]

            #sample the X where we have data
            sample_diag = lambda t: tf.boolean_mask(mask=nan_mask, tensor=t, axis=1)
            #sample_diag = lambda t: t

            mu_f = sample_diag(_mu_f[:, :, 0])
            #mu_f = _mu_f[:, :, 0]


            sigma_f = sample_diag(tf.matrix_diag_part(_sigma_f))
            #sigma_f = tf.matrix_diag_part(_sigma_f)

            f = mu_f

            err = y_p - tf.squeeze(f)
            err = tf.reduce_sum(tf.square(err))

            total_sum = tf.Print(total_sum, [y_p], 'y_p: ')
            total_sum = tf.Print(total_sum, [tf.squeeze(f)], 'tf.squeeze(f): ')

            ell = err + tf.reduce_sum(sigma_f)
            #ell = err + tf.trace(_sigma_f[0, :])

            total_sum = tf.Print(total_sum, [err], 'err: ')
            total_sum = tf.Print(total_sum, [ell], 'ell: ')
            total_sum = tf.Print(total_sum, [noise_sigma], 'noise_sigma: ')
            total_sum = tf.Print(total_sum, [ell], 'total: ')
            total_sum = tf.Print(total_sum, [c2], 'c2: ')
            total_sum = tf.Print(total_sum, [c1], 'c1: ')
            total_sum = tf.Print(total_sum, [pi_k], 'pi_k: ')

            #total_sum += tf.cast(tf.cast(c2, tf.float64)*tf.cast(pi_k*ell, tf.float64), tf.float32)
            
            total_sum += c2*pi_k*ell

        return c1+total_sum

