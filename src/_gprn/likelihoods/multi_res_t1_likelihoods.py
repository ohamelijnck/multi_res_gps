from . import StandardGPRNLikelihood
import tensorflow as tf
import numpy as np
import math
from .. import util
from .. import debugger
from .. sparsity import StandardSparsity
from .. precomputers import Precomputed

class MultiResT1Likelihood(StandardGPRNLikelihood):
    def __init__(self, context):
        self.context = context

    def setup_multi_res(self):
        self.setup_standard()
        self.precomputed = Precomputed(self.data, self.context)

        self.x_train_m = self.data.get_placeholder(source=1, var='x')
        self.y_train_m = self.data.get_placeholder(source=1, var='y')
        self.y_train_m_nans = self.data.get_placeholder(source=1, var='y_nan')

        self.get_multi_res_variables()
        self.sparsity = StandardSparsity(self.data, self.context, self.precomputed)

    def setup(self, data):
        self.data = data
        self.setup_multi_res()

    def get_multi_res_variables(self):
        with tf.variable_scope("parameters", reuse=True):
            self.sigma_gg = tf.get_variable(name='noise_sigma_1')

    def build_graph(self):
        lik = self._build_log_likelihood()
        lik_1 = self._build_log_likelihood_1()
        return lik+lik_1

    def _build_log_likelihood_1(self):
        print('mr_ell_1____#')
        total_sum = 0.0

        M = self.data.get_raw(1, 'M')

        t1 = 0.0
        t2 = 0.0
        t3 = 0.0
        t4 = 0.0
        c1 = 0
        P = self.num_outputs


        for k in range(self.q_num_components):
            pi_k = self.q_weights[k]
            for p in range(1):
                n_p = tf.count_nonzero(self.y_train_m_nans[:,p])
                c1 -= pi_k * (tf.to_float(n_p)/2)*util.safe_log(2*np.pi*util.var_postive(self.sigma_gg[p]))
        c2 = -(1/(2*util.var_postive(self.sigma_gg[p])))


        for k in range(self.q_num_components):
            pi_k = self.q_weights[k]
            for p in range(self.num_outputs):
                nan_mask = tf.cast(self.y_train_nans[:,p], dtype=tf.bool)
                y_p = tf.boolean_mask(mask=nan_mask, tensor=self.y_train[:,p])

                sample_diag = lambda t: tf.boolean_mask(mask=nan_mask, tensor=t, axis=1)

                shp = tf.shape(self.x_train_m)
                shp = tf.Print(shp, [shp], 'shp: ')
                x_stacked = tf.reshape(tf.transpose(self.x_train_m, perm=[1, 0, 2]), [shp[0]*shp[1], shp[2]])

                _mu_f, _sigma_f, _mu_w, _sigma_w = self.sparsity._build_intermediate_conditionals(k, x_stacked)

                mu_f = _mu_f[:, :, 0]
                mu_wi = _mu_w[p,:,:,0]
                sigma_f = tf.matrix_diag_part(_sigma_f)
                sigma_wi = tf.matrix_diag_part(_sigma_w[p,:,:,:])
                shp_f = tf.shape(mu_f)
                shp_w = tf.shape(mu_wi)

                #nan_mask = tf.cast(self.y_train_m_nans[:,p], dtype=tf.bool)
                #sample_diag = lambda t: tf.boolean_mask(mask=nan_mask, tensor=t, axis=1)

                y_m = self.y_train_m[:,p]

                shp = tf.shape(self.x_train_m)
                f = tf.reshape(tf.diag_part(tf.matmul(mu_wi, mu_f, transpose_a=True)),  [shp[1], shp[0]])
                f =  (1/M)*tf.reduce_sum(f, axis=0)

                total_sum += tf.reduce_sum(tf.square((y_m-f)))
        return c1+c2*total_sum
