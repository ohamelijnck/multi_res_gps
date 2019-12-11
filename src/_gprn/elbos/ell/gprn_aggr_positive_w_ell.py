import numpy as np
import tensorflow as tf

from . import ELL
from ... import util
class GPRN_Aggr_Positive_W_ELL(ELL):
    def __init__(self, context, r=0):
        self.context = context
        self.parameters = self.context.parameters
        self.r = r

    def setup(self, elbo):
        self.elbo = elbo

        self.x_train = self.elbo.data.get_placeholder(source=self.r, var='x')
        self.y_train = self.elbo.data.get_placeholder(source=self.r, var='y')
        self.y_train_nans = self.elbo.data.get_placeholder(source=self.r, var='y_nan')

        self.noise_sigma = self.parameters.get(name='noise_sigma_{r}'.format(r=self.r))

    def _build_ell(self):
        print('BUILDING GPRN AGGR LIK')
        total_sum = 0.0

        M = self.elbo.data.get_raw(self.r, 'M')

        t1 = 0.0
        t2 = 0.0
        t3 = 0.0
        t4 = 0.0
        c1 = 0
        P = self.elbo.num_outputs

        active_tasks = self.elbo.data.get_raw(source=self.r, var='active_tasks')

        get_sigma = lambda sig : tf.square(util.var_postive(sig))

        for k in range(self.elbo.q_num_components):
            pi_k = self.elbo.q_weights[k]
            
            for p in active_tasks[self.r]:
                noise_sigma = get_sigma(self.noise_sigma[p])
                n_p = tf.count_nonzero(self.y_train_nans[:,p])
                c1 -= pi_k * (tf.to_float(n_p)/2)*util.safe_log(2*np.pi*noise_sigma)

                c2 = -(1/(2*noise_sigma))

                print(self.y_train_nans)
                nan_mask = tf.cast(self.y_train_nans[:,p], dtype=tf.bool)
                y_p = tf.boolean_mask(mask=nan_mask, tensor=self.y_train[:,p])

                y_m = tf.expand_dims(y_p, -1) #s(N)x1

                t1 = 0.0
                t2 = 0.0
                t3 = 0.0
                t4 = 0.0

                t1 = tf.matmul(y_m, y_m, transpose_a=True)

                shp = tf.shape(self.x_train) #NxMxD
                x_stacked = tf.reshape(tf.transpose(self.x_train, perm=[1, 0, 2]), [shp[0]*shp[1], shp[2]])

                print('x_stacked: ', x_stacked)

                predict_flag = True
                _mu_f, _sigma_f, _mu_w, _sigma_w = self.elbo.sparsity._build_intermediate_conditionals(k, 0, x_stacked, predict=predict_flag)

                #_mu_f = [Q 100 1]
                #_sigma_f = [Q N]
                #_mu_w = [P Q N 1]
                #_sigma_w = [P Q N]

                if predict_flag:
                    sigma_f = _sigma_f # QxN
                    sigma_wi = _sigma_w[p, :, :] #QxN
                else:
                    sigma_f = tf.matrix_diag_part(_sigma_f) #QxN
                    sigma_wi = tf.matrix_diag_part(_sigma_w[p,:,:, :]) #QxN

                mu_f = _mu_f[:, :, 0] #QxN
                mu_wi = _mu_w[p,:,:,0] #QxN

                Q = tf.shape(mu_f)[0]

                def mask(t, shp):
                    t = tf.reshape(t, [Q, shp[0], shp[1]]) #QxNxM
                    t = tf.boolean_mask(mask=nan_mask, tensor=t, axis=1) #Qxs(N)xM
                    t = tf.reshape(t, [Q, tf.shape(t)[1]*tf.shape(t)[2]]) #Qxs(N)M
                    return t

                sigma_f = mask(sigma_f, shp) #QxN_s
                sig_wi = mask(sigma_wi, shp) #QxN_S
                mu_f = mask(mu_f, shp) #QxN_s
                mu_wi = mask(mu_wi, shp) #QxN_s


                N_S = tf.shape(y_m)[0]

                #a1 = tf.diag_part(tf.matmul(, mu_f, transpose_a=True)) #N
                a1 = tf.multiply(util.safe_exp((mu_wi+sig_wi)), mu_f) #QxN_s
                a1 = tf.reduce_sum(a1, axis=0) #N_s

                a1 =  tf.reshape(a1, [1, shp[1], N_S]) #1xMxN_r
                a_is_b = tf.matmul(a1, tf.transpose(a1, perm=[0, 2, 1]))#1xMxM
                a_is_b = tf.trace(a_is_b[0, :, :])
                a_is_b += tf.reduce_sum(tf.multiply(sigma_f, util.safe_exp(2*(mu_wi+sig_wi))))
                #a_is_b += tf.trace(tf.matmul(mu_f, tf.multiply(util.safe_exp(2*(mu_wi+sig_wi)), mu_f), transpose_a=True))

                a2 = tf.diag_part(tf.matmul(util.safe_exp(mu_wi+0.5*sig_wi), mu_f, transpose_a=True))
                a2 =  tf.reshape(a2, [1, shp[1], N_S])

                a_not_b = tf.matmul(a2, tf.transpose(a2, perm=[0, 2, 1]))
                a_not_b = tf.reduce_sum(a_not_b)-tf.trace(a_not_b[0, :, :])


                #b = tf.trace(tf.matmul(sigma_f, util.safe_exp(2*(mu_wi+sig_wi)), transpose_a=True))

                t4 += (1/(M*M))*(a_is_b + a_not_b)


                t3_tmp = tf.reshape(tf.diag_part(tf.matmul(util.safe_exp(mu_wi+0.5*sig_wi), mu_f, transpose_a=True)),  [shp[1], N_S])
                t3_tmp =  tf.expand_dims(tf.reduce_sum(t3_tmp, axis=0), -1)
                t3 += (-2/M)*tf.matmul(y_m, t3_tmp, transpose_a=True)

                t3 = tf.Print(t3, [self.r, tf.reduce_sum(y_m-t3_tmp)], 'y_m-t3')


                #t3_tmp = tf.expand_dims(tf.diag_part(tf.matmul(mu_wi_a, mu_f_a, transpose_a=True)), -1)

                total_sum += c1 + c2*(t1+t2+t3+t4)
        return total_sum




