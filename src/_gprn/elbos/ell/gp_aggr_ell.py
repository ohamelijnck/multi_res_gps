import numpy as np
import tensorflow as tf

from . import GP_ELL
from ... import util
class GP_Aggr_ELL(GP_ELL):
    def __init__(self, context, r=0, a=0):
        self.context = context
        self.parameters = self.context.parameters
        self.r = r #index into the resolution/likelihood

    def setup(self, elbo):
        self.elbo = elbo

        #training data, y_train may have nans
        self.x_train = self.elbo.data.get_placeholder(source=self.r, var='x')
        self.y_train = self.elbo.data.get_placeholder(source=self.r, var='y')

        #a mask of the the nans in y_train
        self.y_train_nans = self.elbo.data.get_placeholder(source=self.r, var='y_nan')

        self.noise_sigma = self.parameters.get(name='noise_sigma_{r}'.format(r=self.r))

        print('self.ell_arr:', self.noise_sigma,' r: ', self.r)



    def _build_ell(self):
        """
            Implements the closed form expected log likelihood for MR-GP/VBagg-normal      
                $\sum^N_n \mathbb{E} \[log \mathcal{N} (y_n | \frac{1}{|S_n|} \sum^{S_n}_{x_s} f(x_s), \sigma_y^2) \]$
            This is derived as:
                $\sum^N_n (y_n - \frac{1}{|S_n|}\sum^{S_n}_{x_s} \mu_f(x_s))^2 + \sum^{S_n}_{a} \sum^{S_n}_{b} \Sigma_f(a,b)$
        """
        print('BUILDING GP AGGR ELL')
        total_sum = 0.0 

        M = self.elbo.data.get_raw(self.r, 'M') #Number of aggregation points ($M=|S_n|$)

        #init terms of the ELL
        c1 = 0
        c2 = 0

        #We are single task, and have a single latent function
        P = 1
        Q = 1


        #TODO: move this to a util function
        get_sigma = lambda sig : tf.square(util.var_postive(sig))

        sig = get_sigma(self.noise_sigma[0])

        for k in range(self.elbo.q_num_components):
            pi_k = self.elbo.q_weights[k]
            p=0 #single task
            n_p = tf.count_nonzero(self.y_train_nans[:,p]) #the number of datapoints without the nans
            c1 -= pi_k * (tf.to_float(n_p)/2)*util.safe_log(2*np.pi*sig)

        for k in range(self.elbo.q_num_components):
            pi_k = self.elbo.q_weights[k]
            p = 0 #single task

            c2 = -(1/(2*sig))

            nan_mask = tf.cast(self.y_train_nans[:,p], dtype=tf.bool)
            y_p = tf.boolean_mask(mask=nan_mask, tensor=self.y_train[:,p])

            y_m = tf.expand_dims(y_p, -1) #N_a x 1

            #calculate q(f) = \int p(f | u) q(u) \mathop{du} = N(mu_f, sigma_f)
                #predict = True returns the vectorised from of the block diagionals of sigma_f
                #predict = False returns the full NMxNM matrix sigma_f
            #_mu_f = Q x N x S x 1
            #_sigma_f =  Q x N x S x S
            _mu_f, _sigma_f, _, _ = self.elbo.sparsity._build_intermediate_conditionals(k, 0, self.x_train, predict=True)
            mu_f = _mu_f[0, :, :, 0] # N x S
            sigma_f = _sigma_f[0, :, :, :] # N x S x S


            #mask missing y's
            sigma_f = tf.boolean_mask(mask=nan_mask, tensor=sigma_f, axis=0) # N_a x S x S
            mu_f = tf.boolean_mask(mask=nan_mask, tensor=mu_f, axis=0) # N_a x S x 1

            #contruct ELBO
            err = y_m - tf.reduce_mean(mu_f, axis=1)[:, None]
            err = tf.square(err)
            err = tf.reduce_sum(err)

            trace_term = (1/(M*M))*tf.reduce_sum(sigma_f)

            #Approximate $\sum^{S_n}_{a} \sum^{S_n}_{b} \Sigma_f(a,b)$ with just the trace. Left for testing.
            #trace_term = tf.reduce_sum(tf.trace(sigma_f))

            total_sum += c2*(err + trace_term)
        return c1+total_sum
