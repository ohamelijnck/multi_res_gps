import numpy as np
import tensorflow as tf

from . import ELL
from ... import util
class GPRN_Aggr_ELL(ELL):
    def __init__(self, context, r=0):
        self.context = context
        self.parameters = self.context.parameters
        self.r = r #index into the resolution/likelihood

    def setup(self, elbo):
        self.elbo = elbo

        self.x_train = self.elbo.data.get_placeholder(source=self.r, var='x')
        self.y_train = self.elbo.data.get_placeholder(source=self.r, var='y')
        #mask of the the nans in y_train
        self.y_train_nans = self.elbo.data.get_placeholder(source=self.r, var='y_nan')
        self.n_data = [self.elbo.data.get_num_training(source=i) for i in range(self.context.num_likelihood_components)]


        #each task in resolution r has its own noise
        self.noise_sigma = self.parameters.get(name='noise_sigma_{r}'.format(r=self.r)) 

    def _build_ell(self):
        """
            Implements the closed form expected log likelihood of MR-GPRN     
                $\sum^{P}_p \sum^N_n \mathbb{E} \[log \mathcal{N} (y_{p,n} | \frac{1}{|S_{|n|} \sum^{S_n}_{x_s} \sum^Q_q W_pq(x_s) f_q(x_s), \sigma_y^2) \]$
        """
        print('BUILDING GPRN AGGR LIK')


        M = self.elbo.data.get_raw(self.r, 'M') #Number of aggregation points ($M=|S_n|$)

        #init terms of the ELL
        total_sum = 0.0
        c1 = 0
        P = self.elbo.num_outputs

        get_sigma = lambda sig : tf.square(util.var_postive(sig))

        #Not all resolutions have all tasks. Get the tasks for resolution r.
        active_tasks = self.elbo.data.get_raw(source=self.r, var='active_tasks') 
                
        
        for k in range(self.elbo.q_num_components):
            pi_k = self.elbo.q_weights[k]
            for p in active_tasks[self.r]:
                n_p = tf.count_nonzero(self.y_train_nans[:,p])

                #constants from the Gaussian Likelihood
                sig = get_sigma(self.noise_sigma[p])
                c1 -= pi_k * (tf.to_float(n_p)/2)*util.safe_log(2*np.pi*sig)
                c2 = -(1/(2*sig))

                #init terms from the ELL
                t1 = 0.0
                t2 = 0.0
                t3 = 0.0
                t4 = 0.0

                nan_mask = tf.cast(self.y_train_nans[:,p], dtype=tf.bool)
                y_p = tf.boolean_mask(mask=nan_mask, tensor=self.y_train[:,p])
                y_m = tf.expand_dims(y_p, -1)

                shp = tf.shape(self.x_train)
                _mu_f, _sigma_f, _mu_w, _sigma_w = self.elbo.sparsity._build_intermediate_conditionals(k, 0, self.x_train, predict=True)

                #_mu_f = Q x N x M x 1
                #_sigma_f = Q x N x M x M
                #_mu_w = P x Q x N x M x 1
                #_sigma_w = P x Q x N x M x M

                #t1: y square term
                t1 = tf.matmul(y_m, y_m, transpose_a=True)

                #t2: y*Wf term
                mu_f = _mu_f[:, :, :, 0] #QxNxM
                mu_wi = _mu_w[p,:,:, :,0] #QxNxM
                sigma_f = _sigma_f[:,:,:,:] #QxNxMxM
                sigma_w = _sigma_w[p,:,:,:,:] #QxNxMxM

                f_q = tf.reduce_mean(tf.multiply(mu_wi, mu_f), axis=2) #average across discrisation region
                f = tf.reduce_sum(f_q, axis=0) #sum up latent functions
                f_m = tf.boolean_mask(mask=nan_mask, tensor=f) #remove missing values
                t2 = -2*tf.reduce_sum(tf.multiply(y_m, f_m))

                f_m = tf.expand_dims(f_m, -1)

                err = tf.reduce_sum(tf.square(y_m-f_m))

                #t3: Wf square term
                Q = tf.shape(mu_f)[0]
                N = tf.shape(mu_f)[1]
                M = tf.cast(tf.shape(mu_f)[2], tf.float32)

                mu_wi = tf.expand_dims(mu_wi, -1)
                mu_f = tf.expand_dims(mu_f, -1)
                #Q x N x M x M
                a = tf.multiply(tf.matmul(mu_wi, tf.transpose(mu_wi, [0, 1, 3, 2])), sigma_f)
                #Q x N x M x M
                b = tf.multiply(tf.matmul(mu_f, tf.transpose(mu_f, [0, 1, 3, 2])), sigma_w)
                #Q x N x M x M
                c = tf.multiply(sigma_f, sigma_w)

                t3 = (1/(M*M))*tf.reduce_sum(a+b+c)


                #get_y_train_nans returns the indices of the nans, it is 0 when there is a nan 
                full_size = float(sum(self.elbo.data.get_y_train_nans(self.r)[:, p]))
                batch_size = tf.cast(tf.shape(y_m)[0], tf.float32)
                scale = full_size/batch_size

                scale =tf.Print(scale, [p, batch_size, full_size, scale], 'scale: ')
                total_sum += scale*(c1 + c2*(err+t3))
        return total_sum



