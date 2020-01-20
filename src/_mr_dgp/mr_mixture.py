import tensorflow as tf
import numpy as np

from gpflow.params import DataHolder, Minibatch
from gpflow import autoflow, params_as_tensors, ParamList
from gpflow.models.model import Model
from gpflow import settings

from .mr_svgp import MR_SVGP

class MR_Mixture(Model):
    def __init__(self, datasets=[], inducing_locations=[], kernels=[], noise_sigmas=[], minibatch_sizes = [], mixing_weight=None, parent_mixtures=None, masks=None, num_samples=1, **kwargs):
        """
            datasets: an array of arrays [X_a, Y_a] ordered by 'trust', ie datasets[0] is the most reliable 
            inducing_points_locations: an array of inducing locations for each of the datasets  
            kernels: an array of kernels for each of the datasets  
            noise_sigmas: an array of noise_sigmas for each of the datasets  
            mixing_weight (MR_Mixing_Weight): an object that will combine the predictions from each of the local experts
            parent_mixtures: an array of parent mixture models
        """
        Model.__init__(self, **kwargs)

        self.dataset_sizes = []
        for d in datasets:
            self.dataset_sizes.append(d[0].shape[0])

        self.num_datasets = len(datasets)
        self.X = []
        self.Y = []
        self.Z = inducing_locations
        self.masks = masks
        self.MASKS = []
        self.kernels = kernels
        self.noise_sigmas = noise_sigmas
        self.num_samples = num_samples

        #gpflow models are Parameterized objects
        print(parent_mixtures)
        self.parent_mixtures = ParamList(parent_mixtures)  if parent_mixtures is not None else None

        self.mixing_weight = mixing_weight

        minibatch=False
        for i, d in enumerate(datasets):
            #TODO: can we just wrap with a ParamList?
            if minibatch:
                _x = Minibatch(d[0], batch_size=minibatch_sizes[i], seed=0) 
                _y = Minibatch(d[1], batch_size=minibatch_sizes[i], seed=0) 
            else:
                _x = DataHolder(d[0]) 
                _y = DataHolder(d[1]) 


            #Check we have some masks
            if self.masks:
                #Check if we have a mask for this dataset
                _mask = None 
                if self.masks[i] is not None:
                    if minibatch:
                        _mask = Minibatch(self.masks[i], batch_size=minibatch_sizes[0], seed=0) 
                    else:
                        _mask = DataHolder(self.masks[i])

            #make it so GPFlow can find _x, _y
            setattr(self, 'x_{i}'.format(i=i), _x)
            setattr(self, 'y_{i}'.format(i=i), _y)
            if self.masks:
                setattr(self, 'mask_{i}'.format(i=i), _mask)

            #save references
            self.X.append(self.__dict__['x_{i}'.format(i=i)])
            self.Y.append(self.__dict__['y_{i}'.format(i=i)])
            if self.masks:
                self.MASKS.append(self.__dict__['mask_{i}'.format(i=i)])

        self.setup()

    def setup(self):
        self.setup_base_gps()

    def setup_base_gps(self):
        self.base_gps = [] # one per dataset
        self.deep_gps = [] # all but the first dataset
        self.parent_gps = [] 
        for i in range(self.num_datasets):
            z = self.Z[0][i]
            k = self.kernels[0][i]
            sig = self.noise_sigmas[0][i]

            gp = MR_SVGP(z, k, sig)
            self.base_gps.append(gp)

            if i > 0:
                z = self.Z[1][i-1]
                k = self.kernels[1][i-1]
                sig = self.noise_sigmas[1][i-1]

                dgp = MR_SVGP(z, k, sig)
                self.deep_gps.append(dgp)

        if self.parent_mixtures:
            for i in range(len(self.parent_mixtures)):
                z = self.Z[2][i]
                k = self.kernels[2][i]
                sig = self.noise_sigmas[2][i]

                gp = MR_SVGP(z, k, sig)

                self.parent_gps.append(gp)

        self.base_gps = ParamList(self.base_gps)
        self.deep_gps = ParamList(self.deep_gps)
        self.parent_gps = ParamList(self.parent_gps)

    @params_as_tensors
    def sample_condition(self, x, samples, gp, predict_y):
        num_samples = tf.shape(samples)[0]

        _x = tf.tile(tf.expand_dims(x, 0), [num_samples, 1, 1, 1]) #num_samples x N x S x D

        #append input space to output of previous GP
        samples = tf.Print(samples, [tf.shape(samples), tf.shape(_x)], 's, _x: ', summarize=100)
        _x = tf.concat([samples, _x], axis=3) #num_samples x N x S x (D+1)

        shp = tf.shape(_x)
        _x = tf.reshape(_x, [shp[0]*shp[1], shp[2], shp[3]]) #(num_samples*N) x S x (D+1)

        #mu in (num_samples*N) x S x 1
        #sig in (num_samples*N) x S x S
        mu, sig = gp.conditional(_x)
        mu = tf.reshape(mu, [shp[0], shp[1], shp[2], 1]) #num_samples x N x S x 1
        sig = tf.reshape(sig, [shp[0], shp[1], shp[2], shp[2]]) #num_samples x N x S x S

        if predict_y:
            mu, sig = gp.likelihood.predict_mean_and_var(mu, sig)

        return mu, sig

    @params_as_tensors
    def propogate(self, num_samples=1, X=None, predict_y=False):
        #each mixture has a set of base GPs trained on each of the datasets
        base_mu = []
        base_sig = []

        #the 2nd layer are the base samples propogates through and the samples from the parent mixtures as well
        dgp_mu = []
        dgp_sig = []

        parent_mu = []
        parent_sig = []

        x_0 = getattr(self, 'x_{i}'.format(i=0))
        if X is not None:
            x_0 = X

        for i in range(self.num_datasets):
            x_i = getattr(self, 'x_{i}'.format(i=i))

            if X is not None:
                x_i = X

            #mu: N x S x 1
            #sig: N x S x S
            mu, sig = self.base_gps[i].conditional(x_i)

            if predict_y:
                mu, sig = self.base_gps[i].likelihood.predict_mean_and_var(mu, sig)

            base_mu.append(mu)
            base_sig.append(sig)

            if i > 0:
                #all dgps are trained on x_0, y_0
                mu, sig = self.base_gps[i].conditional(x_0)

                samples = self.base_gps[i].sample(mu, sig, num_samples) #num_samples x N x S x 1

                mu, sig = self.sample_condition(x_0, samples, self.deep_gps[i-1], predict_y)

                dgp_mu.append(mu)
                dgp_sig.append(sig)

        for i in range(len(self.parent_gps)):
            x_0 = tf.Print(x_0, [num_samples, tf.shape(x_0)], 'x_0: ')
            samples = self.parent_mixtures[i].sample_experts(x_0, num_samples) #num_samples x N x S x 1

            samples = tf.Print(samples, [tf.shape(samples)], 'samples----: ')

            mu, sig = self.sample_condition(x_0, samples, self.parent_gps[i], predict_y)

            parent_mu.append(mu)
            parent_sig.append(sig)

        return base_mu, base_sig, dgp_mu, dgp_sig, parent_mu, parent_sig


    @params_as_tensors
    def sampled_ell(self, y, mu, sig, gp, mask):
        if mask is not None:
            _y = y
            mask.set_shape([None])
            y = tf.boolean_mask(mask=mask, tensor=y, axis=0)
            mu = tf.boolean_mask(mask=mask, tensor=mu, axis=1)
            sig = tf.boolean_mask(mask=mask, tensor=sig, axis=1)

            y = tf.Print(y, [tf.shape(mask),tf.shape(y), tf.shape(_y), mask, y], 'y masked:', summarize=100)

        shp = tf.shape(mu)

        mu = tf.reshape(mu, [shp[0]*shp[1], shp[2], 1]) #num_samples*N x S x 1
        sig = tf.reshape(sig, [shp[0]*shp[1], shp[2], shp[2]]) #num_samples*N x S x S

        y_i = tf.tile(tf.expand_dims(y, 0), [self.num_samples, 1, 1]) #num_samples x N x  D

        shp = tf.shape(y_i)
        y_i = tf.reshape(y_i, [shp[0]*shp[1], shp[2]]) #(num_samples*N) x  1

        sig = tf.Print(sig, [self.num_samples, tf.shape(y_i)], 'y_i_shp: ', summarize=100)
        sig = tf.Print(sig, [tf.shape(mu)], 'mu_sig: ', summarize=100)
        sig = tf.Print(sig, [tf.shape(sig)], 'sig_sig: ', summarize=100)


        
        ell =  gp.expected_log_likelihood(y_i, mu, sig)
        ell = tf.Print(ell, [tf.shape(ell)], 'ell: ')

        ell = tf.reshape(ell, [shp[0], shp[1]]) #num_samples x N
        #TODO: double check this is the right way round
        #ell = tf.reduce_mean(tf.reduce_sum(ell, axis=1), axis=0)
        ell = tf.reduce_sum(tf.reduce_mean(ell, axis=0), axis=0)
        return ell


    @params_as_tensors
    def _build_likelihood(self):
        base_kl_arr = []
        base_ell_arr = []
        dgp_kl_arr = []
        dgp_ell_arr = []
        if self.parent_mixtures:
            parent_kl_arr = []
            parent_ell_arr = []
        else:
            parent_ell_arr = tf.cast(0.0, settings.float_type)
            parent_kl_arr = tf.cast(0.0, settings.float_type)

        base_mu, base_sig, dgp_mu, dgp_sig, parent_mu, parent_sig = self.propogate(num_samples=self.num_samples)
        print(parent_mu)

        y_0 = getattr(self, 'y_{i}'.format(i=0))
        mask_0 = None
        if self.masks: 
            mask_0 = getattr(self, 'mask_{i}'.format(i=0))

        for i in range(self.num_datasets):
            x_i = getattr(self, 'x_{i}'.format(i=i))
            y_i = getattr(self, 'y_{i}'.format(i=i))
            mask_i = None
            if self.masks: 
                mask_i = getattr(self, 'mask_{i}'.format(i=i))


            mu, sig = base_mu[i], base_sig[i]

            scale = self.dataset_sizes[i]/tf.shape(y_i)[0]

            ell =  scale*self.base_gps[i].expected_log_likelihood(y_i, mu, sig)
            ell= tf.Print(ell, [tf.shape(sig)], 'sig: ')
            ell= tf.Print(ell, [tf.shape(mu)], 'mu: ')
            ell= tf.Print(ell, [tf.shape(y_i)], 'y_i: ')

            kl =  self.base_gps[i].kl_term()

            ell = tf.Print(ell, [tf.reduce_sum(ell)], 'base ell {i}: '.format(i=i))

            base_ell_arr.append(tf.reduce_sum(ell))
            base_kl_arr.append(tf.reduce_sum(kl))

            if i > 0:
                kl =  self.deep_gps[i-1].kl_term() #-1 as there are |base_gps| -1 DGPs
                dgp_kl_arr.append(tf.reduce_sum(kl))

                mu, sig = dgp_mu[i-1], dgp_sig[i-1]
                ell = self.sampled_ell(y_0, mu, sig, self.deep_gps[i-1], mask_i)
                scale =self.dataset_sizes[0]/tf.shape(y_0)[0] #trained onto the first dataset
                ell = scale*ell

                ell = tf.Print(ell, [ell], 'dgp ell')

                dgp_ell_arr.append(ell)

        parent_elbo = 0.0
        if self.parent_mixtures:
            for i in range(len(self.parent_mixtures)):
                kl =  self.parent_gps[i].kl_term() #-1 as there are |base_gps| -1 DGPs
                parent_kl_arr.append(tf.reduce_sum(kl))

                ell_p = self.parent_mixtures[i]._build_likelihood()
                parent_elbo += ell_p

                mu, sig = parent_mu[i], parent_sig[i]

                ell = self.sampled_ell(y_0, mu, sig, self.parent_gps[i], mask_0)
                scale =self.dataset_sizes[0]/tf.shape(y_0)[0] #trained onto the first dataset
                ell = scale*ell

                parent_ell_arr.append(ell)


        if len(dgp_kl_arr) == 0:
            dgp_ell_arr = tf.cast(0.0, settings.float_type)
            dgp_kl_arr = tf.cast(0.0, settings.float_type)
        kl =  tf.reduce_sum(base_kl_arr) + tf.reduce_sum(dgp_kl_arr) + tf.reduce_sum(parent_kl_arr)
        ell =  tf.reduce_sum(base_ell_arr) + tf.reduce_sum(dgp_ell_arr) + tf.reduce_sum(parent_ell_arr)

        #self.base_elbo = tf.reduce_sum(base_ell_arr) - tf.reduce_sum(base_kl_arr)
        #self.dgp_elbo = tf.reduce_sum(dgp_ell_arr) - tf.reduce_sum(dgp_kl_arr)


        self.base_elbo = -(tf.reduce_sum(base_ell_arr) - tf.reduce_sum(base_kl_arr))
        self.dgp_elbo = -(tf.reduce_sum(dgp_ell_arr) - tf.reduce_sum(dgp_kl_arr))
        self.parent_elbo = -(tf.reduce_sum(parent_ell_arr) - tf.reduce_sum(parent_kl_arr))
        self.elbo =  -(parent_elbo + ell-kl)

        return -self.elbo

    def set_gp_hyperparam_trainable(self, gp, flag):
        #hyperparameters
        for param in gp.K.parameters:
            param.trainable=flag
        for param in gp.likelihood.parameters:
            param.trainable=flag

    def set_gp_noise_trainable(self, gp, flag):
        for param in gp.likelihood.parameters:
            param.trainable=flag

    def set_gp_trainable(self, gp, flag):
        #variational parameters
        gp.q_mu.trainable=flag
        gp.q_sqrt.trainable=flag
        gp.Z.trainable=flag

    def set_base_gp_noise(self, flag):
        for i in range(self.num_datasets):
            self.set_gp_noise_trainable(self.base_gps[i], flag)

    def set_dgp_gp_noise(self, flag):
        for i in range(self.num_datasets-1):
            self.set_gp_noise_trainable(self.deep_gps[i], flag)

    def enable_base_hyperparameters(self):
        for i in range(self.num_datasets):
            self.set_gp_hyperparam_trainable(self.base_gps[i], True)

    def disable_base_hyperparameters(self):
        for i in range(self.num_datasets):
            self.set_gp_hyperparam_trainable(self.base_gps[i], False)

    def enable_base_elbo(self):
        for i in range(self.num_datasets):
            self.set_gp_trainable(self.base_gps[i], True)
            self.set_gp_hyperparam_trainable(self.base_gps[i], True)

    def disable_base_elbo(self):
        for i in range(self.num_datasets):
            self.set_gp_trainable(self.base_gps[i], False)
            self.set_gp_hyperparam_trainable(self.base_gps[i], False)

    def enable_dgp_hyperparameters(self):
        for i in range(self.num_datasets-1):
            self.set_gp_hyperparam_trainable(self.deep_gps[i], True)

    def disable_dgp_hyperparameters(self):
        for i in range(self.num_datasets-1):
            self.set_gp_hyperparam_trainable(self.deep_gps[i], False)

    def enable_parent_hyperparameters(self):
        for i in range(len(self.parent_mixtures)):
            self.set_gp_hyperparam_trainable(self.parent_gps[i], True)

    def disable_parent_hyperparameters(self):
        for i in range(len(self.parent_mixtures)):
            self.set_gp_hyperparam_trainable(self.parent_gps[i], False)

    def disable_dgp_elbo(self):
        for i in range(self.num_datasets-1):
            self.set_gp_trainable(self.deep_gps[i], False)
            self.set_gp_hyperparam_trainable(self.deep_gps[i], False)

    def enable_dgp_elbo(self):
        for i in range(self.num_datasets-1):
            self.set_gp_trainable(self.deep_gps[i], True)
            self.set_gp_hyperparam_trainable(self.deep_gps[i], True)

    @autoflow()
    def get_inducing_points(self):
        z_base = []
        z_dgp = []
        for i in range(self.num_datasets):
            z_base.append(self.base_gps[i].get_z())
            if i >0:
                z_dgp.append(self.deep_gps[i-1].get_z())
        return z_base, z_dgp

    def train_dgp_elbo(self):
        self.objective = self.base_elbo

    def train_full_elbo(self):
        self.objective = self.elbo

    @autoflow((settings.float_type, [None, None]), (tf.int32, []))
    def predict_all_gps(self, XS, num_samples):
        XS = tf.expand_dims(XS, 1)
        base_mu, base_sig, dgp_mu, dgp_sig, parent_mu, parent_sig = self.propogate(X=XS, num_samples=num_samples)

        return tf.stack(base_mu), tf.stack(base_sig), tf.stack(dgp_mu), tf.stack(dgp_sig), tf.stack(parent_mu), tf.stack(parent_sig)

    @autoflow((settings.float_type, [None, None]), (tf.int32, []))
    def predict_y_all_gps(self, XS, num_samples):
        XS = tf.expand_dims(XS, 1)
        base_mu, base_sig, dgp_mu, dgp_sig, parent_mu, parent_sig = self.propogate(X=XS, num_samples=num_samples, predict_y=True)

        return tf.stack(base_mu), tf.stack(base_sig), tf.stack(dgp_mu), tf.stack(dgp_sig), tf.stack(parent_mu), tf.stack(parent_sig)


    @autoflow((settings.float_type, [None, None]), (tf.int32, []))
    def predict_experts(self, XS, num_samples):
        XS = tf.expand_dims(XS, 1)
        base_mu, base_sig, dgp_mu, dgp_sig, parent_mu, parent_sig = self.propogate(X=XS, num_samples=num_samples)

        dgp_mu = dgp_mu + parent_mu
        dgp_sig = dgp_sig + parent_sig

        mu, sig = self.mixing_weight.predict(base_mu, base_sig, dgp_mu, dgp_sig)

        return mu, sig

    @autoflow((settings.float_type, [None, None]), (tf.int32, []))
    def predict_y_experts(self, XS, num_samples):
        XS = tf.expand_dims(XS, 1)
        base_mu, base_sig, dgp_mu, dgp_sig, parent_mu, parent_sig = self.propogate(X=XS, num_samples=num_samples, predict_y=True)
        
        dgp_mu = dgp_mu + parent_mu
        dgp_sig = dgp_sig + parent_sig

        mu, sig = self.mixing_weight.predict(base_mu, base_sig, dgp_mu, dgp_sig)

        return mu, sig

    def sample_experts(self, XS, num_samples):
        base_mu, base_sig, dgp_mu, dgp_sig, parent_mu, parent_sig = self.propogate(X=XS, num_samples=num_samples)

        dgp_mu = dgp_mu + parent_mu
        dgp_sig = dgp_sig + parent_sig

        samples = self.mixing_weight.sample(base_mu, base_sig, dgp_mu, dgp_sig, num_samples)

        return samples

