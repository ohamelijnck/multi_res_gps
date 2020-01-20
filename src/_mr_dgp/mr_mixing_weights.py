import tensorflow as tf
import numpy as np

from gpflow import autoflow, params_as_tensors, ParamList
from gpflow.models.model import Model
from gpflow import settings

from . import utils 

class MR_Mixing_Weights(object):
    def __init__(self):
        pass

    def predict(self):
        pass


class MR_Average_Mixture(MR_Mixing_Weights):
    def __init__(self):
        MR_Mixing_Weights.__init__(self)
    
    def predict(self, base_mu, base_sig, dgp_mu, dgp_sig):
        base_mu_samples = tf.tile(tf.expand_dims(base_mu[0], 0), [tf.shape(dgp_mu[0])[0], 1, 1, 1])
        base_sig_samples = tf.tile(tf.expand_dims(base_sig[0], 0), [tf.shape(dgp_mu[0])[0], 1, 1, 1])

        mu =  (1/(1+len(dgp_mu)))*tf.reduce_sum([base_mu_samples]+dgp_mu, axis=0)
        sig =  ((1/(1+len(dgp_sig)))**2)*tf.reduce_sum([base_sig_samples]+dgp_sig, axis=0)
        return mu, sig

    def sample(self, base_mu, base_sig, dgp_mu, dgp_sig, num_samples):
        base_mu_samples = tf.tile(tf.expand_dims(base_mu[0], 0), [tf.shape(dgp_mu[0])[0], 1, 1, 1])
        base_sig_samples = tf.tile(tf.expand_dims(base_sig[0], 0), [tf.shape(dgp_mu[0])[0], 1, 1, 1])

        z = tf.random_normal(tf.shape(base_mu_samples), dtype=settings.float_type)
        samples =  utils.reparameterize(base_mu_samples, base_sig_samples, z, full_cov = False) # num_samples x N x S x 1


        samples =  (1/(1+len(dgp_mu)))*tf.reduce_sum([samples]+dgp_mu, axis=0)

        return samples

class MR_Base_Only(MR_Mixing_Weights):
    def __init__(self, i=0):
        MR_Mixing_Weights.__init__(self)
        self.i=i

    def predict(self, base_mu, base_sig, dgp_mu, dgp_sig):
        base_mu_samples = tf.tile(tf.expand_dims(base_mu[self.i], 0), [tf.shape(base_mu[0])[0], 1, 1, 1])
        base_sig_samples = tf.tile(tf.expand_dims(base_sig[self.i], 0), [tf.shape(base_mu[0])[0], 1, 1, 1])

        return base_mu_samples, base_sig_samples

    def sample(self, base_mu, base_sig, dgp_mu, dgp_sig, num_samples):
        base_mu = tf.Print(base_mu, [tf.shape(base_mu)], 'base_mu: ')
        base_mu_samples = tf.tile(tf.expand_dims(base_mu[self.i], 0), [num_samples, 1, 1, 1])
        base_sig_samples = tf.tile(tf.expand_dims(base_sig[self.i], 0), [num_samples, 1, 1, 1])

        z = tf.random_normal(tf.shape(base_mu_samples), dtype=settings.float_type)
        samples =  utils.reparameterize(base_mu_samples, base_sig_samples, z, full_cov = False) # num_samples x N x S x 1

        return samples


class MR_DGP_Only(MR_Mixing_Weights):
    def __init__(self, i=0):
        MR_Mixing_Weights.__init__(self)
        self.i=i

    def predict(self, base_mu, base_sig, dgp_mu, dgp_sig):
        base_mu_samples = dgp_mu[self.i]
        base_sig_samples = dgp_sig[self.i]

        return base_mu_samples, base_sig_samples

    def sample(self, base_mu, base_sig, dgp_mu, dgp_sig, num_samples):
        base_mu_samples = dgp_mu[self.i]
        base_sig_samples = dgp_sig[self.i]

        z = tf.random_normal(tf.shape(base_mu_samples), dtype=settings.float_type)
        samples =  utils.reparameterize(base_mu_samples, base_sig_samples, z, full_cov = False) # num_samples x N x S x 1

        return samples


class MR_Variance_Mixing(MR_Mixing_Weights):
    def __init__(self):
        MR_Mixing_Weights.__init__(self)

    def predict(self, base_mu, base_sig, dgp_mu, dgp_sig):
        pred_vars = []
        for i in range(len(base_sig)):
            pred_var = base_sig[i]
            _train_max = tf.reduce_max(pred_var)
            _train_min = tf.reduce_min(pred_var)

            pred_var = tf.log(pred_var)
            _train_max = tf.log(_train_max)
            _train_min = tf.log(_train_min)

            #avg_training_pred = (_train_min+_train_max)/2 #for aq
            avg_training_pred = (_train_min+0.01)/2 #for biased observations
            mm = avg_training_pred
            stretch_constant = 10.0 
            pred_var = (tf.tanh((pred_var-mm)*stretch_constant)+1)/2

            pred_var = tf.tile(tf.expand_dims(pred_var, 0), [tf.shape(dgp_mu[0])[0], 1, 1, 1])
            pred_vars.append(pred_var)

        base_mu_samples = tf.tile(tf.expand_dims(base_mu[0], 0), [tf.shape(dgp_mu[0])[0], 1, 1, 1])
        base_sig_samples = tf.tile(tf.expand_dims(base_sig[0], 0), [tf.shape(dgp_mu[0])[0], 1, 1, 1])

        mu_weight = pred_vars[0]
        sig_weight = tf.multiply(pred_vars[0], pred_vars[0])

        mu_total = tf.multiply(1-pred_vars[0], base_mu_samples)
        sig_total = tf.multiply(1-pred_vars[0], tf.multiply(1-pred_vars[0], base_mu_samples))

        reset_prev_mu = True
        prev_mu = None
        prev_sig = None

        for i in range(len(dgp_mu)):
            if reset_prev_mu:
                prev_mu = tf.multiply(mu_weight, dgp_mu[i])
                prev_sig = tf.multiply(sig_weight, dgp_sig[i])
                reset_prev_mu = False
            else:
                #pred_vars is [base[0], dgp[0], dgp[1]]
                #use prev_vars[i] because we want scale by the weight of the previous dgp
                prev_mu = tf.multiply(1-pred_vars[i], prev_mu)
                prev_sig = tf.multiply(1-pred_vars[i], tf.multiply(1-pred_vars[i], prev_mu))


                mu_weight = tf.multiply(pred_vars[i], mu_weight)
                sig_weight = tf.multiply(pred_vars[i], tf.multiply(pred_vars[i], sig_weight))

                prev_mu +=  tf.multiply(mu_weight, dgp_mu[i])

                mu_total += prev_mu
                sig_total += prev_sig

                reset_prev_mu = True

        if not reset_prev_mu:
            mu_total += prev_mu
            sig_total += prev_sig


        return mu_total, sig_total

    def sample(self, base_mu, base_sig, dgp_mu, dgp_sig, num_samples):
        base_mu_samples = dgp_mu[0]
        base_sig_samples = dgp_sig[0]

        z = tf.random_normal(tf.shape(base_mu_samples), dtype=settings.float_type)
        samples =  utils.reparameterize(base_mu_samples, base_sig_samples, z, full_cov = False) # num_samples x N x S x 1

        return samples

class MR_Variance_Mixing_1(MR_Mixing_Weights):
    def __init__(self):
        MR_Mixing_Weights.__init__(self)

    def predict(self, base_mu, base_sig, dgp_mu, dgp_sig):
        pred_vars = []
        for i in range(len(base_sig)):
            pred_var = base_sig[i]
            _train_max = tf.reduce_max(pred_var)
            _train_min = tf.reduce_min(pred_var)

            pred_var = tf.log(pred_var)
            _train_max = tf.log(_train_max)
            _train_min = tf.log(_train_min)

            avg_training_pred = (_train_min+0.01)/2 #for aq
            mm = avg_training_pred
            stretch_constant = 2.0 
            pred_var = (tf.tanh((pred_var-mm)*stretch_constant)+1)/2

            pred_var = tf.tile(tf.expand_dims(pred_var, 0), [tf.shape(dgp_mu[0])[0], 1, 1, 1])
            pred_vars.append(pred_var)

        base_mu_samples = tf.tile(tf.expand_dims(base_mu[0], 0), [tf.shape(dgp_mu[0])[0], 1, 1, 1])
        base_sig_samples = tf.tile(tf.expand_dims(base_sig[0], 0), [tf.shape(dgp_mu[0])[0], 1, 1, 1])

        mu_weight = pred_vars[0]
        sig_weight = tf.multiply(pred_vars[0], pred_vars[0])

        mu_total = tf.multiply(1-pred_vars[0], base_mu_samples)
        sig_total = tf.multiply(1-pred_vars[0], tf.multiply(1-pred_vars[0], base_sig_samples))

        reset_prev_mu = True
        prev_mu = None
        prev_sig = None

        for i in range(len(dgp_mu)):
            if reset_prev_mu:
                prev_mu = tf.multiply(mu_weight, dgp_mu[i])
                prev_sig = tf.multiply(sig_weight, dgp_sig[i])
                reset_prev_mu = False
            else:
                #pred_vars is [base[0], dgp[0], dgp[1]]
                #use prev_vars[i] because we want scale by the weight of the previous dgp
                prev_mu = tf.multiply(1-pred_vars[i], prev_mu)
                prev_sig = tf.multiply(1-pred_vars[i], tf.multiply(1-pred_vars[i], prev_mu))

                mu_weight = tf.multiply(pred_vars[i], mu_weight)
                sig_weight = tf.multiply(pred_vars[i], tf.multiply(pred_vars[i], sig_weight))

                mu_total += prev_mu
                sig_total += prev_sig

                reset_prev_mu = True

        if not reset_prev_mu:
            mu_total += prev_mu
            sig_total += prev_sig


        return mu_total, sig_total

    def sample(self, base_mu, base_sig, dgp_mu, dgp_sig, num_samples):
        base_mu_samples = dgp_mu[0]
        base_sig_samples = dgp_sig[0]

        z = tf.random_normal(tf.shape(base_mu_samples), dtype=settings.float_type)
        samples =  utils.reparameterize(base_mu_samples, base_sig_samples, z, full_cov = False) # num_samples x N x S x 1

        return samples
