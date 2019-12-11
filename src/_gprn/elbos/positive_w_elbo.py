import tensorflow as tf
import numpy as np
import math
from . import ELBO
from . import StandardELBO
from .. import util
from .. import debugger
from .. sparsity import Sparsity
from .. precomputers import Precomputed


class PositiveWELBO(StandardELBO):
    def __init__(self,  context, ell):
        super(PositiveWELBO, self).__init__(context, ell)

    def get_expected_values(self, mu_f, sigma_f, mu_w, sigma_w):
        m_w = util.safe_exp(mu_w+0.5*sigma_w)
        s_w = util.safe_exp(2*(mu_w+sigma_w))-m_w*mu_w

        return mu_f, sigma_f, m_w, s_w

    def get_f_w(self, k=0, p=0):
        _mu_f, _sigma_f, _mu_w, _sigma_w = self.sparsity._build_intermediate_conditionals(k, self.x_train, predict=True)

        mu_f = _mu_f[:, :, 0]
        mu_wi = _mu_w[p,:,:,0]
        sigma_f = _sigma_f[0, :, :]
        sigma_wi = _sigma_w[p,0,:,:]
        return get_expected_values(mu_f, sigma_f, mu_wi, sigma_wi)

    def get_posterior(self, r=0):
        k=0
        p=0
        x_train = self.ell.ell_arr[r].x_train
        shp= tf.shape(x_train)
        x_stacked = tf.reshape(tf.transpose(x_train, perm=[1, 0, 2]), [shp[0]*shp[1], shp[2]])
        _mu_f, _sigma_f, _mu_w, _sigma_w = self.sparsity._build_intermediate_conditionals(k, 0, x_stacked, predict=True)

        _mu_f= tf.Print(_mu_f, [tf.shape(_mu_f)], '_mu_f: ')
        _mu_f= tf.Print(_mu_f, [tf.shape(_sigma_f)], '_sigma_f: ')
        _mu_f= tf.Print(_mu_f, [tf.shape(_mu_w)], '_mu_w: ')
        _mu_f= tf.Print(_mu_f, [tf.shape(_sigma_w)], '_sigma_w: ')


        mu_f = _mu_f[:, :, 0]
        mu_wi = _mu_w[p,:,:,0]
        sigma_f = _sigma_f[0, :]
        sigma_wi = _sigma_w[p,0,:]
        #return mu_f, sigma_f, mu_wi, sigma_wi
        return self.get_expected_values(mu_f, sigma_f, mu_wi, sigma_wi)

        
        
   
