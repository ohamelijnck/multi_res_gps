from . import CompositeCorrections
from .. import Parameters
from ..scores import *

import numpy as np
import tensorflow as tf
class CurvatureCorrection(CompositeCorrections, Parameters):
    def __init__(self, context):
        Parameters.__init__(self, context)
        CompositeCorrections.__init__(self, context)
        self.context = context
        self.J = None
        self.H = None

    def get_c(self):
        M = np.linalg.cholesky(self.H+0.01*np.eye(self.H.shape[0]))
        Ma = np.linalg.cholesky(np.matmul(self.H,np.linalg.solve(self.J+0.01*np.eye(self.J.shape[0]), self.H))+0.01*np.eye(self.H.shape[0]))
        C = np.linalg.solve(M, Ma)
        return C

  

    def unwrap_params(self, params):
        arr = []
        for p in params:
            a = tf.reshape(p, [tf.reduce_prod(tf.shape(p))])
            arr.append(a)
        return tf.concat(arr, axis=0)

    def apply_correction(self):
        C = self.get_c()
        params = self.get_params()
        theta_hat = [tf.convert_to_tensor(p) for p in self.theta_hat]


        theta = self.unwrap_params(params)
        theta_hat = self.unwrap_params(theta_hat)


        transformed = theta_hat + tf.squeeze(tf.matmul(tf.convert_to_tensor(C, dtype=tf.float32), tf.expand_dims(theta-theta_hat, -1)))


        i = 0
        for p in params:
            _prod = np.prod(p.shape)
            _p = tf.reshape(transformed[i:(i+_prod)], tf.shape(p))

            p_name= p.name.split('/')[1]
            self.save(name=p_name, var=_p)
            i = i+_prod

    def learn_information_matrices(self, session, optimise_flag = True):
        print('learn_information_matrices')
        self.estimate_information_matrices(optimise_flag)
        print('get_params')
        self.theta_hat = self.get_params(session)


