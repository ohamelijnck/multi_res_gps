import numpy as np
import tensorflow as tf

from . import Kernel
from .. import util

class ModulatedSE(Kernel):
    _id = -1
    def __init__(self, num_dimensions=1, sigma=1.0, length_scale=1.0,ARD=True, mask=None):
        super(ModulatedSE, self).__init__(mask)
        ModulatedSE._id += 1
        self.id = ModulatedSE._id

        self.ARD = ARD
        self.length_scale = length_scale
        self._sigma = sigma
        self.num_dimensions = num_dimensions
        self.setup_flag = False

    def setup(self, context):
        self.setup_flag = True
        self.context = context
        self.parameters = self.context.parameters


        self.ls_u= self.parameters.create(name='modulated_se_length_scale_u_'+str(self.id), init=[self.length_scale], trainable=False)
        self.ls_g = self.parameters.create(name='modulated_se_length_scale_g_'+str(self.id), init=[self.length_scale], trainable=False)

        self.parameters = [self.ls_u, self.ls_g]

    def _kernel(self, X1, X2, jitter=False, debug=False, include_dimensions=None):
        ls_u = util.var_postive(self.ls_u)
        ls_u = self.ls_u
        ls_g = util.var_postive(self.ls_g)
        ls_g = self.ls_g

        ls_e = tf.sqrt(1/((2/tf.square(ls_g)) + (1/tf.square(ls_u))))
        ls_s = tf.sqrt(2*tf.square(ls_g) + (tf.square(tf.square(ls_g))/tf.square(ls_u)))
        ls_m = tf.sqrt(2*tf.square(ls_u) + tf.square(ls_g))

        ls_u = tf.Print(ls_u, [ls_u], 'ls_u: ')
        ls_u = tf.Print(ls_u, [ls_g], 'ls_g: ')
        ls_u = tf.Print(ls_u, [ls_e], 'ls_e: ')
        ls_u = tf.Print(ls_u, [ls_s], 'ls_s: ')
        ls_u = tf.Print(ls_u, [ls_m], 'ls_m: ')

        lhs = util.safe_exp(-tf.matmul(X1, X1, transpose_b = True)/(2*tf.square(ls_m)))
        lhs = tf.transpose(tf.reshape(tf.tile(tf.diag_part(lhs), [tf.shape(X2)[0]]), [tf.shape(X2)[0], tf.shape(X1)[0]]))


        _X1 = tf.transpose(tf.expand_dims(X1, -1), perm=[1, 0, 2])  # D x N1 x 1
        _X2 = tf.expand_dims(tf.transpose(X2, perm=[1, 0]), -2)  # D x N2 x 1
        T = tf.transpose(tf.subtract(_X1, _X2), perm=[0, 1, 2])  # D x N1 x N2

        mhs = util.safe_exp(-tf.square(T)/(2*tf.square(ls_s)))
        mhs = tf.reduce_prod(mhs, axis=0)


        rhs = util.safe_exp(-tf.matmul(X2, X2, transpose_b = True)/(2*tf.square(ls_m)))
        rhs = tf.reshape(tf.tile(tf.diag_part(rhs), [tf.shape(X1)[0]]), [tf.shape(X1)[0], tf.shape(X2)[0]])
        
        #ls_u = tf.Print(ls_u, [lhs], 'lhs: ')
        #ls_u = tf.Print(ls_u, [mhs], 'mhs: ', summarize=100)
        #ls_u = tf.Print(ls_u, [rhs], 'rhs: ')

        K = tf.multiply(tf.multiply(lhs, mhs), rhs)
        c =  tf.pow((ls_e/ls_u), self.num_dimensions)

        K = c*K

        if jitter is True:
            K =  util.add_jitter(K, self.context.jitter) 

        return K
    
