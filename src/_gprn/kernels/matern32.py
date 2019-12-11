import numpy as np
import tensorflow as tf

from . import Kernel
from .. import util
from .. import Parameters

class Matern32(Kernel):
    _id = 0
    def __init__(self, num_dimensions=1, num_components = 1, length_scale = 0.1, means = None, variances=None , var_scale=1.0, mean_scale=1.0, init=False, mask=None, train=True):
        super(Matern32, self).__init__(mask)
        Matern32._id += 1
        self.id = Matern32._id
        self.ARD=True
        self.length_scale = length_scale
        self.num_dimensions = num_dimensions
        self.train = train
        self.setup_flag = False


    def setup(self, context):
        self.setup_flag = True
        self.context = context
        self.parameters = self.context.parameters
        if self.ARD:
            self.length_scales = self.parameters.create(name='matern32_length_scale_'+str(self.id), init=self.length_scale*tf.ones(shape=[self.num_dimensions]), trainable=True)
        else:
            self.length_scales = self.parameters.create(name='matern32_length_scale_'+str(self.id), init=[self.length_scale], trainable=True)


        self.sigma = self.parameters.create(name='matern32_sigma_'+str(self.id), init=0.0, trainable=True, scope=Parameters.HYPER_SCOPE)

    def _kernel(self, _X1, _X2, jitter=False, debug=False, include_dimensions=None):
        ls = util.var_postive(self.length_scales)
        sigma = util.var_postive(self.sigma)

        if include_dimensions is not None:
            ls = ls[include_dimensions]

        X1 = tf.transpose(tf.expand_dims(_X1, -1), perm=[1, 0, 2])  # D x N1 x 1
        X2 = tf.expand_dims(tf.transpose(_X2, perm=[1, 0]), -2)  # D x N2 x 1
        r = tf.transpose(tf.abs(tf.subtract(X1, X2)), perm=[0, 1, 2])  # D x N1 x N2

        if include_dimensions is not None:
            r = r[include_dimensions, :, :]

        r = tf.clip_by_value(r, 0, 1e8)
        r = tf.transpose(tf.transpose(r)/ls)

        k =  (1+tf.scalar_mul(np.sqrt(3), r))*util.safe_exp(-tf.scalar_mul(np.sqrt(3), r))

        k = sigma*tf.reduce_prod(k, axis=0)

        if jitter:
            k =  util.add_jitter(k, self.context.jitter) 

        return k
 
    def get_parameters(self):
        return [self.length_scales]


