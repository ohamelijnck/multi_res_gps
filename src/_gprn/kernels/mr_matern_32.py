import numpy as np
import tensorflow as tf

from . import Kernel
from .. import util
from .. import Parameters

class MR_MATERN_32(Kernel):
    _id = 0
    def __init__(self, num_dimensions=1, num_components = 1, length_scale = 0.1, means = None, variances=None , var_scale=1.0, mean_scale=1.0, init=False, mask=None, train=True):
        super(MR_MATERN_32, self).__init__(mask)
        MR_MATERN_32._id += 1
        self.id = MR_MATERN_32._id
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
            self.length_scales = self.parameters.create(name='MR_MATERN_32_length_scale_'+str(self.id), init=self.length_scale*tf.ones(shape=[self.num_dimensions]), trainable=True)
        else:
            self.length_scales = self.parameters.create(name='MR_MATERN_32_length_scale_'+str(self.id), init=[self.length_scale], trainable=True)


        self.sigma = self.parameters.create(name='MR_MATERN_32_sigma_'+str(self.id), init=0.0, trainable=True, scope=Parameters.HYPER_SCOPE)

    def _kernel_non_vectorised(self, _X1, _X2, jitter=False, debug=False, include_dimensions=None):
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
 
    def _kernel(self, _X1, _X2, jitter=False, debug=False, include_dimensions=None):
        """
        inputs:
            X1 \in R^{N \cross D}
            X2 \in R^{M \cross D}
        output:
            K \in R^{N \cross M}
        """

        X1_raw = _X1
        X2_raw = _X2
        sigma = util.var_postive(self.sigma)
        #sigma = 1.0
        ls = util.var_postive(self.length_scales)

        if include_dimensions is not None:
            #ls = self.s(ls, include_dimensions)
            #print('include_dimensions: ', include_dimensions[0])
            _X1 = tf.expand_dims(_X1[:, :, include_dimensions[0]], -1) #N x S x 1
            _X2 = tf.expand_dims(_X2[:, :, include_dimensions[0]], -1) #N x S x 1
        else:
            include_dimensions = []


        X1 = _X1 # 
        X2 = tf.transpose(_X2, perm=[0, 2, 1]) #D x 1 x N2
        T = tf.transpose(tf.subtract(X1, X2), perm=[0, 1, 2])  # D x N1 x N2

        r = T
        r = tf.abs(r)
        #r = tf.clip_by_value(T, 0, 1e8)


        ls= 1/(tf.expand_dims(tf.expand_dims(ls, -1), -1))
        r= ls*r

        k =  (1+tf.scalar_mul(np.sqrt(3), r))*util.safe_exp(-tf.scalar_mul(np.sqrt(3), r))
        k = sigma*k


        
        if jitter is True:
            #val =  util.add_jitter(val, self.context.jitter) 
            k =  k+(self.context.jitter*tf.eye(tf.shape(k)[1]))[None, :, :]


        return k


    def get_parameters(self):
        return [self.length_scales]


