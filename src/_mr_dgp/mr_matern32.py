import numpy as np
import tensorflow as tf
import gpflow
from gpflow import settings
from gpflow import params_as_tensors



class MR_MATERN_32(gpflow.kernels.Stationary):
    @params_as_tensors
    def Kdiag(self, X, presliced=False):
        return tf.fill(tf.shape(X)[:-1], tf.squeeze(self.variance))

    @params_as_tensors
    def K(self, _X1, _X2=None, presliced=False):
        #Only implemented for input_dimension = 1


        sigma = self.variance
        #sigma = 1.0
        ls = self.lengthscales

        jitter_flag = False
        if _X2 is None:
            jitter_flag = True
            _X2 = _X1


        if not presliced:
            if isinstance(self.active_dims, np.ndarray):
                _X1 = tf.expand_dims(_X1[:, :, self.active_dims[0]], -1)
                _X2 = tf.expand_dims(_X2[:, :, self.active_dims[0]], -1)


        X1 = _X1 # 
        X2 = tf.transpose(_X2, perm=[0, 2, 1]) #D x 1 x N2
        T = tf.transpose(tf.subtract(X1, X2), perm=[0, 1, 2])  # D x N1 x N2

        r = T
        r = tf.abs(r)
        #r = tf.clip_by_value(T, 0, 1e8)


        ls= 1/(tf.expand_dims(tf.expand_dims(ls, -1), -1))
        r= ls*r

        k =  (1+tf.scalar_mul(np.sqrt(3), r))*tf.exp(-tf.scalar_mul(np.sqrt(3), r))
        k = sigma*k


        
        if jitter_flag is True:
            #val =  util.add_jitter(val, self.context.jitter) 
            jit = tf.cast(settings.jitter, settings.float_type)
            k =  k+(jit*tf.eye(tf.shape(k)[1]))[None, :, :]


        return k

