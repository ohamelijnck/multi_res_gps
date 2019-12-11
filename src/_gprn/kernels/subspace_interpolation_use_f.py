import numpy as np
import tensorflow as tf

from . import Kernel
from .. import util

class SubspaceInterpolationUseF(Kernel):
    _id = 0
    def __init__(self, K1, K2, inducing_locations, interpolation_index = [0], mask=None):
        super(SubspaceInterpolationUseF, self).__init__(mask)
        #SubspaceInterpolation._id += 1
        self.inducing_locations = inducing_locations
        self.K1 = K1
        self.K2 = K2
        #self.interpolation_index = interpolation_index
        #self.rest_index = np.setdiff1d(np.arange(K2.num_dimensions), self.interpolation_index)

    def setup(self, context):
        print('SETTING UP')
        self.context = context
        self.parameters = self.context.parameters
        #does not need to be setup as it will be setup in base model
        self.K1.setup(self.context)
        #self.K2.setup(self.context)
        #self.a = tf.Variable(0, name='polynomial_a_'+str(SubspaceInterpolation._id), dtype=tf.float32, trainable=True)
        self.a = self.parameters.create(name='polynomial_a_'+str(SubspaceInterpolationUseF._id), init=0.0, trainable=True)

    def standardise(self, k1, k2, K):
        m1 = tf.transpose(tf.reshape(tf.tile(k1, [tf.shape(k2)[0]]), [tf.shape(k2)[0], tf.shape(k1)[0]]))
        m2 = tf.reshape(tf.tile(k2, [tf.shape(k1)[0]]), [tf.shape(k1)[0], tf.shape(k2)[0]])

        m3 = tf.sqrt(tf.multiply(m1, m2))
        return tf.div(K, m3) #element wise division

    def get_f(self, X):
        sparsity = self.context._model.model.elbo.sparsity
        mu_x1, _, _, _ = sparsity._build_intermediate_conditionals(0, self.context.use_latent_f_target, X)
        return mu_x1

    def _kernel(self, X1, X2, jitter=False, debug=False):
        #local kernel
        k_2_overlap= self.K2.k2.kernel(X1, X2)
        k_2_independent = self.K2.k2.kernel(X1, X2)
        k_1_independent = self.K1.k2.kernel(X1, X2)
        k_1_overlop = self.K1.k1._kernel(X1, X2)

        mu_x1 = self.get_f(X1)
        mu_x2 = self.get_f(X2)


        _K = lambda x1, x2: tf.pow(tf.matmul(x1, x2, transpose_b=True) + util.var_postive(self.a), 1.0)
        #_K = lambda x1, x2: tf.pow(tf.matmul(x1, x2, transpose_b=True), 1.0)
        K = _K(mu_x1[0, :], mu_x2[0, :])
        K_11 = _K(mu_x1[0, :], mu_x1[0, :])
        K_22 = _K(mu_x2[0, :], mu_x2[0, :])
        #K = tf.pow(K, 2.0)
        K = self.standardise(tf.diag_part(K_11), tf.diag_part(K_22) , K)

        r = 0.1
        #K = r*k_1_overlop+(1-r)*K
        #K = K_zz
        #K = tf.multiply(k_1_independent, K)


        if jitter: 
            K = util.add_jitter(K, self.context.jitter)

        return K
   
    def __kernel(self, X1, X2, jitter=False, debug=False):
        self.inducing_locations = self.context.parameters.get(name='inducing_locations_{r}'.format(r=1))

        k_g_zz = self.K2.k1.kernel(self.inducing_locations, self.inducing_locations, jitter=True) 
        k_g_z_x2 = self.K1.k1.kernel(self.inducing_locations, X2)
        k_g_x1_z = self.K1.k1.kernel(X1, self.inducing_locations)

        #k_g_x1_x2 = self.K2.kernel(X1, X2)
        #k_2_x1_x2 = self.K1.kernel(X1, X2)

        K =  tf.matmul(k_g_x1_z, tf.cholesky_solve(tf.cholesky(k_g_zz), k_g_z_x2))

        #K = k_2_x1_x2

        if False:
            K = tf.Print(K, [k_g_zz], 'k_g_zz_latent')
            K = tf.Print(K, [self.inducing_locations], 'self.inducing_locations')
            K = tf.Print(K, [X1], 'X1')
            K = tf.Print(K, [k_g_x1_z], 'k_g_x1_z_latent')
            K = tf.Print(K, [k_g_z_x2], 'k_g_z_x2_latent')
            K = tf.Print(K, [K], 'K_latent', summarize=500)

        #K = k_2_x1_x2

        #K = k_2_x1_x2

        K = tf.multiply(K, self.K2.k2.kernel(X1, X2, jitter=jitter))


        #K = k_g_x1_x2

        if jitter: 
            K = util.add_jitter(K, self.context.jitter)
        return K 

    def get_parameters(self):
        return [self.a]+self.K2.get_parameters()
        #return [self.length_scales]



