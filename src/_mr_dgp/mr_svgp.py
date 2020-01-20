import tensorflow as tf
import numpy as np

from .mr_gaussian import MR_Gaussian
from . import utils

from gpflow.params import Parameter, Parameterized
from gpflow.features import InducingPoints
from gpflow import transforms
from gpflow import settings
from gpflow import params_as_tensors, ParamList




class MR_SVGP(Parameterized):
    def __init__(self, Z, K, noise_sigma, white=True,**kwargs):
        Parameterized.__init__(self, **kwargs)

        self.white=True
        self.num_inducing = Z.shape[0]
        self.inducing_locations = Z
        self.K = K

        
        self.likelihood = MR_Gaussian(variance=noise_sigma)
        
        self.setup()

    def setup(self):
        self.setup_variational_parameters()

    def setup_variational_parameters(self):
        self.Z = Parameter(self.inducing_locations) # M x D

        self.q_mu = Parameter(np.zeros((self.num_inducing, 1))) # M x 1

        q_sqrt = np.tile(np.eye(self.num_inducing)[None, :, :], [1, 1, 1])
        transform = transforms.LowerTriangular(self.num_inducing, num_matrices=1)
        self.q_sqrt = Parameter(q_sqrt, transform=transform) # 1 x M x M

    @params_as_tensors
    def get_z(self):
        return self.Z
    
        
    def sample(self, mu, sig, num_samples):
        mu = tf.expand_dims(mu, 0) # 1 x N x S x 1
        sig = tf.expand_dims(sig, 0) # 1 x N x S x S
        mu = tf.tile(mu, [num_samples, 1, 1, 1])
        sig = tf.tile(sig, [num_samples, 1, 1, 1])

        z = tf.random_normal(tf.shape(mu), dtype=settings.float_type)
        samples =  utils.reparameterize(mu, sig, z, full_cov = False) # num_samples x N x S x 1
        return samples

    def marginal(self, m, s_chol, k_zz,  k_xz, k_xx):
        #k_zz = tf.tile(k_zz, [tf.shape(k_xz)[0], 1, 1]) # N x M x M
        
        _s_chol = s_chol
        s_chol = tf.tile(tf.expand_dims(s_chol, 0), [tf.shape(k_xz)[0], 1, 1]) # N x M x M

        #k_zz in 1 x M x M 
        k_zz_chol = tf.cholesky(tf.cast(k_zz, tf.float64)) #  M x M
        _k_zz_chol = k_zz_chol
        k_zz_chol = tf.tile(k_zz_chol, [tf.shape(k_xz)[0], 1, 1]) # N x M x M

        if self.white:
            mu = tf.matmul(k_xz, tf.linalg.triangular_solve(tf.transpose(k_zz_chol, [0, 2 ,1]), m, lower=False, name='m_'))
        else:
            mu = tf.matmul(k_xz, tf.linalg.triangular_solve(tf.transpose(k_zz_chol), tf.linalg.triangular_solve(k_zz_chol, m, lower=True), lower=False))

        A = tf.linalg.triangular_solve(k_zz_chol, tf.transpose(k_xz, [0, 2, 1]), lower=True, name='A1_') # N x M x S
        sig = k_xx - tf.matmul(tf.transpose(A, [0, 2, 1]), A) # N x S x S

        if self.white:
            a = tf.linalg.triangular_solve(tf.transpose(_k_zz_chol[0, :, :]), _s_chol, lower=False, name='A2_')
            a = tf.expand_dims(a, 0)
            a = tf.tile(a, [tf.shape(k_xz)[0], 1, 1])
            A = tf.matmul(k_xz, a)
            #A = tf.matmul(k_xz, util.tri_mat_solve(tf.transpose(k_zz_chol, [0, 2, 1]), s_chol, lower=False, name='A2_'))
        else:
            A = tf.matmul(k_xz, util.mat_solve(k_zz, s_chol))

        sig = sig + tf.matmul(A, tf.transpose(A, [0, 2, 1]))

        return mu, sig

    @params_as_tensors
    def conditional(self, XS):
        #XS in N x S x D
        N = tf.shape(XS)[0]

        _Z = tf.expand_dims(self.Z, 0) #1 x M x D

        k_xx = self.K.K(XS) # N x S x S
        k_xz = self.K.K(XS, _Z) # N x S x M
        k_zz = self.K.K(_Z) # 1 x M x M

        m = tf.tile(tf.expand_dims(self.q_mu, 0), [N, 1, 1]) # N x M x 1
        s_sqrt =  self.q_sqrt[0, :, :] # M x M
        return self.marginal(m, s_sqrt, k_zz, k_xz, k_xx)


    def kl_term(self):
        KL = -0.5 * tf.cast(self.num_inducing, settings.float_type)
        KL -= 0.5 * tf.reduce_sum(tf.log(tf.matrix_diag_part(self.q_sqrt) ** 2))

        if not self.white:
            KL += tf.reduce_sum(tf.log(tf.matrix_diag_part(self.Lu))) 
            KL += 0.5 * tf.reduce_sum(tf.square(tf.matrix_triangular_solve(self.Lu_tiled, q_sqrt, lower=True)))
            Kinv_m = tf.cholesky_solve(self.Lu, q_mu)
            KL += 0.5 * tf.reduce_sum(q_mu * Kinv_m)
        else:
            KL += 0.5 * tf.reduce_sum(tf.square(self.q_sqrt))
            KL += 0.5 * tf.reduce_sum(self.q_mu**2)

        return KL

    @params_as_tensors
    def expected_log_likelihood(self, Y, mu, sig):
        ell =  self.likelihood.variational_expectations(mu, sig, Y)
        return ell
