import numpy as np
import tensorflow as tf

from . import SingleGP
from . import Composite
from .. import util
from ..elbos import *
from ..elbos.ell import *
from ..predictions import PredictionSingleGP
from ..kernels import *
from ..likelihoods import CompositeLikelihood
from ..likelihoods import SingleGPLikelihood

class LatentAggr(SingleGP, Composite):
    def __init__(self,  context):
        SingleGP.__init__(self, context)
        Composite.__init__(self, context)
        self.context.model = 'LatentAggr'
        self.context = context
        #self.elbo = LatentAggrELBO(self.context)
        self.context.num_latent_process = 2
        self.context.num_likelihood_components = 2

        self.predictor = PredictionSingleGP(self.context)

        self._kernels_0 = self.context.kernels[0]['f'][0]
        self._kernels_1 = self.context.kernels[1]['f'][0] 
       
    def _setup_multi_res_variables(self):
        self._setup_standard_variables()
        self.inducing_locations_arr = []
        self.noise_sigma_arrs = []

        self.lik_weights = self.parameters.save(name='likelihood_weights', var=np.ones(self.context.num_likelihood_components))

        for r in range(1, self.context.num_likelihood_components):
            num_inducing = self.data.get_num_inducing(source=r)

            if  self.use_diag_covar:
                num_sigma = num_inducing
            else:
                num_sigma = int(num_inducing*(num_inducing+1)/2)

            inducing_locations = self.parameters.create(name='inducing_locations_{r}'.format(r=r), init=self.data.get_inducing_points_from_source(r).astype(np.float32),  trainable=self.train_inducing_points_flag)
            self.inducing_locations_arr.append(inducing_locations)

            q_means_u = self.parameters.create(name='q_means_u_{r}'.format(r=r), init=tf.random_uniform([self.q_num_components, self.num_latent, num_inducing], -1, 1, tf.float32, seed=0),  trainable=True)
            q_means_v = self.parameters.create(name='q_means_v_{r}'.format(r=r), init=tf.random_uniform([self.q_num_components, self.num_outputs, self.num_latent, num_inducing], 0, 1, tf.float32, seed=1),  trainable=True)

            q_covars_u_raw = self.parameters.create(name='q_covars_u_{r}_raw'.format(r=r), init=0.1*tf.random_uniform([self.q_num_components, self.num_latent, num_sigma], -0.5, 0.5, dtype=tf.float32, seed=0),  trainable=True)
            q_covars_v_raw = self.parameters.create(name='q_covars_v_{r}_raw'.format(r=r), init=tf.random_uniform([self.q_num_components, self.num_outputs, self.num_latent, num_sigma], -0.1, 0.1, tf.float32, seed=1),  trainable=True)

            #q_covars_u_raw = self.parameters.save('q_covars_u_{r}_raw'.format(r=r), self.parameters.get(name='q_covars_u_{r}_raw'.format(r=0)))
            #q_covars_u_raw = self.parameters.save('q_covars_v_{r}_raw'.format(r=r), self.parameters.get(name='q_covars_v_{r}_raw'.format(r=0)))


            self.parameters.load_posterior_covariance(name='q_covars_u_{r}'.format(r=r), from_name='q_covars_u_{r}_raw'.format(r=r), shape=[self.q_num_components, self.num_latent, num_sigma], n=num_inducing)
            self.parameters.load_posterior_covariance(name='q_covars_v_{r}'.format(r=r), from_name='q_covars_v_{r}_raw'.format(r=r), shape=[self.q_num_components, self.num_outputs, self.num_latent, num_sigma], n=num_inducing)

            self.parameters.load_posterior_cholesky(name='q_cholesky_u_{r}'.format(r=r), from_name='q_covars_u_{r}_raw'.format(r=r), shape=[self.q_num_components, self.num_latent, num_sigma], n=num_inducing)
            self.parameters.load_posterior_cholesky(name='q_cholesky_v_{r}'.format(r=r), from_name='q_covars_v_{r}_raw'.format(r=r), shape=[self.q_num_components, self.num_outputs, self.num_latent, num_sigma], n=num_inducing)

            sig = self.parameters.create(name='noise_sigma_{r}'.format(r=r), init=self.context.noise_sigmas[r][0], trainable=self.context.noise_sigmas[r][1])
            self.noise_sigma_arrs.append(sig)

        inducing_locations = self.parameters.get(name='inducing_locations_{r}'.format(r=0))
        if True:
            if self.context.use_latent_f:
                if False:
                    self.context.use_latent_f_direction =1
                    self.context.use_latent_f_target = 0
                    self.context.kernels[1]['f'][0].K1 = SubspaceInterpolationUseF(self._kernels_1, self._kernels_0, inducing_locations, interpolation_index=self.context.interpolation_index)
                else:
                    self.context.use_latent_f_target = 1
                    self.context.kernels[0]['f'][0] = SubspaceInterpolationUseF(self._kernels_0 , self._kernels_1, inducing_locations, interpolation_index=self.context.interpolation_index)
                #self.context.kernels[1]['f'][0] = SubspaceInterpolationUseF(self._kernels_1, self._kernels_0, inducing_locations, interpolation_index=self.context.interpolation_index)
            else:
                self.context.use_latent_f_direction = 0
                #self.context.kernels[1]['f'][0] = SubspaceInterpolation(self._kernels_0 , self._kernels_1, inducing_locations, interpolation_index=self.context.interpolation_index)
                self.context.kernels[0]['f'][0] = SubspaceInterpolation(self._kernels_1, self._kernels_0, inducing_locations, interpolation_index=self.context.interpolation_index)
                #self.context.kernels[0]['f'][0] = SubspaceInterpolation(self._kernels_1, self._kernels_0, inducing_locations, interpolation_index=self.context.interpolation_index)

        #self.context.kernels[0]['f'][0] = SubspaceInterpolation(self._kernels_1 , self._kernels_0, self.inducing_locations_arr[0], interpolation_index=[0, 1])
        #self.context.kernels[0]['f'][0] = SubspaceInterpolation(self._kernels_0 , self._kernels_1, self.inducing_locations_arr[0], interpolation_index=[0, 1])

    def _setup_elbo(self):
        ell_arr = [GP_ELL(self.context, r=0), GP_ELL(self.context, r=1)]
        self.ell = Composite_ELL(self.context, ell_arr)

        self.elbo = SingleGP_ELBO(self.context, ell=self.ell)

        lik_arr = [SingleGPLikelihood(self.context, r=0), SingleGPLikelihood(self.context, r=1)]
        #lik_arr = [GPRN_Aggr_Likelihood(self.context, r=1)]

        #self.likelihood = CompositeLikelihood(self.context, lik_arr)
        self.likelihood = SingleGPLikelihood(self.context, r=0)

    def setup(self, data):
        self.data = data
        self._setup_multi_res_variables()

    def build_elbo_graph(self):
        self._setup_elbo()
        self.elbo.setup(self.data)
        return self.elbo.build_graph()

    def _setup_standard_kernels(self):
        self.setup_base_kernels()

    def fit(self):
        pass

 





 
