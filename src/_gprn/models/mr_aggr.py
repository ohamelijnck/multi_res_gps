import numpy as np
import tensorflow as tf

from . import Model
from . import SingleGP
from .. import util
from ..elbos import SingleGP_ELBO
from ..elbos.ell import *
from ..likelihoods import SingleGPLikelihood
from ..likelihoods import GP_Aggr_Likelihood
from ..likelihoods import CompositeLikelihood
from ..predictions import PredictionSingleGP

class MRAggr(DGP):
    def __init__(self,  context):
        super(GPAggr, self).__init__(context)
        self.context.model = 'GPAggr'
        self.context = context
        
    def _setup_multi_res_1(self):
        
        self._single_gp_setup()
        self.gprn_structure = True

        #ell_arr = [GP_ELL(self.context, r=0), GP_Aggr_ELL(self.context, r=1)]
        #ell_arr = [GP_ELL(self.context, r=0), GP_ELL(self.context, r=0)]
        ell_arr = [GPRN_Aggr_ELL(self.context, r=1)]

        self.ell = Composite_ELL(self.context, ell_arr)

        #self.elbo = MultiResT1ELBO(self.context, ell=self.ell)
        self.elbo = SingleGP_ELBO(self.context, ell=self.ell)

        #lik_arr = [SingleGPLikelihood(self.context, r=0), GP_Aggr_Likelihood(self.context, r=1)]
        lik_arr = [SingleGPLikelihood(self.context, r=0), SingleGPLikelihood(self.context, r=0)]
        #lik_arr = [GPRN_Aggr_Likelihood(self.context, r=1)]

        self.likelihood = CompositeLikelihood(self.context, lik_arr)
        self.predictor = PredictionSingleGP(self.context)


    def setup(self, data):
        self.data = data
        self._setup_multi_res_variables()
        self._setup_multi_res_1()

    
    def _setup_multi_res_variables(self):
        self.context.num_latent_process = 1
        self.context.num_likelihood_components = self.data.get_num_sources()

        self._setup_standard_variables()
        self.noise_sigma_arrs = []
        self.lik_weights = self.parameters.save(name='likelihood_weights', var=np.ones(self.context.num_likelihood_components).astype(np.float32))

        for r in range(1, self.context.num_likelihood_components):
            #different noise per lik component
            sig = self.parameters.create(name='noise_sigma_{r}'.format(r=r), init=self.context.noise_sigmas[r][0], trainable=self.context.noise_sigmas[r][1])
            self.noise_sigma_arrs.append(sig)

    def get_variational_parameters(self):
        arr = self.get_single_gp_variational_parameters()
        return arr

    def get_free_parameters(self):
        arr = self.noise_sigma_arrs
        return self.get_single_gp_free_parameters()+arr

    def fit(self):
        pass

    def build_elbo_graph(self):
        self.elbo.setup(self.data)
        return self.elbo.build_graph()


