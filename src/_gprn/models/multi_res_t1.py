import numpy as np
import tensorflow as tf

from . import Model
from . import Standard
from .. import util
from ..elbos import StandardELBO
from ..elbos.ell import *
from ..likelihoods import StandardGPRNLikelihood
from ..likelihoods import GPRN_Aggr_Likelihood
from ..likelihoods import CompositeLikelihood
from ..predictions import PredictionStandard
from ..predictions import PredictionPositiveW

class MultiResT1(Standard):
    def __init__(self,  context):
        super(MultiResT1, self).__init__(context)
        self.context.model = 'MultiResT1'
        self.context = context

        self.mean_v_scale = -0.1
        self.covar_v_scale = -0.1
        
    def _setup_multi_res_1(self):
        
        self._standard_setup()
        self.gprn_structure = True

        #ell_arr = [GPRN_ELL(self.context, r=0), GPRN_Aggr_ELL(self.context, r=1)]
        ell_arr = [GPRN_Positive_W_ELL(self.context, r=0), GPRN_Aggr_Positive_W_ELL(self.context, r=1)]
        #ell_arr = [GPRN_Aggr_ELL(self.context, r=1)]

        self.ell = Composite_ELL(self.context, ell_arr)

        #self.elbo = MultiResT1ELBO(self.context, ell=self.ell)
        self.elbo = StandardELBO(self.context, ell=self.ell)

        lik_arr = [StandardGPRNLikelihood(self.context, r=0), GPRN_Aggr_Likelihood(self.context, r=1)]
        #lik_arr = [GPRN_Aggr_Likelihood(self.context, r=1)]

        self.likelihood = CompositeLikelihood(self.context, lik_arr)
        #self.predictor = PredictionStandard(self.context)
        self.predictor = PredictionPositiveW(self.context)

    def setup(self, data):
        self.data = data
        self._setup_multi_res_variables()
        self._setup_multi_res_1()

    def _setup_multi_res_variables(self):
        self.context.num_latent_process = 1
        self.context.num_likelihood_components = self.data.get_num_sources()

        self._setup_standard_variables()
        self.noise_sigma_arrs = []
        self.lik_weights = self.parameters.save(name='likelihood_weights', var=np.ones(self.context.num_likelihood_components))

        for r in range(1, self.context.num_likelihood_components):
            #different noise per lik component
            sig = self.parameters.create(name='noise_sigma_{r}'.format(r=r), init=self.context.noise_sigmas[r][0], trainable=self.context.noise_sigmas[r][1])
            self.noise_sigma_arrs.append(sig)

    def get_variational_parameters(self):
        arr = self.get_standard_variational_parameters()
        return arr

    def get_free_parameters(self):
        arr = self.noise_sigma_arrs
        return self.get_standard_free_parameters()+arr

    def fit(self):
        pass

    def build_elbo_graph(self):
        self.elbo.setup(self.data)
        return self.elbo.build_graph()
