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
from ..predictions import PredictionMRSingleGP
from ..predictions import PredictionSingleGPLogTransform
from .. import Parameters

class GPAggr(SingleGP):
    def __init__(self,  context):
        super(GPAggr, self).__init__(context)
        self.context.model = 'GPAggr'
        self.context = context
        
    def _setup_multi_res_1(self):
        ell_arr = []
        lik_arr = []
        for r in range(self.context.num_likelihood_components):
            ell_arr.append(GP_Aggr_ELL(self.context, r=r))
            #ell_arr.append(GP_ELL(self.context, r=r))
            lik_arr.append(GP_Aggr_Likelihood(self.context, r=r))
            #lik_arr.append(SingleGPLikelihood(self.context, r=r))

        self.ell = Composite_ELL(self.context, ell_arr)
        self.likelihood = CompositeLikelihood(self.context, lik_arr)

        self.elbo = SingleGP_ELBO(self.context, ell=self.ell)

        if self.context.log_transform:
            self.predictor = PredictionSingleGPLogTransform(self.context)
        else:
            self.predictor = PredictionMRSingleGP(self.context)


    def setup(self, data):
        self.data = data
        self.gprn_structure = False
        super(GPAggr, self).setup(data)
        self._setup_multi_res_variables()
        self._setup_multi_res_1()

    def _setup_multi_res_variables(self):
        self.context.num_latent_process = 1
        self.context.num_likelihood_components = self.data.get_num_sources()

        self.lik_weights = self.parameters.save(name='likelihood_weights', var=np.ones(self.context.num_likelihood_components).astype(np.float32))

        for r in range(1, self.context.num_likelihood_components):
            #different noise per lik component
            sig = self.parameters.create(name='noise_sigma_{r}'.format(r=r), init=self.context.noise_sigmas[r][0], trainable=self.context.noise_sigmas[r][1], scope=Parameters.HYPER_SCOPE)


    def build_elbo_graph(self):
        self.elbo.setup(self.data)
        return self.elbo.build_graph()

