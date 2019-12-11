import numpy as np
import tensorflow as tf

from . import Model
from . import SingleGP
from .. import util
from ..elbos import StandardELBO
from ..elbos import PositiveWELBO
from ..elbos.ell import *
from ..likelihoods import GPRN_Aggr_Likelihood
from ..likelihoods import CompositeLikelihood
from ..predictions import PredictionStandard
from ..predictions import PredictionPositiveW
from ..predictions import PredictionPositiveWLogTransform

class GPRN_PositiveW_Aggr(SingleGP):
    def __init__(self,  context):
        super(GPRN_PositiveW_Aggr, self).__init__(context)
        self.context.model = 'GPRN_PositiveW_Aggr'
        self.context = context

        self.mean_v_scale = -1.0
        self.covar_v_scale = -1.0
        
    def _setup_multi_res_1(self):
        ell_arr = []
        lik_arr = []
        for r in range(self.context.num_likelihood_components):
            ell_arr.append(GPRN_Aggr_Positive_W_ELL(self.context, r=r))
            lik_arr.append(GPRN_Aggr_Likelihood(self.context, r=r))

        self.ell = Composite_ELL(self.context, ell_arr)
        self.likelihood = CompositeLikelihood(self.context, lik_arr)

        self.elbo = PositiveWELBO(self.context, ell=self.ell)

        if self.context.log_transform:
            self.predictor = PredictionPositiveWLogTransform(self.context)
        else:
            self.predictor = PredictionPositiveW(self.context)


    def setup(self, data):
        self.data = data
        self.gprn_structure = True
        super(GPRN_PositiveW_Aggr, self).setup(data)
        self._setup_multi_res_variables()
        self._setup_multi_res_1()

    def _setup_multi_res_variables(self):
        self.context.num_latent_process = 1
        self.context.num_likelihood_components = self.data.get_num_sources()

        self.lik_weights = self.parameters.save(name='likelihood_weights', var=np.ones(self.context.num_likelihood_components).astype(np.float32))

        for r in range(1, self.context.num_likelihood_components):
            #different noise per lik component
            sig = self.parameters.create(name='noise_sigma_{r}'.format(r=r), init=self.context.noise_sigmas[r][0], trainable=self.context.noise_sigmas[r][1])


    def build_elbo_graph(self):
        self.elbo.setup(self.data)
        return self.elbo.build_graph()



