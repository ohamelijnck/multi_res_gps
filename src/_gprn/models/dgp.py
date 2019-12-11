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
from .. import Parameters

class DGP(SingleGP):
    def __init__(self,  context):
        super(DGP, self).__init__(context)
        self.context.model = 'DGP'
        self.context = context
        self.context.num_latent_process = 2
        self.gprn_structure = False



        
    def _setup_multi_res_1(self):
        #ell_arr = [GP_ELL(self.context, r=0), GP_Aggr_ELL(self.context, r=1, a=1)]
        ell_arr = [GP_ELL(self.context, r=0), GP_ELL(self.context, r=1, a=1)]
        #ell_arr = [GP_Aggr_ELL(self.context, r=0, a=0), GP_ELL(self.context, r=0, a=1)]
        #ell_arr = [GP_Aggr_ELL(self.context, r=0, a=0), GP_Aggr_ELL(self.context, r=1, a=1)]
        #ell_arr = [GP_ELL(self.context, r=0), GP_ELL(self.context, r=0)]
        #ell_arr = [GP_Aggr_ELL(self.context, r=1)]

        self.ell = Composite_ELL(self.context, ell_arr)

        self.elbo = SingleGP_ELBO(self.context, ell=self.ell)

        #lik_arr = [SingleGPLikelihood(self.context, r=0), GP_Aggr_Likelihood(self.context, r=1)]
        lik_arr = [SingleGPLikelihood(self.context, r=0), SingleGPLikelihood(self.context, r=1)]

        self.likelihood = CompositeLikelihood(self.context, lik_arr)
        self.predictor = PredictionSingleGP(self.context)


    def setup(self, data):
        self.context.num_likelihood_components = data.get_num_sources()
        super(DGP, self).setup(data)

        self._setup_multi_res_variables()
        self._setup_multi_res_1()

    
    def _setup_multi_res_variables(self):
        self._setup_base_variables(a=1)
        #self.parameters.create(name='noise_sigma_{a}'.format(a=1), init=self.context.noise_sigmas[0][0], trainable=self.context.noise_sigmas[0][1], scope=Parameters.HYPER_SCOPE)


    def build_elbo_graph(self):
        self.elbo.setup(self.data)
        return self.elbo.build_graph()


