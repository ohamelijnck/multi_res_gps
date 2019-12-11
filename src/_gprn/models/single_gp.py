import numpy as np
import tensorflow as tf

from . import Model
from . import Base
from .. import util
from ..elbos import SingleGP_ELBO
from ..elbos.ell import *
from ..likelihoods import SingleGPLikelihood
from ..likelihoods import CompositeLikelihood
from ..predictions import PredictionSingleGP
from ..predictions import PredictionSingleGPLogTransform

class SingleGP(Base):
    def __init__(self,  context):
        super(SingleGP, self).__init__(context)
        self.context.model = 'SingleGP'
        self.context = context
        self.context.num_likelihood_components= 1
        self.context.num_latent_process = 1
        self.gprn_structure=False
       
    def setup(self, data):
        self.data = data
        super(SingleGP, self).setup()

        self.setup_variables()

    def setup_variables(self):
        super(SingleGP, self).setup_variables()

        self.likelihood_weights = self.parameters.save(name='likelihood_weights', var=np.ones(self.context.num_likelihood_components))

    def build_elbo_graph(self):
        r = 0
        ell_arr = [GP_ELL(self.context, r=r)]
        self.ell = Composite_ELL(self.context, ell_arr)
        self.elbo = SingleGP_ELBO(self.context, self.ell)
        self.likelihood = CompositeLikelihood(self.context, [SingleGPLikelihood(self.context, r=r)])

        if self.context.log_transform:
            self.predictor = PredictionSingleGPLogTransform(self.context)
        else:
            self.predictor = PredictionSingleGP(self.context)

        self.elbo.setup(self.data)
        return self.elbo.build_graph()

    def _setup_standard_kernels(self):
        self.setup_base_kernels()

    def fit(self):
        pass
  



