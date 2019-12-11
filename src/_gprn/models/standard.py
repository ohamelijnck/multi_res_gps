import numpy as np
import tensorflow as tf

from . import Model
from . import Base
from .. import util
from ..elbos import StandardELBO
from ..elbos.ell import GPRN_ELL
from ..likelihoods import StandardGPRNLikelihood
from ..likelihoods import CompositeLikelihood
from ..predictions import PredictionStandard
from ..predictions import PredictionStandardGPLogTransform

class Standard(Base):
    def __init__(self,  context):
        super(Standard, self).__init__(context)
        self.context.model = 'Standard'
        self.context = context
        self.context.num_latent_process = 1
        self.context.num_likelihood_components = 1
        self.context.multi_res = False


    def _standard_setup(self):
        self.ell = GPRN_ELL(self.context, r=0)
        self.elbo = StandardELBO(self.context, self.ell)
        self.likelihood = CompositeLikelihood(self.context, [StandardGPRNLikelihood(self.context, r=0)])

        if self.context.log_transform:
            self.predictor = PredictionStandardGPLogTransform(self.context)
        else:
            self.predictor = PredictionStandard(self.context)

    def setup(self, data):
        self.data = data
        self.gprn_structure = True
        super(Standard, self).setup()

        super(Standard, self).setup_variables()

        self.likelihood_weights = self.parameters.save(name='likelihood_weights', var=np.ones(self.context.num_likelihood_components))
        self._standard_setup()


    def get_standard_variational_parameters(self):
        arr = [self.inducing_locations]
        return arr + self.get_base_variational_parameters()

    def get_variational_parameters(self):
        return self.get_standard_variational_parameters()

    def get_free_parameters(self):
        return self.get_standard_free_parameters()

    def get_standard_free_parameters(self):
        return self.get_base_free_parameters()


    def _setup_standard_kernels(self):
        self.setup_base_kernels()

    def fit(self):
        pass
  


