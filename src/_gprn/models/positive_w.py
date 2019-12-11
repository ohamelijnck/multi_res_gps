import numpy as np
import tensorflow as tf

from . import Model
from . import Base
from .. import util
from ..elbos import StandardELBO
from ..elbos.ell import GPRN_Positive_W_ELL
from ..predictions import PredictionPositiveW
from ..likelihoods import CompositeLikelihood
from ..likelihoods import StandardGPRNLikelihood

class PositiveW(Base):
    def __init__(self,  context):
        super(PositiveW, self).__init__(context)
        self.context.model = 'PositiveW'
        self.context = context
        self.context.num_latent_process = 1
        self.context.num_likelihood_components = 1

        self.mean_v_scale = -0.1
        self.covar_v_scale = -0.01

    def _setup_postive_w_variables(self):
        self._setup_base_variables()
        self.inducing_locations = self.parameters.create(name='inducing_locations_0', init=self.data.get_inducing_points_from_source(0).astype(np.float32), trainable=self.train_inducing_points_flag)

    def _positive_w_setup(self):
        super(PositiveW, self).setup()

        self.predictor = PredictionPositiveW(self.context)
        self.likelihood = CompositeLikelihood(self.context, [StandardGPRNLikelihood(self.context, r=0)])
        self.ell = GPRN_Positive_W_ELL(self.context, r=0)
        self.elbo = StandardELBO(self.context, self.ell)


    def setup(self, data):
        self.data = data
        self.gprn_structure = True
        super(PositiveW, self).setup_variables()
        #self._setup_postive_w_variables()
        self.likelihood_weights = self.parameters.save(name='likelihood_weights', var=np.ones(self.context.num_likelihood_components))
        self._positive_w_setup()

    def fit(self):
        pass






