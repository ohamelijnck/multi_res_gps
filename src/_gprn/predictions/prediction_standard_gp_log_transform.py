import tensorflow as tf
import numpy as np
import math
from .. import util
from .. import debugger
from .. sparsity import StandardSparsity
from . import PredictionStandard

from tensorflow.contrib.distributions import MultivariateNormalTriL
from tensorflow.contrib.distributions import MultivariateNormalDiag 

class PredictionStandardGPLogTransform(PredictionStandard):
    def build_sample_graph(self, r=0):
         sample = self.build_sample_standard()
         return util.safe_exp(sample)

 
