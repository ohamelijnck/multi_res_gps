from . import CompositeCorrections
from .. import Parameters
from ..scores import *

import numpy as np
import tensorflow as tf
class NoCorrection(CompositeCorrections, Parameters):
    def __init__(self,context):
        Parameters.__init__(self, context)
        CompositeCorrections.__init__(self, context)
        self.context = context

    def apply_correction(self):
        pass

    def learn_information_matrices(self, session):
        pass





