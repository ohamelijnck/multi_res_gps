import numpy as np
import tensorflow as tf

from . import ELL
from ... import util

class Composite_ELL(ELL):
    def __init__(self, context, ell_arr):
        self.context = context
        self.ell_arr = ell_arr
        self.num_lik_components = len(self.ell_arr)
        self.parameters = self.context.parameters



    def setup(self, elbo):
        self.elbo = elbo
        self.data = self.elbo.data
        self.likelihood_weights = self.parameters.get(name='likelihood_weights')
        for ell in self.ell_arr:
            ell.setup(elbo)



    def _build_ell(self):
        print('BUILD COMPOSITE ELL')
        total = 0
        i = 0

        for ell in self.ell_arr:
            r = i
            #scale by composite weight
            _total = self.likelihood_weights[i]*ell._build_ell()
            _total = tf.Print(_total, [self.likelihood_weights[i]], 'ell_weight_{r}: '.format(r=i))
            total = total + _total
            i += 1
        return total

