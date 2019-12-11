from .. import CompositeCorrections
from . import Likelihood
import tensorflow as tf
import numpy as np
class CompositeLikelihood(Likelihood):
    def __init__(self, context, lik_arr):
        print('CLIINIT')
        self.context = context
        self.lik_arr = lik_arr
        self.parameters = self.context.parameters

    def setup(self, data):
        self.data = data
        print('cl')
        print(self.data)
        self.likelihood_weights = self.parameters.get(name='likelihood_weights')
        for lik in self.lik_arr:
            lik.setup(self.data)
        

    def build_graph(self):
        print('BUILD COMPOSITE ELL')

        total_sum = 0.0
        i = 0
        p=0
        for lik in self.lik_arr:
            r = i
            #total_sum += 0.5*(total_n/n)*self.likelihood_weights[i]*lik._build_log_likelihood()
            #total_sum += (n/batch_size)*self.likelihood_weights[i]*lik._build_log_likelihood()
            total_sum += self.likelihood_weights[i]*lik._build_log_likelihood()
            
            i += 1
        return total_sum

