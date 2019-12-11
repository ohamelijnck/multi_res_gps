from ..scores import *

import tensorflow as tf
import numpy as np

class CL_Corrections(object):
    def __init__(self, model, context):
        self.model = model
        self.context = context
        self.parameters = self.context.parameters
        self.J = None
        self.H = None

    def estimate_information_matrices(self, optimise_flag = False):
        self.model.optimise(optimise_flag, not optimise_flag)

        #param_arr = self.model.get_free_parameters()
        #param_arr = self.model.get_variational_parameters()
        param_arr = [self.parameters.get('se_length_scale_1'), self.parameters.get('se_length_scale_0'), self.parameters.get('q_means_u_0')]
        score = FisherInformation(self.model)
        fisher_s = score.observed_matrix(param_arr)
        self.J = fisher_s

        score = LikelihoodHessian(self.model)
        self.H = score.observed_matrix(param_arr)

        tf.reset_default_graph()


    def magnitude_correction(self):
        N = self.J.shape[0]
        eigen_values = np.linalg.eigvals(np.linalg.solve(self.H,self.J))
        print('H_inv', np.linalg.inv(self.H))
        print('J_inv', np.linalg.inv(self.J))
        k = N/(np.trace(np.linalg.solve(self.H+0.00001*np.eye(self.H.shape[0]),self.J)))
        return k

    
    def curvature_correction(self):
        M = np.linalg.cholesky(self.H+0.001*np.eye(self.H.shape[0]))
        Ma = np.linalg.cholesky(np.matmul(self.H,np.linalg.solve(self.J, self.H))+0.001*np.eye(self.H.shape[0]))
        C = np.linalg.solve(M, Ma)
        return C
