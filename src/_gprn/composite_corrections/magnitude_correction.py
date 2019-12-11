from . import CompositeCorrections
from .. import Parameters
from ..scores import *

import numpy as np
import scipy as sp
import tensorflow as tf
class MagnitudeCorrection(CompositeCorrections, Parameters):
    def __init__(self,context):
        Parameters.__init__(self, context)
        CompositeCorrections.__init__(self, context)
        self.context = context
        self.J = None
        self.H = None

    def get_k(self):
        print('apply_correction: get_k')
        print('H: ', self.H)
        print('H: ', self.H.shape)
        print('J: ', self.J)
        print('J: ', self.J.shape)
        N = self.J.shape[0]
        print('apply_correction: get_k: N ', N)

        H = self.H
        J = self.J

        H = H.astype(np.float64)
        J = J.astype(np.float64)

        if N == 1:
            k =  N/(np.trace(self.J/self.H))
        if N ==2:
            H_inv = (1/(H[1][1]*H[0][0] -H[0][1]*H[1][0] )) * np.array([[H[1][1], -H[0][1]], [-H[1][0], H[0][0]]])
            k = N/np.trace(H_inv @ J)
        else:
            jitter = 1e-4*np.eye(self.H.shape[0])
            #jitter = 0.0


            #H_chol = np.linalg.cholesky(self.H+jitter)

            #H_inv = (1/(H[1][1]*H[0][0] -H[0][1]*H[1][0] )) * np.array([[H[1][1], -H[0][1]], [-H[1][0], H[0][0]]])

            #print('HJ:', sp.linalg.solve_triangular(H_chol, sp.linalg.solve_triangular(H_chol,self.J, lower=True), lower=False))
            #print('H-1H:', sp.linalg.solve_triangular(H_chol, sp.linalg.solve_triangular(H_chol,self.H, lower=True), lower=False))
            #print('H-1H 2:', N/np.trace(H_inv @ self.J))

            #k = N/(np.trace(sp.linalg.solve_triangular(H_chol, sp.linalg.solve_triangular(H_chol,self.J, lower=True), lower=False)))
            k = N/(np.trace(sp.linalg.solve(H, J)))
        return k.astype(np.float32)

    def get_k_1(self):
        print('apply_correction: get_k_1')
        jitter = 1e-4*np.eye(self.H.shape[0])
        J_chol = np.linalg.cholesky(self.J+jitter)
        cho_sol = lambda a, b: sp.linalg.solve_triangular(a, sp.linalg.solve_triangular(a,b, lower=True), lower=False)
        cho_solve = lambda a, b: sp.linalg.solve_triangular(a,b, lower=True)
        
        #k = np.trace(np.matmul(self.H, np.linalg.solve(self.J+jitter,self.H.T+jitter)))/np.trace(self.H)

        print(np.trace(np.matmul(self.H, cho_sol(J_chol, self.H))))
        aa = cho_solve(J_chol, self.H)
        print(np.trace(np.matmul(aa.T, aa)))
        print(np.trace(np.multiply(aa.T, aa)))
        k = np.trace(np.matmul(aa.T, aa))/np.trace(self.H)
        return k


    def apply_correction(self):
        self.H= np.squeeze(self.H)
        self.J= np.squeeze(self.J)

        print(self.H.shape)
        if len(self.H.shape)==0:
            self.H= np.expand_dims(np.expand_dims(self.H,-1), -1)
            self.J= np.expand_dims(np.expand_dims(self.J,-1), -1)

        print('apply_correction')
        k = self.get_k()
        print('k: ', k)
        print('k1: ', self.get_k_1())
        weights = k*tf.ones(self.context.num_likelihood_components)
        self.context.parameters.save(name='likelihood_weights', var=weights)

    def learn_information_matrices(self, session, optimise_flag = True):
        print('learn_information_matrices')
        self.estimate_information_matrices(optimise_flag)
        print('get_params')
        self.theta_hat = self.get_params(session)




