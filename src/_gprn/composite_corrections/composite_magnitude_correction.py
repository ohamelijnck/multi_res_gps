from . import CompositeCorrections
from .. import Parameters
from ..scores import *
from scipy.optimize import nnls

import numpy as np
import tensorflow as tf
class CompositeMagnitudeCorrection(CompositeCorrections, Parameters):
    def __init__(self,context):
        Parameters.__init__(self, context)
        CompositeCorrections.__init__(self, context)
        self.context = context
        self.J = None
        self.H = None
        self.H_parts = None

    def get_composite_weights(self, J, H, H_parts):
        R = self.model.context.num_likelihood_components

        num_params =  self.J.shape[0]
        jitter = 0.000001*np.eye(self.H.shape[0])

        A = lambda r: np.trace(np.matmul(np.linalg.solve(H+jitter,H_parts[r]+jitter), np.linalg.solve(H+jitter, J+jitter)))

        mat = []
        vec = []
        for i in range(R):
            row = []
            vec.append(num_params)
            for j in range(R):
                if i == j:
                    t = A(j)
                else:
                    t = 0
                row.append(t)
            mat.append(row)

        mat = np.array(mat)
        vec = np.array(vec)

        print(nnls.__code__.co_varnames)

        w= nnls(mat, vec)
        
        print(w)

        N = self.J.shape[0]
        k = N/(np.trace(np.linalg.solve(H+jitter,J+jitter)))
        print('k', k)

        return w

    def _get_composite_weights(self, J, H, H_parts):
        R = self.model.context.num_likelihood_components

        P =  self.J.shape[0]
        jitter = 0.000001*np.eye(self.H.shape[0])

        print(R)
        A = [np.trace(np.matmul(np.linalg.solve(H+jitter,H_parts[r]+jitter), np.linalg.solve(H+jitter, J+jitter))) for r in range(R)]
        print('A: ', A)
        B = [np.trace(H_parts[i]) for i in range(R)]
        print('B: ', B)
        c = np.trace(np.matmul(H, np.linalg.solve(J+jitter, H+jitter)))
        print('c: ', c)

        k1 = lambda i: (1/(R*R))*np.sum([B[r]*B[i]/(2*A[i]*A[i]) for r in range(R)])
        K1 = [k1(i) for i in range(R)]
        print('K1: ', K1)


        k2 = lambda i: (-1/(R*K1[i]))*np.sum([B[r]*P/A[i] for r in range(R)])
        K2 = [k2(i) for i in range(R)]
        print('K2: ', K2)

        k3 = lambda i: 2*P*A[i] - (B[i]/R)*(-K2[i]+c)
        K3 = [k3(i) for i in range(R)]
        print('K3: ', K3)

        d = lambda i : 2*A[i]*A[i]
        D = [d(i) for i in range(R)]
        print('D: ', D)

        x = lambda i, j : 2*A[i]*A[j] + (B[i]/(R*R*K1[1]*2*A[i]*A[i]))*np.sum([B[r]*2*A[i]*A[j] for r in range(R)])

        mat = []
        vec = []
        for i in range(R):
            row = []
            vec.append(K3[i])
            for j in range(R):
                if i == j:
                    t = D[i]
                else:
                    t = x(i, j)
                row.append(t)
            mat.append(row)


        mat = np.array(mat)
        vec = np.array(vec)

        print('mat', mat)
        print('mat', np.linalg.det(mat))
        print('vec', vec)
        w = np.sqrt(np.abs(np.linalg.solve(mat, vec)))

        N = self.J.shape[0]
        k = N/(np.trace(np.linalg.solve(H+jitter,J+jitter)))
        print('k', k)
        print('w', np.linalg.solve(mat, vec))


        



    def tmp(self):
        A = lambda r: np.trace(np.matmul(np.linalg.solve(H+jitter,H_parts[r]+jitter), np.linalg.solve(H+jitter, J+jitter)))

        mat = []
        vec = []
        for i in range(R):
            row = []
            vec.append(num_params)
            for j in range(R):
                if i == j:
                    t = A(j)
                else:
                    t = A(j)
                row.append(t)
            mat.append(row)


        mat = np.array(mat)
        vec = np.array(vec)

        print('vec', vec)
        w = np.sqrt(np.abs(np.linalg.solve(mat, vec)))

        N = self.J.shape[0]
        k = N/(np.trace(np.linalg.solve(H+jitter,J+jitter)))
        print('k', k)


        print('mat', mat)
        print('mat', np.linalg.det(mat))
        print('vec', vec)
        w = np.sqrt(np.abs(np.linalg.solve(mat, vec)))

        N = self.J.shape[0]
        k = N/(np.trace(np.linalg.solve(H+jitter,J+jitter)))
        print('k', k)

        return w

    def apply_correction(self):
        weights = self.get_composite_weights(self.J, self.H, self.H_parts)

        self.save(name='likelihood_weights', var=weights)


    def learn_information_matrices(self, session):
        self.estimate_information_matrices(True)
        self.theta_hat = self.get_params(session)




