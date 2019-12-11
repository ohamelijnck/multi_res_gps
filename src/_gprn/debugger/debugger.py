import numpy as np
import tensorflow as tf
import math
from .. import util

def debug_inference(inference, dummy, entropy, cross_entropy, expected_log_likelhood):
    dummy = tf.Print(dummy, [entropy], 'entropy: ')
    dummy = tf.Print(dummy, [cross_entropy], 'cross_entropy: ')
    dummy = tf.Print(dummy, [expected_log_likelhood], 'expected_log_likelhood: ')
    #dummy = tf.Print(dummy, [inference.q_means_u], 'self.q_means_u: ')
    #dummy = tf.Print(dummy, [inference.q_covars_u], 'self.q_covars_u: ')
    #dummy = tf.Print(dummy, [inference.q_means_v], 'self.q_means_v: ')
    #dummy = tf.Print(dummy, [inference.q_covars_v], 'self.q_covars_v: ')

    return dummy

def matrix_conditions(session, inference):
    for j in range(inference.num_latent):
        k_j = inference.kern_f[j]
        K_zz_f = k_j.kernel(inference.inducing_locations, inference.inducing_locations, jitter=True)
        mat = K_zz_f.eval(session=session)
        cond = np.linalg.cond(mat)
        sigma = k_j.sigma.eval(session=session)
        ls = k_j.length_scales.eval(session=session)
        print('MATRIX CONDITION F('+str(j)+'): ', cond)
        print('SIGMA F('+str(j)+'): ', sigma)
        print('LENGTH_SCALES F('+str(j)+'): ', ls)

        print(mat)

    for j in range(inference.num_latent):
        for i in range(inference.num_outputs):
            k_ij = inference.kern_w[i][j]
            K_zz_w = k_ij.kernel(inference.inducing_locations, inference.inducing_locations, jitter=True)
            mat = K_zz_w.eval(session=session)
            cond = np.linalg.cond(mat)
            sigma = k_ij.sigma.eval(session=session)
            ls = k_ij.length_scales.eval(session=session)
            print('MATRIX CONDITION W('+str(i)+','+str(j)+'): ', cond)
            print('SIGMA W('+str(i)+','+str(j)+'): ', sigma)
            print('LENGTH_SCALES W('+str(i)+','+str(j)+'): ', ls)
            print(mat)

