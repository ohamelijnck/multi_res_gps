import tensorflow as tf
import numpy as np
import math
from . import ELBO
from .. import util
from .. import debugger
from .. sparsity import StandardSparsity,MRSparsity

class StandardELBO(ELBO):
    def __init__(self, context, ell):
        self.context = context
        self.ell = ell

    def setup_standard(self):
        self.num_latent_process = self.context.num_latent_process
        self.num_train = self.data.get_num_training(source=0)
        self.batch_size = self.data.get_batch_size(source=0)
        self.num_latent = self.context.num_latent
        self.num_outputs = self.context.num_outputs
        self.num_weights = self.context.num_weights
        self.num_inducing = self.data.get_num_inducing(source=0)
        self.use_diag_covar_flag = self.context.use_diag_covar_flag
        self.jitter=self.context.jitter

        #=====Components for q(u,v) as an MoG
        self.q_num_components = self.context.num_components
        self.num_sigma = self.num_inducing*(self.num_inducing+1)/2
        
        self.parameters = self.context.parameters
        self.get_standard_variables()
        
        if self.context.multi_res:
            self.sparsity = MRSparsity(self.data, self.context)
        else:
            self.sparsity = StandardSparsity(self.data, self.context)



    def setup(self, data):
        self.data = data
        self.setup_standard()
        self.ell.setup(self)

    def get_f_w(self, k=0, p=0):
        _mu_f, _sigma_f, _mu_w, _sigma_w = self.sparsity._build_intermediate_conditionals(k, self.x_train)

        mu_f = _mu_f[:, :, 0]
        mu_wi = _mu_w[p,:,:,0]
        sigma_f = tf.matrix_diag_part(_sigma_f)
        sigma_wi = tf.matrix_diag_part(_sigma_w[p,:,:,:])
        return mu_f, sigma_f, mu_wi, sigma_wi

        
    def get_standard_variables(self):
        self.q_means_u_arr = []
        self.q_covars_u_arr = []
        self.q_means_v_arr = []
        self.q_covars_v_arr = []

        self.q_chol_covars_arr = []
        self.inducing_locations_arr = []

        for r in range(self.num_latent_process):
            self.q_means_u_arr.append(self.parameters.get(name='q_means_u_{r}'.format(r=r)))
            self.q_means_v_arr.append(self.parameters.get(name='q_means_v_{r}'.format(r=r)))
            self.q_covars_u_arr.append(self.parameters.get(name='q_covars_u_{r}'.format(r=r)))
            self.q_covars_v_arr.append(self.parameters.get(name='q_covars_v_{r}'.format(r=r)))

            self.inducing_locations_arr.append(self.parameters.get(name='inducing_locations_{r}'.format(r=r)))

        self.q_weights = self.parameters.get(name='q_weights')

    def build_graph(self):
        entropy = self._build_entropy()
        cross_entropy = self._build_cross_entropy()
        expected_log_likelhood = self._build_ell()
        
        dummy = 0.0
        dummy = debugger.debug_inference(self, dummy, entropy, cross_entropy, expected_log_likelhood)

        dummy = tf.Print(dummy, [-([expected_log_likelhood] + (cross_entropy - entropy))], 'ELBO: ')
        #elbo = dummy*0.0 + [expected_log_likelhood] + (cross_entropy - entropy)
        elbo = dummy*0.0 + [expected_log_likelhood] + (cross_entropy - entropy)

        #elbo = dummy*0.0 + [expected_log_likelhood]

        return  elbo

    def _build_entropy_sum(self, r, m1, s1, m2, s2, same_flag=False, z_num=None):
        if z_num is None: z_num = self.data.get_num_inducing(source=r)
        #covar_sum = tf.cholesky(s1+s2+self.jitter*tf.eye(z_num))
        #covar_sum = tf.cast(tf.cholesky(tf.cast(util.add_jitter(s1+s2, self.context.jitter), tf.float64)), tf.float32)

        covar_sum = tf.cast(tf.cholesky(tf.cast(util.add_jitter(s1+s2, 1e-6), tf.float64)), tf.float32)

       # p = util.log_normal_chol(x=tf.expand_dims(m1, -1), mu=tf.expand_dims(m2, -1), chol=covar_sum, n=z_num)
        



        p = util.log_normal_chol(x=tf.expand_dims(m1, -1), mu=tf.expand_dims(m2, -1), chol=covar_sum, n=z_num)

        return p

    def _build_l_sum(self, k, r):
        l_sum = [0.0 for i in range(self.q_num_components)]
        for l in range(self.q_num_components):
            pi_l = self.q_weights[l]
            u_sum = 0.0
            v_sum = 0.0

            for j in range(self.num_latent):
                m_f_lj, s_f_lj = self.q_means_u_arr[r][l,j,:], self.q_covars_u_arr[r][l,j,:,:]
                m_f_kj, s_f_kj = self.q_means_u_arr[r][k,j,:], self.q_covars_u_arr[r][k,j,:,:]
                
                p = self._build_entropy_sum(r, m_f_lj, s_f_lj, m_f_kj, s_f_kj,k==l)


                u_sum = u_sum+p

                for i in range(self.num_outputs):
                    m_w_lij, s_w_lij = self.q_means_v_arr[r][l,i,j,:], self.q_covars_v_arr[r][l,i,j,:,:]
                    m_w_kij, s_w_kij = self.q_means_v_arr[r][k,i,j,:], self.q_covars_v_arr[r][k,i,j,:,:]

                    p = self._build_entropy_sum(r, m_w_lij, s_w_lij, m_w_kij, s_w_kij,k==l)

                    v_sum = v_sum+p

            l_sum[l] =  util.safe_log(pi_l) + u_sum + v_sum
        return l_sum

    def _build_entropy(self):
        print('entropy')
        total_sum = 0.0
        for k in range(self.q_num_components):
            pi_k = self.q_weights[k]
            l_sum = []
            for r in range(self.num_latent_process):
                l_sum = l_sum + self._build_l_sum(k, r)
            total_sum += pi_k * tf.reduce_logsumexp(l_sum)

        return total_sum

  
    def _build_cross_entropy_sum(self, k1, m1, s1, n,debug=False):
        k_chol = tf.cholesky(k1)
        m1 = tf.expand_dims(m1, 1)

        d = tf.trace(tf.cholesky_solve(k_chol, s1))
        p = util.log_normal_chol(x=0.0, mu=m1, chol=k_chol,n=n)

        result =  p-0.5*d
        return result

    def _build_cross_entropy(self):
        print('cross entropy')
        total_sum = 0.0

        for k in range(self.q_num_components):
            pi_k = self.q_weights[k]
            u_sum = 0.0
            v_sum = 0.0
            for r in range(self.num_latent_process):
                kern_f = self.context.kernels[r]['f']
                kern_w = self.context.kernels[r]['w']
                num_inducing = self.data.get_num_inducing(source=r)
                for j in range(self.num_latent):
                    if self.context.whiten:
                        k_j = tf.eye(num_inducing)
                    else:
                        k_j = kern_f[j].kernel(self.inducing_locations_arr[r], self.inducing_locations_arr[r], jitter=True)
                    u_sum += self._build_cross_entropy_sum(k_j, self.q_means_u_arr[r][k,j,:], self.q_covars_u_arr[r][k,j,:, :], self.num_inducing, debug=True)

                    for i in range(self.num_outputs):
                        if self.context.whiten:
                            k_ij = tf.eye(num_inducing)
                        else:
                            k_ij = kern_w[i][j].kernel(self.inducing_locations_arr[r], self.inducing_locations_arr[r], jitter=True)
                        v_sum += self._build_cross_entropy_sum(k_ij, self.q_means_v_arr[r][k,i,j,:], self.q_covars_v_arr[r][k,i,j,:,:], self.num_inducing, debug=True)

            total_sum = total_sum + pi_k * (u_sum+v_sum)

        return total_sum

    def _build_ell(self):
        return self.ell._build_ell()
