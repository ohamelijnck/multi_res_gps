import tensorflow as tf
import numpy as np
import math
from . import ELBO
from .. import util
from .. import debugger
from .. sparsity import StandardSparsity, MRSparsity
from .ell import GP_ELL

class SingleGP_ELBO(ELBO):
    def __init__(self, context, ell):
        self.context = context
        self.parameters = self.context.parameters
        self.ell = ell

    def setup_standard(self):
        self.num_latent_process = self.context.num_latent_process
        self.num_train = self.data.get_num_training(source=0)
        self.batch_size = self.data.get_batch_size(source=0)
        self.num_latent = self.context.num_latent
        self.num_outputs = self.context.num_outputs
        self.num_weights = self.context.num_weights
        self.use_diag_covar_flag = self.context.use_diag_covar_flag
        self.jitter=self.context.jitter

        #=====Components for q(u,v) as an MoG
        self.q_num_components = self.context.num_components
        
        self.get_standard_variables()
        #self.sparsity = StandardSparsity(self.data, self.context)
        self.sparsity = MRSparsity(self.data, self.context)
        #self.sparsity = DGPSparsity(self.data, self.context)

    def setup(self, data):
        self.data = data
        self.setup_standard()
        self.ell.setup(self)

    def get_posterior(self, r=0):
        total_sum = 0
        total_mu = 0
        total_var = 0

        x_train = self.data.get_placeholder(source=0, var='x')
        x_train = self.parameters.get(name='inducing_locations_{r}'.format(r=r))

        for k in range(self.q_num_components):
            pi_k = self.q_weights[k]
            print('x_train: ', x_train)
            x_train = tf.Print(x_train, [tf.shape(x_train)], 'x_train: ')
            _mu_f, _sigma_f, _, _ = self.sparsity._build_intermediate_conditionals(k, r, x_train)
            print('_mu_f: ', _mu_f)
            _mu_f = tf.Print(_mu_f, [tf.shape(_mu_f)], '_mu_f: ')
            total_mu = total_mu + pi_k*_mu_f
            total_var = total_var + pi_k*_sigma_f
        return total_mu, total_var, total_mu, total_var
        
    def get_standard_variables(self):
        self.q_means_arr = []
        self.q_covars_arr = []
        self.q_chol_covars_arr = []
        self.inducing_locations_arr = []

        self.q_covar_raw = self.parameters.get(name='q_covars_u_0_raw')

        for r in range(self.num_latent_process):
            self.q_means_arr.append(self.parameters.get(name='q_means_u_{r}'.format(r=r)))
            self.inducing_locations_arr.append(self.parameters.get(name='inducing_locations_{r}'.format(r=r)))
            q_covars_u = self.parameters.get(name='q_covars_u_{r}'.format(r=r))

            self.q_covars_arr.append(self.parameters.get(name='q_covars_u_{r}'.format(r=r)))
            self.q_chol_covars_arr.append(self.parameters.get(name='q_cholesky_u_{r}'.format(r=r)))

        self.q_weights = self.parameters.get(name='q_weights')

    def build_graph(self):
        expected_log_likelhood = self._build_ell()
        entropy = self._build_entropy()
        cross_entropy = self._build_cross_entropy()
        
        dummy = 0.0
        dummy = debugger.debug_inference(self, dummy, entropy, cross_entropy, expected_log_likelhood)

        total_batch_size = sum([self.data.get_batch_size(source=i) for i in range(self.context.num_likelihood_components)])
        total_data = sum([self.data.get_num_training(source=i) for i in range(self.context.num_likelihood_components)])
        total_inducing = sum([self.data.get_inducing_points_from_source(source=i).shape[0] for i in range(self.context.num_latent_process)])


        kl = self._build_kl()

        expected_log_likelhood = tf.Print(expected_log_likelhood, [total_batch_size], 'total_batch_size: ')
        expected_log_likelhood = tf.Print(expected_log_likelhood, [total_data], 'total_data: ')
        expected_log_likelhood = tf.Print(expected_log_likelhood, [total_inducing], 'total_inducing: ')
        expected_log_likelhood = tf.Print(expected_log_likelhood, [cross_entropy-entropy], 'cross-ent: ')
        expected_log_likelhood = tf.Print(expected_log_likelhood, [kl], 'kl: ')
        dummy = tf.Print(dummy, [-([expected_log_likelhood] + (cross_entropy - entropy))], 'ELBO: ')



        #elbo = dummy*0.0 + expected_log_likelhood + (self.batch_size/self.data.get_num_inducing(source=0))*(cross_entropy - entropy)
        #elbo = dummy*0.0 + *expected_log_likelhood + (cross_entropy - entropy)
        #elbo = dummy*0.0 + (total_inducing/total_data)*expected_log_likelhood + (cross_entropy - entropy)
        #elbo = dummy*0.0 + [expected_log_likelhood] + (cross_entropy - entropy)
        elbo = dummy*0.0 + [expected_log_likelhood] + kl
        #elbo = dummy*0.0 + [expected_log_likelhood] + kl
        #elbo = dummy*0.0 + [expected_log_likelhood]
        #elbo = tf.expand_dims(cross_entropy, -1)
        #elbo = dummy*0.0 + expected_log_likelhood + (cross_entropy - entropy)
        #elbo = dummy*0.0 + expected_log_likelhood + kl
        #elbo = dummy*0.0 + expected_log_likelhood + kl
        #elbo = dummy*0.0 + expected_log_likelhood + kl
        #elbo = dummy*0.0 + expected_log_likelhood + (cross_entropy - entropy)

        return  elbo

    def _build_entropy_sum(self, r, m1, s1, m2, s2, same_flag=False, z_num=None):
        if z_num is None: z_num = self.data.get_num_inducing(source=r)

        if same_flag:
            covar_sum = np.sqrt(2)*self.q_chol_covars_arr[0][0, 0, :, :]
        else:
            covar_sum = tf.cast(tf.cholesky(tf.cast(util.add_jitter(s1+s2, 1e-6), tf.float64)), tf.float32)

        p = util.log_normal_chol(x=tf.expand_dims(m1, -1), mu=tf.expand_dims(m2, -1), chol=covar_sum, n=z_num)
        

        return p

    def _build_l_sum(self, k, r):
        l_sum = [0.0 for i in range(self.q_num_components)]
        for l in range(self.q_num_components):
            pi_l = self.q_weights[l]
            u_sum = 0.0

            j = 0

            m_f_lj, s_f_lj = self.q_means_arr[r][l,j,:], self.q_covars_arr[r][l,j,:,:]
            m_f_kj, s_f_kj = self.q_means_arr[r][k,j,:], self.q_covars_arr[r][k,j,:,:]
            
            p = self._build_entropy_sum(r, m_f_lj, s_f_lj, m_f_kj, s_f_kj, k==j)

            u_sum = u_sum+p

            l_sum[l] =  util.safe_log(pi_l) + u_sum 

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
        num_inducing = self.data.get_num_inducing(source=0)
        k_chol = tf.cast(tf.cholesky(tf.cast(k1, tf.float64)), tf.float32)

        m1 = tf.expand_dims(m1, 1)

        d = tf.trace(util.tri_mat_solve(tf.transpose(k_chol), util.tri_mat_solve(k_chol, s1, lower=True), lower=False))
        p = util.log_normal_chol(x=0.0, mu=m1, chol=k_chol, n=n)

        result =  p-0.5*d

        return result

    def _build_cross_entropy(self):
        print('cross entropy')
        total_sum = 0.0

        for k in range(self.q_num_components):
            pi_k = self.q_weights[k]
            pi_k = tf.Print(pi_k, [pi_k], 'pi_k: ')
            u_sum = 0.0
            v_sum = 0.0
            j = 0
            i = 0

            for r in range(self.num_latent_process):
                kern_f = self.context.kernels[r]['f'] #when using a SingleGP the kernels will be defined in r=0
                num_inducing = self.data.get_num_inducing(source=r)

                if self.context.whiten:
                    k_j = tf.eye(num_inducing)
                else:
                    Z = self.inducing_locations_arr[r]
                    if self.context.multi_res:
                        Z = tf.expand_dims(Z, 0)

                    k_j = kern_f[j]._kernel(Z, Z, jitter=True)

                    if self.context.multi_res:
                        k_j = k_j[0, :, :] # M x M

                u_sum += self._build_cross_entropy_sum(k_j, self.q_means_arr[r][k,j,:], self.q_covars_arr[r][k,j,:, :], num_inducing)

                total_sum = total_sum + pi_k * (u_sum)

        return total_sum

    def _build_kl(self):
        total_sum = 0.0

        for r in range(self.num_latent_process):
            z_num = self.data.get_num_inducing(source=r)
            m1 =  tf.expand_dims(self.q_means_arr[r][0,0,:], -1)
            s1 =  self.q_covars_arr[r][0, 0, :, :]
            s1_chol =  self.q_chol_covars_arr[r][0, 0, :, :]

            Z = self.inducing_locations_arr[0]
            if self.context.multi_res:
                Z = tf.expand_dims(Z, 0)

            s2 =  self.context.kernels[r]['f'][0].kernel(Z, Z, jitter=True)

            if self.context.multi_res:
                s2 = s2[0, :, :] # M x M

            s2_chol =  tf.cholesky(s2)

            if self.context.whiten:
                #mahal = tf.matmul(m1,  m1, transpose_a=True)
                mahal = tf.reduce_sum(tf.square(m1))
            else:
                mahal = tf.matmul(m1, util.tri_mat_solve(tf.transpose(s2_chol), util.tri_mat_solve(s2_chol, m1, lower=True), lower=False), transpose_a=True)

            log_det =util.log_chol_matrix_det(s1_chol)

            if not self.context.whiten:
                log_det =  log_det-util.log_chol_matrix_det(tf.cholesky(s2))

            if self.context.whiten:
                tr_term = tf.trace(s1)
            else:
                tr_term = tf.trace(util.mat_solve(s2, s1))

            #total_sum =  total_sum + -0.5*(mahal - z_num - log_det + tr_term)
            total_sum =  total_sum + 0.5*(log_det + z_num - mahal - tr_term)
        return total_sum

    def _build_ell(self):
        return self.ell._build_ell()


