class MR_Conditional(object):
    def _build_marginal(self, m, s_chol, k_zz,  k_xz, k_xx, predict=True):
        k_zz_chol = tf.cholesky(tf.cast(k_zz, tf.float64)) # N x M x M
            
        if self.context.whiten:
            mu = tf.matmul(k_xz, util.tri_mat_solve(tf.transpose(k_zz_chol, [0, 2 ,1]), m, lower=False))
        else:
            mu = tf.matmul(k_xz, util.tri_mat_solve(tf.transpose(k_zz_chol), util.tri_mat_solve(k_zz_chol, m, lower=True), lower=False))

        A = util.tri_mat_solve(k_zz_chol, tf.transpose(k_xz, [0, 2, 1]), lower=True) # N x M x S
        sig = k_xx - tf.matmul(tf.transpose(A, [0, 2, 1]), A) # N x S x S

        if self.context.whiten:
            A = tf.matmul(k_xz, util.tri_mat_solve(tf.transpose(k_zz_chol, [0, 2, 1]), s_chol, lower=False))
        else:
            A = tf.matmul(k_xz, util.mat_solve(k_zz, s_chol))

        sig = sig + tf.matmul(A, tf.transpose(A, [0, 2, 1]))

        return mu, sig


