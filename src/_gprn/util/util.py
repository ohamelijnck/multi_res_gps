import tensorflow as tf
import numpy as np
#from tensorflow.python.ops.distributions.util import fill_lower_triangular

def vec_to_lower_triangle_matrix(n,vec):
    indices = tf.constant(list(zip(*np.tril_indices(n))))
    #mat = tf.sparse_to_dense(sparse_indices=indices, output_shape=[n,n], sparse_values=vec, default_value=0)
    mat = tf.scatter_nd(indices=indices, shape=[n,n], updates=vec)
    return mat

def vec_cholesky_to_mat(n, vec, jitter):
    vec = vec_to_lower_triangle_matrix(n, vec)
    mat = tf.matmul(vec, vec, transpose_b=True)
    return mat

def safe_exp(val):
    val = tf.clip_by_value(val, -1e20, 1e20) 
    e = tf.exp(val)
    return tf.clip_by_value(e, 1e-20, 1e20)

def safe_log(val):
    val = tf.clip_by_value(val, 1e-20, 1e20) 
    l = tf.log(val)
    return tf.clip_by_value(l, -1e20, 1e20) 

def log_sum_exp(arr):
    #https://en.wikipedia.org/wiki/LogSumExp
    max_val = tf.reduce_max(arr)
    return max_val + safe_log(tf.reduce_sum(safe_exp(arr-max_val)))

def log_chol_matrix_det(chol):
    return 2*tf.reduce_sum(safe_log(tf.diag_part(chol)))

def chol_solve(a, b):
    cast_chol = lambda a, b: tf.cast(tf.cholesky_solve(tf.cast(tf.cholesky(a), tf.float64), tf.cast(b, tf.float64)), tf.float32)
    return cast_chol(a, b)

def svd_solve(A, b):
    b = tf.cast(b, tf.float64)
    s, u, v = tf.linalg.svd(tf.cast(A, tf.float64))
    w_max = tf.reduce_max(s)
    z = tf.where(tf.less(1/s, 1e-12*w_max), 1/s, tf.zeros(tf.shape(s), tf.float64)) 
    #z = 1/s 
    A_1 = tf.matmul(v, tf.cast(tf.matmul(tf.diag(z), tf.linalg.adjoint(u)), tf.float64), adjoint_a=False)
    x = tf.matmul(A_1, b)
    return tf.cast(x, tf.float32)

def mat_solve(a, b):
    mat_solve_cast = lambda a, b: tf.cast(tf.matrix_solve(tf.cast(a, tf.float64), tf.cast(b, tf.float64), adjoint=True), tf.float32)
    return mat_solve_cast(a, b)

def tri_mat_solve(a, b, lower=True, name=''):
    mat_solve_cast = lambda a, b, lower: tf.cast(tf.linalg.triangular_solve(tf.cast(a, tf.float64), tf.cast(b, tf.float64), lower=lower, name=name+'mat_solve'), tf.float32)
    return mat_solve_cast(a, b, lower)

def log_normal_chol(x, mu, chol, n, k=None):
    if x == 0.0:
        err = mu
    else:
        err = tf.subtract(mu, x)

    val =  -0.5*(n*safe_log(2*np.pi)+log_chol_matrix_det(chol)+tf.matmul(err, tri_mat_solve(tf.transpose(chol), tri_mat_solve(chol, err, lower=True), lower=False), transpose_a=True))

    return val

def var_postive(sigma):
    return safe_exp(sigma)
    #return tf.square(sigma)

def inv_var_postive(sigma):
    return safe_log(sigma)
    #return tf.sqrt(sigma)

def sample_index_with_prob_weights(weights, n):
    r = tf.squeeze(tf.random_uniform(shape=[1], minval=0, maxval=1, dtype=tf.float32))
    k = tf.constant(-1)
    for l in range(n):
        k = tf.cond(
            tf.reduce_sum(weights[:l]) < r,
            lambda: tf.constant(l),
            lambda: k
        )
    return k

def covar_to_mat(n, covar, use_diag_flag, jitter):
    if use_diag_flag:
        return tf.square(tf.diag(covar))
    else:
        return vec_cholesky_to_mat(n, covar, jitter)

def add_jitter(K, _jit):
    #a = tf.convert_to_tensor([0.0, -tf.reduce_min(tf.self_adjoint_eigvals(K))])
    #min_eigval = a[tf.argmax(a)]
    #a = tf.convert_to_tensor([_jit, min_eigval])
    #jit = a[tf.argmax(a)]
    jit = _jit

    _K =  K+(jit*tf.eye(tf.shape(K)[0]))
    
    return _K



