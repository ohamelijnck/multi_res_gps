import numpy as np
import tensorflow as tf

from . import Kernel
from .. import util

#adapted from GPFlow
class ArcCosine(Kernel):
    _id = -1
    implemented_orders = {0, 1, 2}
    def __init__(self, num_dimensions, order=2, variance=1.0, weight_variances=1., bias_variance=1., ARD=None, mask=None):
        super().__init__(mask)
        ArcCosine._id += 1
        self.id = ArcCosine._id

        self.num_dimensions = num_dimensions
        self.order = order
        self._variance = variance
        self._bias_variance = bias_variance
        self._weight_variances = weight_variances

    def setup(self, context):
        self.context = context
        self.parameters = self.context.parameters
        self.variance_raw = self.parameters.create(name='arc_cos_variance_'+str(self.id), init=0.1*tf.ones(shape=[1]), trainable=True)
        self.bias_variance_raw = self.parameters.create(name='arc_cos_bias_variance_'+str(self.id), init=0.1*tf.ones(shape=[1]), trainable=True)
        self.weight_variances_raw = self.parameters.create(name='arc_cos_weight_variances_'+str(self.id), init=0.1*tf.ones(shape=[self.num_dimensions]), trainable=True)

        self.variance = util.var_postive(self.variance_raw)
        self.bias_variance = util.var_postive(self.bias_variance_raw)
        self.weight_variances = util.var_postive(self.weight_variances_raw)
        self.parameters = [self.variance_raw, self.bias_variance_raw, self.weight_variances_raw]

    def _weighted_product(self, weight_variances, bias_variance, X, X2=None):
        if X2 is None:
            return tf.reduce_sum(weight_variances * tf.square(X), axis=-1) + bias_variance
        return tf.matmul((weight_variances * X), X2, transpose_b=True) + bias_variance

    def _J(self, theta):
        """
        Implements the order dependent family of functions defined in equations
        4 to 7 in the reference paper.
        """
        if self.order == 0:
            return np.pi - theta
        elif self.order == 1:
            return tf.sin(theta) + (np.pi - theta) * tf.cos(theta)
        elif self.order == 2:
            return 3. * tf.sin(theta) * tf.cos(theta) + \
                   (np.pi - theta) * (1. + 2. * tf.cos(theta) ** 2)

    def _kernel(self, X, X2, jitter=False, debug=False, include_dimensions=None):
        #if not presliced:
        #    X, X2 = self._slice(X, X2)

        variance = self.variance
        bias_variance = self.bias_variance
        weight_variances = self.weight_variances

        if include_dimensions is not None:
            weight_variances = self.s(self.weight_variances, include_dimensions)

        X_denominator = tf.sqrt(self._weighted_product(weight_variances, bias_variance, X))
        if X2 is None:
            X2 = X
            X2_denominator = X_denominator
        else:
            X2_denominator = tf.sqrt(self._weighted_product(weight_variances, bias_variance, X2))

        numerator = self._weighted_product(weight_variances, bias_variance, X, X2)
        X_denominator = tf.expand_dims(X_denominator, -1)
        X2_denominator = tf.matrix_transpose(tf.expand_dims(X2_denominator, -1))
        cos_theta = numerator / X_denominator / X2_denominator
        jitter = self.context.jitter
        theta = tf.acos(jitter + (1 - 2 * jitter) * cos_theta)

        return variance * (1. / np.pi) * self._J(theta) * \
               X_denominator ** self.order * \
               X2_denominator ** self.order

    def get_parameters(self):
        return self.parameters

