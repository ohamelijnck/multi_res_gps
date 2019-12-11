import numpy as np
import tensorflow as tf

from . import Score
from .. import util
from .. precomputers import Precomputed

class LikelihoodHessian(Score):
    def __init__(self, model):
        self.model = model
        self.context = self.model.context
        self.data = self.model.data
        self.setup()

    def setup(self):
        self.hessian_parts = []

    def observed_matrix(self, parameters):
        """
            Given u(\theta; y_{nm}) = \nabla cl (\theta;y_{nm})
            \frac{1}{n}\sum^n_{n=1} \sum^{m}_{m=1} u(\theta_{cl}; y_{nm})u(\theta_{cl}; y_{nm})^T
        """
        s_total = 0

        num_likelihood_components= self.context.num_likelihood_components
        likelihood = self.model.model.likelihood
        total_arr = None
        total_n = 0

        f_dict = self.data.next_batch(epoch=0, force_all=True)
        for r in range(num_likelihood_components):
            print('r: ', r)
            param_scores = None
            for p in parameters:
                score = tf.gradients(likelihood.lik_arr[r]._build_log_likelihood(), p)
                p_shape = self.model.session.run(tf.shape(p), feed_dict=f_dict)

                if score[0] is None:
                    #nusiance parameter 
                    a = np.zeros([np.prod(p_shape).astype(int), 1])
                else:
                    a = np.array(self.model.session.run(score, feed_dict=f_dict))
                    a = np.reshape(a, [np.prod(p_shape).astype(int), 1])

                if param_scores is None:
                    param_scores = a
                else:
                    param_scores = np.concatenate([param_scores, a], axis=0)


            param_vec = np.expand_dims(np.squeeze(np.array(param_scores)), -1)
            a = np.matmul(param_vec, param_vec.T)
            self.hessian_parts.append(np.copy(a))

            print('hessian: a: ', a)
            if total_arr is None:
                total_arr = a
            else:
                total_arr += np.copy(a)

        s =  total_arr

        if len(parameters) == 1:
            s = np.expand_dims(np.expand_dims(s, -1), -1)
        print('s.shape: ', s.shape)

        return s

 

