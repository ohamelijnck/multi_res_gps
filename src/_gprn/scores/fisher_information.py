import numpy as np
import tensorflow as tf

from . import Score
from .. import util
from .. precomputers import Precomputed

class FisherInformation(Score):
    def __init__(self, model):
        self.model = model
        self.context = self.model.context
        self.data = self.model.data
        self.r = 0

        self.setup()

    def setup(self):
        pass

    def observed_matrix(self, parameters):
        """
            Given u(\theta; y_{nm}) = \nabla cl (\theta;y_{nm})
            \frac{1}{n}\sum^n_{n=1} \sum^{m}_{m=1} u(\theta_{cl}; y_{nm})u(\theta_{cl}; y_{nm})^T
        """
        s_total = 0

        num_sources = self.data.get_num_sources()
        likelihood = self.model.model.likelihood
        total_arr = None
        total_n = 0

        r = 0

        param_scores = None
        lik_graph = likelihood.build_graph()
        f_dict = self.data.next_batch(epoch=0,force_all=True)
        print(f_dict)
        for p in parameters:
            print('lik_graph: ', lik_graph)
            print('p:', p)
            score = tf.gradients(lik_graph, p)
            print('score:', score)
            p_shape = self.model.session.run(tf.shape(p), feed_dict=f_dict)
            if score[0] is None:
                #nusiance parameter 
                a = np.zeros([np.prod(p_shape).astype(int), 1])
            else:
                a = np.array(self.model.session.run(score, feed_dict=f_dict))
                a = np.reshape(a, [np.prod(p_shape).astype(int), 1])
                print('fisher: a: ', a)

            if param_scores is None:
                param_scores = a
            else:
                param_scores = np.concatenate([param_scores, a], axis=0)

        param_vec = np.expand_dims(np.squeeze(np.array(param_scores)), -1)
        total_arr = np.matmul(param_vec, param_vec.T)

        s =  total_arr
        if len(parameters) == 1:
            s = np.expand_dims(np.expand_dims(s, -1), -1)
        print('s.shape: ', s.shape)

        return s


  
