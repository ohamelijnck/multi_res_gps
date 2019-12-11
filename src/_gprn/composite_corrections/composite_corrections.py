from ..scores import *
from ..optimisers import *
from ..import Parameters

import numpy as np
import tensorflow as tf

class CompositeCorrections(object):
    def __init__(self, context):
        self.context = context
        self.model = self.context._model
        self.J = None
        self.H = None
        self.H_parts = None

    def get_params(self, session=None):
        #arr =  [self.context.parameters.get('se_length_scale_1'), self.context.parameters.get('se_length_scale_0'), self.context.parameters.get('q_means_u_0')]

        arr_1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=Parameters.VARIATIONAL_SCOPE)
        arr_2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=Parameters.HYPER_SCOPE)

        #arr = arr_1+arr_2
        #arr = arr_1
        arr = arr_2

        #arr =  [self.context.parameters.get('se_length_scale_0'), self.context.parameters.get('noise_sigma_0'),self.context.parameters.get('noise_sigma_1')]
        #arr =  [self.context.parameters.get('noise_sigma_0'),self.context.parameters.get('noise_sigma_1')]
        
        #arr =  [self.context.parameters.get('noise_sigma_1')]

        #arr =  [self.context.parameters.get('noise_sigma_0')]
        #arr =  [self.context.parameters.get('se_length_scale_0'),  self.context.parameters.get('se_sigma_0'), self.context.parameters.get('q_means_u_0')]
        arr =  [self.context.parameters.get('se_length_scale_0')]
        #arr =  [self.context.parameters.get('se_sigma_0'), self.context.parameters.get('se_sigma_1')]
        #arr =  [self.context.parameters.get('se_sigma_0')]
        #arr =  [self.context.parameters.get('se_length_scale_0')]
        #arr =  [self.context.parameters.get('se_length_scale_0'), self.context.parameters.get('se_sigma_0')]
        #arr =  [self.context.parameters.get('se_sigma_0'), self.context.parameters.get('se_length_scale_0')]
        #arr =  [self.context.parameters.get('q_means_u_0')]
        print(arr)
        #arr =  [self.context.parameters.get('se_length_scale_0')]
        #arr =  [self.context.parameters.get('noise_sigma_0'), self.context.parameters.get('noise_sigma_1')]

        

        if session is not None:
            new_arr = []
            with session.as_default():
                for p in arr:
                    new_arr.append(p.eval())
            print('new_arr: ', new_arr)
            return new_arr
        return arr

    def estimate_information_matrices(self, optimise_flag = False):
        param_arr = self.get_params()
        print('param_arr: ', param_arr)
        saver = tf.train.Saver()
        if optimise_flag:
            optimsier = MaximumLikelihoodOptimiser(self.context)
            #optimsier = Optimiser1(self.context)
            elbos = optimsier.optimise(self.model.elbo, self.model.likelihood, self.model.session, self.model.data, saver, False)
            save_path = saver.save(self.model.session, self.context.restore_location)
        else:
            self.model.session.run(tf.global_variables_initializer())
            saver.restore(self.model.session, self.context.restore_location)


        #self.model.optimise(optimise_flag, not optimise_flag)


        print('CALCULATE FISHER')
        #param_arr = self.model.get_free_parameters()
        score = FisherInformation(self.model)
        fisher_s = score.observed_matrix(param_arr)
        self.J = fisher_s

        print('CALCULATE HESSIAN')
        score = LikelihoodHessian(self.model)
        self.H = score.observed_matrix(param_arr)
        self.H_parts = score.hessian_parts


    
