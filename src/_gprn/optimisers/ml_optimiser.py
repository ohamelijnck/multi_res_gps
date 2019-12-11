import numpy as np
import tensorflow as tf
import math


from .. import util
from . import Optimiser

class MaximumLikelihoodOptimiser(Optimiser):
    def __init__(self, context):
        self.context  = context

    def get_variables(self):
        with tf.variable_scope("parameters", reuse=True):
            self.q_means_u = tf.get_variable(name='q_means_u')
            self.q_covars_u = tf.get_variable(name='q_covars_u_raw')
            self.q_covars_u_raw = tf.get_variable(name='q_covars_u_raw')
            self.q_means_v = tf.get_variable(name='q_means_v')
            self.q_covars_v = tf.get_variable(name='q_covars_v_raw')
            self.q_raw_weights = tf.get_variable(name='q_raw_weights')
            self.inducing_locations = tf.get_variable(name='inducing_locations')

            if self.context.model == 'MultiRes':
                self.q_covars_ugg = tf.get_variable(name='q_covars_ugg')
                self.q_means_ugg = tf.get_variable(name='q_means_ugg')

                self.q_covars_ug = tf.get_variable(name='q_covars_ug')
                self.q_means_ug = tf.get_variable(name='q_means_ug')

                self.q_covars_uh = tf.get_variable(name='q_covars_uh_raw')
                self.q_means_uh = tf.get_variable(name='q_means_uh')

                self.variations_params_h = [self.q_means_u, self.q_covars_u, self.q_means_v, self.q_covars_v, self.q_means_ug, self.q_covars_ug, self.q_means_uh, self.q_covars_uh]
                self.variations_params_m = [self.q_means_ugg, self.q_covars_ugg]

            self.q_raw_weights = tf.get_variable(name='q_raw_weights')

        graph = tf.get_default_graph()

        #self.x_train_ph = graph.get_tensor_by_name("train_inputs:0")
        #self.y_train_ph = graph.get_tensor_by_name("train_outputs:0")
        #self.x_test_ph = graph.get_tensor_by_name("test_inputs:0")
        #self.y_train_nans_ph = graph.get_tensor_by_name("train_outputs_nans:0")

        self.x_train_ph = self.data.get_placeholder(source=0, var='x')
        self.y_train_ph = self.data.get_placeholder(source=0, var='y')
        self.x_test_ph = self.data.get_placeholder(source=0, var='xs')
        self.y_train_nans_ph = self.data.get_placeholder(source=0, var='y_nan')

        self.inducing_locations_params = [self.inducing_locations]
        self.variations_params = [self.q_means_u, self.q_covars_u, self.q_means_v, self.q_covars_v,self.q_raw_weights]
        self.hyper_params_no_inducing = [var for var in  tf.trainable_variables() if var is not self.variations_params and var is not self.inducing_locations]
        self.hyper_params = [var for var in  tf.trainable_variables() if var is not self.variations_params]
        self.all_params = [var for var in  tf.trainable_variables() if var is not self.inducing_locations]

    def optimise(self, elbo, likelihood, session, data, saver, restore):
        self.elbo = elbo
        self.likelihood = likelihood
        self.session = session
        self.data = data
        debug = self.context.debug



        step_3 = [True, tf.train.RMSPropOptimizer(0.01).minimize(-self.likelihood)]
        #step_3 = [True, tf.train.RMSPropOptimizer(1e-3).minimize(-self.likelihood)]
        #step_3 = [True, tf.train.AdagradOptimizer(0.01).minimize(-self.likelihood)]
        steps = [step_3]
        train_step = steps[0][1]

        if restore:
            self.session.run(tf.global_variables_initializer())
            saver.restore(self.session, self.context.restore_location)
        else:
            self.session.run(tf.global_variables_initializer())

        epoch = 0
        epoch_vals = []
        convergence = 0.001
        while epoch < self.context.num_epochs:
            try:
                f_dict = self.data.next_batch(epoch)
                lik = self.session.run(self.likelihood, feed_dict=f_dict)
                print(lik)
                new_val = lik
                epoch_vals.append(-new_val)
                self.session.run(
                    train_step, 
                    feed_dict=f_dict
                )
                epoch = epoch + 1
            except (KeyboardInterrupt, SystemExit):
                break







