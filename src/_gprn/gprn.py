import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline


from .parameters import Parameters
from .dataset import Dataset
from .util import util
from .minibatch import MiniBatch
from .predictions import Prediction
from .optimisers import *
from .context import *
from .elbos import *
from .composite_corrections import *

from . import debugger

class GPRN(object):
    def __init__(self, data, model=None, context=None):
        self.context = context or ContextFactory().create() 
        self.context._model = self
        self._model = model
        self.data = data

        self.predict_graph_built = False
        #self.context.parameters = MagnitudeCorrection(context)
        #self.context.parameters = CurvatureCorrection(context)

        self.correction_name = ''

        if not hasattr(self.context, 'use_latent_f'):
            self.context.use_latent_f = False

        if hasattr(self.context, 'correction'):
            self.context.parameters = self.context.correction
        else:
            if not hasattr(self.context, 'parameters'):
                self.context.parameters = NoCorrection

        print(self.context.parameters )
        if self.context.parameters == NoCorrection: 
            self.correction_name = 'NoCorrection'

        self.context.parameters = self.context.parameters(context)
        #self.context.parameters = CompositeMagnitudeCorrection(context)

        #self.context.parameters = Parameters(context)
        self.model = model(self.context) or Standard(self.context)

        self.load_variables_from_context()

    def load_variables_from_context(self):
        #self.model = self.context.model
        self.num_outputs = self.context.num_outputs
        self.jitter = self.context.jitter

    def build_graph(self, apply_correction=True):
        self.data.setup(self.context)
        self.data.create_placeholders()

        self.model.setup(self.data)
        self.model.setup_kernels()

        if apply_correction:
            self.context.parameters.apply_correction()

        self.elbo = self.model.build_elbo_graph()
        self.likelihood = self.model.build_likelihood_graph()

    def get_variational_parameters(self):
        return self.model.get_variational_parameters()

    def get_free_parameters(self):
        return self.model.get_free_parameters()

    def get_inducing_locations(self):
        Z = None
        with self.session.as_default():
            Z = self.model.inducing_locations.eval()
        return Z

    def get_u_v(self):
        q_means_v, q_covars_v = None, None
        with self.session.as_default():
            q_means_u = self.model.elbo.q_means_u_arr[0].eval()[0, 0,:]
            q_covars_u = tf.diag_part(self.model.elbo.precomputed.q_covar_u_arr[0][0, 0, :, :]).eval()
            if self.gprn_structure:
                q_means_v = self.model.elbo.q_means_v_arr[0].eval()[0, 0, 0,:]
                q_covars_v = tf.diag_part(self.model.elbo.precomputed.q_covar_v_arr[0][0, 0, 0, :, :]).eval()

        return q_means_u, q_covars_u, q_means_v, q_covars_v

    def get_posterior(self, r=0):
        f_dict = self.data.next_batch(0, force_all=True)
        mu_f, sigma_f, mu_wi, sigma_wi = self.session.run(self.model.elbo.get_posterior(r=0), feed_dict=f_dict)
        return mu_f, sigma_f, mu_wi, sigma_wi

    def optimise(self,  train = True, restore=False):
        self.num_train = self.data.get_num_training(source=0)

        elbos = []
        curv = train
        #curv = False

        if  (not restore and self.correction_name != 'NoCorrection'):
            print('in')
            self.build_graph(apply_correction=False)
            self.session = tf.Session()

            self.context.parameters.learn_information_matrices(self.session, optimise_flag=train)

            self.session.close()
            tf.reset_default_graph()

        #print('hi')

        self.build_graph(apply_correction=True)
        self.session = tf.Session()


        if not restore and self.correction_name != 'NoCorrection':
            train = True
            pass
        saver = tf.train.Saver()

        with tf.Session() as sess:
            writer = tf.summary.FileWriter('restore', sess.graph)
            writer.close()



        if train:
            optimsier = Optimiser1(self.context)
            #optimsier = MaximumLikelihoodOptimiser(self.context)
            #self.check_gradients()
            elbos = optimsier.optimise(self.elbo, self.likelihood, self.session, self.data, saver, restore)

            #self.print_vars()

            save_path = saver.save(self.session, self.context.restore_location)
        else:
            if restore:
                self.session.run(tf.global_variables_initializer())
                saver.restore(self.session, self.context.restore_location)

                f_dict = self.data.next_batch(0)
                
                options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                elbo = self.session.run(self.elbo, feed_dict=f_dict,options=options, run_metadata=run_metadata)
                fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                chrome_trace = fetched_timeline.generate_chrome_trace_format()
                with open('restore/timeline_01.json', 'w') as f:
                    f.write(chrome_trace)
                print(-np.array(elbo))
            else:
                self.session.run(tf.global_variables_initializer())
        #self.print_vars()

        return elbos

    def check_gradients(self):
        for v in tf.trainable_variables():
            print(v.name, 'grad: ', tf.gradients(self.elbo, v))

    def print_vars(self):
        variables_names = [v.name for v in tf.trainable_variables()]
        values = self.session.run(variables_names)
        for k, v in zip(variables_names, values):
            print ("Variable: ", k)
            print ("Shape: ", v.shape)
            print (v)

    def predict_f_w(self, x_test, r=0,a=0):
        self.x_test = tf.constant(x_test, dtype=tf.float32) 
        num_test = x_test.shape[0]
        f, w = self.model.build_prediction_graph(self.x_test, num_test, a=a, r=r,seperate=True)
        print('f: ', f)
        print('w: ', w)
        pred_f =  self.session.run(f, feed_dict={self.x_test: x_test})
        pred_w =  self.session.run(w, feed_dict={self.x_test: x_test})
        return pred_f, pred_w

    def predict(self, x_test, a=0, r=0, num_samples=5000, debug=False):
        self.x_test = tf.constant(x_test, dtype=tf.float32) 
        num_test = x_test.shape[0]

        if self.predict_graph_built == False:
            if self.context.monte_carlo is False:
                self.expected_value, self.variance = self.model.build_prediction_graph(self.x_test, num_test, a=a, r=r)
            else:
                self.sample_mc = self.model.build_sample_graph(self.x_test, num_test)
            #self.predict_graph_built = True

        if debug:
            self.session = tf.Session()
            self.session.run(tf.global_variables_initializer())

        if self.context.monte_carlo is False:
            pred_mean =  self.session.run(self.expected_value, feed_dict={self.x_test: x_test})
            pred_var =  self.session.run(self.variance, feed_dict={self.x_test: x_test})
            return pred_mean, pred_var
        else: 
            mean = None
            results = []
            var = None

            print(self.sample_mc)
            for n in range(num_samples):
                print('sample ',n)
                pred_mean =  self.session.run(self.sample_mc, feed_dict={self.x_test: x_test})
                results.append(pred_mean)

            results = np.array(results)
            print('results_shape, ',results.shape)

            mean = np.array([np.mean(results[:,:,i], axis=0) for i in range(self.context.num_outputs)]).T
            #ddof=1 to use sample variance=unbiased variance
            var = np.array([np.var(results[:,:,i], axis=0, ddof=1) for i in range(self.context.num_outputs)]).T
            print('mean_shape, ', mean.shape)
            print(var.shape)
            return mean, var

    def close(self):
        self.session.close()
        tf.reset_default_graph()

