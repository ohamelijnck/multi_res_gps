import numpy as np
import tensorflow as tf
import math
import matplotlib.pyplot as plt


from .. import util
from . import Optimiser

class Optimiser1(Optimiser):
    def __init__(self, context):
        self.context  = context


    def monthly_vars(self):
        v = [self.q_means_ugg, self.q_covars_ugg]
        v += self.context.kern_gg[0].get_parameters()
        return v

    def multi_res_vars(self):
        v = [self.q_means_ug, self.q_covars_ug, self.q_means_uh, self.q_covars_uh]
        v += self.context.kern_h[0].get_parameters()
        return v

    def get_f_variables(self):
        arr =  [self.context.parameters.get(name=i) for i in ['q_means_u_0', 'q_covars_u_0_raw',  'q_raw_weights']]
        for r_kernels in self.context.kernels:
            for k in r_kernels['f']:
                arr = arr + k.get_parameters()

        return arr


    def get_w_variables(self):
        arr =  [self.context.parameters.get(name=i) for i in ['q_means_v_0', 'q_covars_v_0_raw', 'q_raw_weights']]

        for r_kernels in self.context.kernels:
            if 'w' in r_kernels:
                for row in r_kernels['w']:
                    for k in row:
                        arr = arr + k.get_parameters()
        return arr

    def get_variables(self):
        self.q_means_u = self.context.parameters.get(name='q_means_u_0')
        self.q_covars_u = self.context.parameters.get(name='q_covars_u_raw')
        self.q_covars_u_raw = self.context.parameters.get(name='q_covars_u_raw')
        self.q_means_v = self.context.parameters.get(name='q_means_v')
        self.q_covars_v = self.context.parameters.get(name='q_covars_v_raw')
        self.q_raw_weights = self.context.parameters.get(name='q_raw_weights')

        if self.context.model == 'MultiRes':
            self.q_covars_ugg = self.context.parameters.get(name='q_covars_ugg')
            self.q_means_ugg = self.context.parameters.get(name='q_means_ugg')

            self.q_covars_ug = self.context.parameters.get(name='q_covars_ug')
            self.q_means_ug = self.context.parameters.get(name='q_means_ug')

            self.q_covars_uh = self.context.parameters.get(name='q_covars_uh_raw')
            self.q_means_uh = self.context.parameters.get(name='q_means_uh')

            self.variations_params_h = [self.q_means_u, self.q_covars_u, self.q_means_v, self.q_covars_v, self.q_means_ug, self.q_covars_ug, self.q_means_uh, self.q_covars_uh]
            self.variations_params_m = [self.q_means_ugg, self.q_covars_ugg]

        self.q_raw_weights = self.context.parameters.get(name='q_raw_weights')

        graph = tf.get_default_graph()

        #self.x_train_ph = graph.get_tensor_by_name("train_inputs:0")
        #self.y_train_ph = graph.get_tensor_by_name("train_outputs:0")
        #self.x_test_ph = graph.get_tensor_by_name("test_inputs:0")
        #self.y_train_nans_ph = graph.get_tensor_by_name("train_outputs_nans:0")

        self.x_train_ph = self.data.get_placeholder(source=0, var='x')
        self.y_train_ph = self.data.get_placeholder(source=0, var='y')
        self.x_test_ph = self.data.get_placeholder(source=0, var='xs')
        self.y_train_nans_ph = self.data.get_placeholder(source=0, var='y_nan')


        self.hyper_params = [var for var in  tf.trainable_variables() if var is not self.variations_params]




    def optimise(self, elbo, likelihood, session, data, saver, restore):
        self.elbo = elbo
        self.session = session
        self.data = data

        #self.get_variables()

        #self.session.run(tf.global_variables_initializer())
        #elbo = self.session.run(self.elbo)
        #exit()


        debug = self.context.debug

        #step_1 = [True, tf.train.RMSPropOptimizer(1e-5, momentum=0.9).minimize(-self.elbo, var_list=self.hyper_params_no_inducing)]
        if self.context.model == 'MultiRes':
            step_1 = [False, tf.train.RMSPropOptimizer(0.01, momentum=0.9).minimize(-self.elbo[0], var_list=self.monthly_vars())]
        #step_2 = [True, tf.train.RMSPropOptimizer(0.01, momentum=0.9).minimize(-self.elbo[0], var_list=self.multi_res_vars())]

        #step_2 = [True, tf.train.RMSPropOptimizer(1e-5, momentum=0.9).minimize(-self.elbo, var_list=self.all_params)]
        #step_3 = [True, tf.train.RMSPropOptimizer(0.01, momentum=0.9).minimize(-self.elbo[0])]
        print(self.get_f_variables())
        print(self.get_w_variables())
        step_1 = [True, tf.train.AdamOptimizer(0.01).minimize(-self.elbo[0],var_list=self.get_f_variables())]
        step_2 = [False, tf.train.AdamOptimizer(0.01).minimize(-self.elbo[0],var_list=self.get_w_variables())]
        step_3 = [True, tf.train.RMSPropOptimizer(0.01).minimize(-self.elbo[0])]

        #step_inducing = [False, tf.train.RMSPropOptimizer(1e-5, momentum=0.9).minimize(-self.elbo, var_list=self.inducing_locations_params)]

        if self.context.model == 'MultiRes':
            #steps = [step_1, step_3 ]
            steps = [step_3 ]
        else:
            #steps = [step_1,step_2]
            #steps = [step_1]
            steps = [step_3 ]

        current_step = 0

        print('Variable init')
        if restore:
            self.session.run(tf.global_variables_initializer())
            saver.restore(self.session, self.context.restore_location)
        else:
            self.session.run(tf.global_variables_initializer())



        self.print_vars()


        epoch = 0
        elbos = []
        convergence = 0.001
         
        _min_steps = 100
        _max_steps = 1000

        min_steps = _min_steps
        max_steps = _max_steps

        epoch = 0


        debug = False
        while epoch < self.context.num_epochs:
            try:
                print('epoch: {epoch}'.format(epoch=epoch))

                train_step = steps[current_step][1]
                run_multiple_flag = steps[current_step][0]

                if run_multiple_flag is None:
                    current_step = (current_step + 1) % len(steps)
                    min_steps = _min_steps
                    max_steps = _max_steps
                    continue

                f_dict = self.data.next_batch(epoch)

                self.session.run(
                    train_step, 
                    feed_dict=f_dict
                )

                elbo= self.session.run(self.elbo, feed_dict=f_dict)

                print('elbo')
                print(-elbo)


                if debug:
                    break
                    pass

                #new_elbo = -elbo[0][0]


                if ((max_steps <= 0) or (len(elbos) != 0 and min_steps <= 0 and np.abs(new_elbo-elbos[-1]) <= convergence)):
                    #toggle
                    print('toggle')
                    print(epoch)
                    print('step: {step}'.format(step=current_step))

                    if run_multiple_flag is False:
                        steps[current_step][0] = None

                    current_step = (current_step + 1) % len(steps)
                    min_steps = _min_steps
                    max_steps = _max_steps

                #elbos.append(-elbo[0][0])


                epoch = epoch + 1
                min_steps = min_steps - 1
                max_steps = max_steps - 1

            except (KeyboardInterrupt, SystemExit):
                break
            except Exception as e:
                print(e)
                exit()
                variables_names = [v.name for v in tf.trainable_variables()]
                values = self.session.run(variables_names)
                for k, v in zip(variables_names, values):
                    print ("Variable: ", k)
                    print ("Shape: ", v.shape)
                    print (v)
                #break

        if debug:
            exit()


        #plt.plot(np.squeeze(ell_arr)[100:])
        #plt.show()
        #plt.plot(np.squeeze(ent_arr)[100:])
        #plt.show()
        #plt.plot(np.squeeze(cross_arr)[100:])
        #plt.show()



        self.print_vars()
        return elbos

    def print_vars(self):
        variables_names = [v.name for v in tf.trainable_variables()]
        values = self.session.run(variables_names)
        for k, v in zip(variables_names, values):
            print ("Variable: ", k)
            print ("Shape: ", v.shape)
            print (v)
        #break

    def nat_opt(self):
        self.q_num_components, self.num_latent, self.num_inducing
        for k in range(self.gprn.q_num_components):
            for q in range(self.gprn.num_latent):
                m = self.gprn.q_means_u[k,q,:]
                s = self.gprn.q_covars_u[k,q,:]
                s = util.covar_to_mat(self.gprn.num_inducing, s, self.gprn.use_diag_covar_flag, self.gprn.jitter)

                s_inv = tf.inverse(s)

                theta1 = tf.matmul(s_inv, m)
                theta2 = tf.scalar_mul(-0.5, s_inv)

                n1 = m
                n2 = tf.matmul(m, m, transpose_b=True) + s

                #update: 
                l = 0.1

                theta1 = theta1 + l*tf.gradients(self.gprn.elbo, n1)
                theta2 = theta2 + l*tf.gradients(self.gprn.elbo, n2)

                s_prime = tf.inverse(tf.scalar_mul(-1/0.5, theta2))

                self.gprn.q_means_u[k,q,:] = tf.matmul(s_prime, theta1)

                #transform s back to chol/diag form

