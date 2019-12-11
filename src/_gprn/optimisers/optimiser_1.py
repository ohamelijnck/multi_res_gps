import numpy as np
import tensorflow as tf
import pandas as pd
import math


from .. import util
from .. import Parameters
from . import Optimiser

class Optimiser1(Optimiser):
    def __init__(self, context):
        self.context  = context
   
    def get_source_vars(self, r):
        q_means_u = self.context.parameters.get(name='q_means_u_{r}'.format(r=r))

        q_means_v = self.context.parameters.get(name='q_means_v_{r}'.format(r=r))

        q_covars_u_raw = self.context.parameters.get(name='q_covars_u_{r}_raw'.format(r=r))
        q_covars_v_raw = self.context.parameters.get(name='q_covars_v_{r}_raw'.format(r=r))

        a = [q_means_u, q_means_v, q_covars_u_raw, q_covars_v_raw]
        return a + self.context.kernels[r]['f'][0].get_parameters()


    def optimise(self, elbo, likelihood, session, data, saver, restore):
        self.elbo = elbo
        self.session = session
        self.data = data

        debug = self.context.debug

        if self.context.split_optimise:
            var_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=Parameters.VARIATIONAL_SCOPE)
            hyper_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=Parameters.HYPER_SCOPE)

            var_params = []
            a = 0
            q_means_u = self.context.parameters.get(name='q_means_u_{r}'.format(r=a))
            q_covars_u = self.context.parameters.get(name='q_covars_u_{r}'.format(r=a))

            q_means_v = self.context.parameters.get(name='q_means_v_{r}'.format(r=a))
            q_covars_v = self.context.parameters.get(name='q_covars_v_{r}'.format(r=a))

            m_n = q_means_u
            s_n = q_covars_u + tf.matmul(tf.expand_dims(q_means_u, -1), tf.expand_dims(q_means_u, -1), transpose_b=True)

            m_v = q_means_v
            s_v = q_covars_v + tf.matmul(tf.expand_dims(q_means_v, -1), tf.expand_dims(q_means_v, -1), transpose_b=True)

            var_params = [m_n, s_n, m_v, s_v]

            step_2 = [False, tf.train.AdamOptimizer(0.01).minimize(-self.elbo[0],var_list=var_params)]

        #step_3 = [True, tf.train.AdamOptimizer(1e-3).minimize(-self.elbo[0])]
        step_3 = [True, tf.train.AdamOptimizer(0.01).minimize(-self.elbo[0])]
        #step_3 = [True, tf.train.AdamOptimizer(0.01).minimize(-self.elbo[0])]

        #step_3 = [True, tf.train.RMSPropOptimizer(0.01).minimize(-self.elbo[0])]
        #step_inducing = [False, tf.train.RMSPropOptimizer(1e-5, momentum=0.9).minimize(-self.elbo, var_list=self.inducing_locations_params)]
        #steps = [step_2, step_3]
        #steps = [step_2, step_3]

        if self.context.split_optimise:
            steps = [step_2, step_3]
        else:
            steps = [step_3]


        current_step = 0

        print('Variable init')
        if restore:
            self.session.run(tf.global_variables_initializer())
            saver.restore(self.session, self.context.restore_location)
        else:
            self.session.run(tf.global_variables_initializer())

        epoch = 0
        elbos = []
        convergence = 0.001
         
        _min_steps = 100
        _max_steps = 1000

        min_steps = _min_steps
        max_steps = _max_steps

        epoch = 0

        self.print_vars()


        debug = False
        total_saved = []
        total_saved_dx = []
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

                elbo = np.squeeze(np.array(elbo))

                print('elbo')
                print(-elbo)

                if self.context.save_parameters == True:
                    params = ['noise_sigma_0', 'q_means_u_0', 'q_covars_u_0_raw']
                    row = np.array([-elbo.flatten()])
                    names = ['elbo']
                    for p in params:
                        val = self.context.parameters.get(p).eval(self.session)
                        val = np.expand_dims(val.flatten(), 0)
                        names = names + [p+'_{i}'.format(i=i) for i in range(val.shape[1])]

                        row = val if row is None else np.concatenate([row, val], axis=1)
                    #val = self.context.parameters.get('q_mu').eval(self.session)

                    if epoch == 0:
                        pd.DataFrame(row, columns=names).to_csv(self.context.save_parameters_location, header=True, index=False)
                    else:
                        pd.DataFrame(row, columns=names).to_csv(self.context.save_parameters_location, mode='a', header=False, index=False)



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
                variables_names = [v.name for v in tf.trainable_variables()]
                values = self.session.run(variables_names)
                for k, v in zip(variables_names, values):
                    print ("Variable: ", k)
                    print ("Shape: ", v.shape)
                    print (v)
                exit()
                break

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

