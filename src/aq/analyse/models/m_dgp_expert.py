import logging, os
import numpy as np
import tensorflow as tf

#disable TF warnings
if True:
    logging.disable(logging.WARNING)
    os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
    tf.logging.set_verbosity(tf.logging.ERROR)

import gpflow
from gpflow import settings
from gpflow.training import AdamOptimizer

import _mr_dgp as MR_DGP
from _mr_dgp import MR_Mixture
from _mr_dgp import MR_SE, MR_Linear, MR_KERNEL_PRODUCT
from _mr_dgp.mr_mixing_weights import MR_Average_Mixture, MR_Base_Only, MR_DGP_Only, MR_Variance_Mixing

from scipy.cluster.vq import kmeans2

import matplotlib.pyplot as plt

import sys
import os
import glob



np.random.seed(0)

def get_config():
    return  [
        {
            'name': 'CMGP-DGP-EXPERT',
            'pretty_name': r'\cmgpdgp',
            'model_name': 'mr_dgp',
            'file_prefix': 'cmgp_dgp_expert_1',
            'experiment_id': 0,
            'plot_fine_grid': True,
            'num_outputs': 1,
            'output_names': ['SAT', 'LAQN'],
            'kernel_type': 'sum',
            'order': 'normal',
            'vis_iter': 10,
            'restore': False,
            'ignore': False,
            #'z':[[500], [200, 100]],
            'z':[[100], [200, 100]],
            'log': False,
            #PLOTTING CONFIG
            'plot_order': 'top',
            'plot_fine_grid_order': 0,
        }
    ]

def batch_predict(m, XS, num_samples):
    batch_size = 100
    NS = XS.shape[0]

    if NS < batch_size:
        batch_size = XS.shape[0]

    num_batches = int(np.ceil(NS/batch_size))

    ys_arr = None
    ys_var_arr = None
    i = 0
    print("--------------- ", NS, batch_size, num_batches, " ---------------")

    for b in range(num_batches):
        if b == num_batches-1:
            batch = XS[i:, :]
        else:
            batch = XS[i:i+batch_size, :]

        i = i+batch_size


        if True:
            ys, ys_var = m.predict_y_experts(batch,  num_samples)
        else:
            ys = np.ones([num_samples, batch.shape[0],1, 1])
            ys_var = np.ones([num_samples, batch.shape[0], 1, 1])

        if ys_arr is None:
            ys_arr = ys
            ys_var_arr = ys_var
        else:
            ys_var_arr = np.concatenate([ys_var_arr, ys_var], axis=1)
            ys_arr = np.concatenate([ys_arr, ys], axis=1)


    #ys = np.concatenate(ys_arr, axis=0)
    #ys_var = np.concatenate(ys_var_arr, axis=0)

    
    return ys_arr, ys_var_arr

def prediction(CONFIG, m, X, xs, xs_grid, xs_fine_grid, vis_xs):
    def get_sample_mean_var(ys, vs):
        ys = ys[:, :, 0, :]
        vs = vs[:, :, 0, :]
        mu = np.mean(ys, axis=0)
        sig = np.mean(vs+ys**2, axis=0)-np.mean(ys, axis=0)**2
        return mu, sig


    num_samples = 20

    #X_mu, X_sig = m.predict_y_experts(X[1][:, 0, :], num_samples)
    X_mu, X_sig = batch_predict(m, X[1][:, 0, :], num_samples)
    X_mu, X_sig = get_sample_mean_var(X_mu, X_sig)
    pred_y = np.concatenate([X_mu, X_sig], axis=1)
    np.save('../results/{prefix}_y'.format(prefix=CONFIG['file_prefix']), pred_y)

    xs_mu, xs_sig = batch_predict(m, xs, num_samples)
    xs_mu, xs_sig = get_sample_mean_var(xs_mu, xs_sig)
    pred_ys = np.concatenate([xs_mu, xs_sig], axis=1)
    np.save('../results/{prefix}_ys'.format(prefix=CONFIG['file_prefix']), pred_ys)

    if True:
        ys_fine_grid_0, ys_var_fine_grid_0 = batch_predict(m, xs_fine_grid, num_samples)
        ys_fine_grid_0, ys_var_fine_grid_0 = get_sample_mean_var(ys_fine_grid_0, ys_var_fine_grid_0)
        pred_ys_fine_grid = np.concatenate([ys_fine_grid_0, ys_var_fine_grid_0], axis=1)
        np.save('../results/{prefix}_ys_small_fine_grid'.format(prefix=CONFIG['file_prefix']), pred_ys_fine_grid)

    print(X_mu.shape)
    print(X_sig.shape)
    
    
def main(CONFIG, return_m=False, force_restore=False):
    print(CONFIG)
    #===========================GET DATA===========================
    CONFIG['num_layers'] = 2
    num_z = CONFIG['z']

    #X[0],Y[0] are SAT DATA
    #X[1],Y[1] are LAQN DATA

    #X[0]: 196 x 100 x 6
    #X[1]: 1020 x 1 x 6
    X = np.load('../data/data_with_features/processed_x_train.npy', allow_pickle=True)
    Y = np.load('../data/data_with_features/processed_y_train.npy', allow_pickle=True)

    if CONFIG['log']:
        Y[0] = np.log(Y[0])
        Y[1] = np.log(Y[1])
   
    print('num nans Y[0]): ', sum(np.isnan(Y[0]))) #0
    print('num nans Y[1]): ', sum(np.isnan(Y[1]))) #206

    #xs: 1034 x 6
    xs = np.load('../data/data_with_features/processed_x_test.npy', allow_pickle=True) #100 , 50
    #xs_grid: 30000 x 6
    xs_grid = np.load('../data/data_with_features/processed_x_test_small_grid.npy', allow_pickle=True)
    #xs_fine_grid: 30000 x 6
    xs_fine_grid = np.load('../data/data_with_features/processed_x_test_small_fine_grid.npy', allow_pickle=True)

    #vis_xs: 50 x 6
    vis_xs = np.load('../data/data_with_features/processed_vis_x_{id}.npy'.format(id=CONFIG['vis_iter']), allow_pickle=True) #100 , 50

    #===========================CLEAN DATA===========================
    _OLD_X = X.copy()
    #--------- Remove NANs from Y[1]
    _x, _y = X[1], Y[1]
    idx = (~np.isnan(Y[1][:, 0]))
    X[1] = _x[idx, :] #814x1x6
    Y[1] = _y[idx, :] #814x1


    #===========================SETUP MODEL===========================
    def get_inducing_points(X, num_z=None):
        if len(X.shape) == 3:
            X = X.reshape([X.shape[0]*X.shape[1], X.shape[2]])

        if num_z is None or num_z > X.shape[0]:
            Z = X
        else:
            Z = kmeans2(X, num_z, minit='points')[0] 
        return Z

    def get_kernel_product(K, active_dims=[0], lengthscales=[1.0], variances=[1.0], name=''):
        if not isinstance(K, list):
            K = [K for i in range(len(active_dims))]

        if lengthscales is None:
            kernels = [K[i](input_dim=1, variance=variances[i], active_dims=[active_dims[i]], name=name+'_{i}'.format(i=i)) for i in range(len(active_dims))]
        else:
            kernels = [K[i](input_dim=1, lengthscales=lengthscales[i], variance=variances[i], active_dims=[active_dims[i]], name=name+'_{i}'.format(i=i)) for i in range(len(active_dims))]
        return gpflow.kernels.Product(kernels, name=name+'_product')

    def make_mixture(dataset, parent_mixtures = None, name_prefix=''):
        base_kernel_ad = range(dataset[0][0].shape[-1]-1)
        base_kernel_ls = [0.1 for i in base_kernel_ad]
        base_kernel_v = [1.0 for i in base_kernel_ad]
        K_base_1 = get_kernel_product(MR_SE, active_dims=base_kernel_ad, lengthscales=base_kernel_ls, variances=base_kernel_v, name=name_prefix+'MR_SE_BASE_1')

        sat_kernel_ad = [0,1,2, 3, 4]
        sat_kernel_ls = [0.01, 0.1, 0.1, 0.1, 0.1]
        sat_kernel_v = [1.0, 1.0, 1.0, 1.0, 1.0]

        K_base_2 = get_kernel_product(MR_SE, active_dims=sat_kernel_ad, lengthscales=sat_kernel_ls, variances=sat_kernel_v, name=name_prefix+'MR_SE_BASE_2')

        dgp_kernel_ad = [0, 2, 3,  4, 5]
        dgp_kernel_ls = [5.0, 0.1, 0.1, 0.1,  0.1]
        dgp_kernel_v = [1.0, 1.0, 1.0, 1.0, 0.1]

        K_dgp_1 = get_kernel_product(MR_SE, active_dims=dgp_kernel_ad, lengthscales=dgp_kernel_ls, variances=dgp_kernel_v, name=name_prefix+'MR_SE_DGP_1')
        K_parent_1 = None
        

        base_Z = [get_inducing_points(dataset[0][0], 100), get_inducing_points(dataset[1][0], 100)]

        sliced_dataset = np.concatenate([np.expand_dims(dataset[0][0][:, 0, i], -1) for i in dgp_kernel_ad[1:]], axis=1)
        dgp_Z = get_inducing_points(np.concatenate([dataset[0][1], sliced_dataset], axis=1), 100)

        def insert(D, col, i):
            col = np.expand_dims(col, -1)
            d_1 = D[:, :i]
            d_2 = D[:, i:]
            print(d_1.shape)
            print(d_2.shape)
            return np.concatenate([d_1,col,d_2],axis=1)

        print(dgp_Z.shape)
        dgp_Z = insert(dgp_Z, np.ones([dgp_Z.shape[0]]), 1)
        print(dgp_Z.shape)

        dgp_Z = [dgp_Z]
        parent_Z = dgp_Z

        inducing_points = [base_Z, dgp_Z, parent_Z]
        noise_sigmas = [[0.1, 0.1], [1.0], [1.0]]
        minibatch_sizes = [100, 100]

        m = MR_Mixture(
            datasets = dataset, 
            inducing_locations = inducing_points, 
            kernels = [[K_base_1, K_base_2], [K_dgp_1], [K_parent_1]], 
            noise_sigmas = noise_sigmas,
            minibatch_sizes = minibatch_sizes,
            #mixing_weight = MR_DGP_Only(), 
            mixing_weight = MR_Variance_Mixing(), 
            #mixing_weight = MR_Base_Only(i=0), 
            parent_mixtures = parent_mixtures,
            num_samples=1,
            name=name_prefix+"MRDGP"
        )

        return m

    dataset = [[X[1], Y[1]], [X[0], Y[0]]]
    m1 = make_mixture(dataset, name_prefix='m1_')
    tf.local_variables_initializer()
    tf.global_variables_initializer()
    m1.compile()
    m = m1

    #===========================SETUP TRAINING===========================
    tf_session = m.enquire_session()

    def logger(x):    
        if (logger.i % 10) == 0:
            session =  m.enquire_session()
            objective = m.objective.eval(session=session)
            print(logger.i, ': ', objective)

        logger.i+=1
    logger.i = 0

    train=True
    restore=not train
    #m1.train_base_elbo()

    if restore:
        USE_RESULTS_FROM_CLUSTER = False
        DATA_ROOT = ''
        if USE_RESULTS_FROM_CLUSTER:
            list_of_files = (glob.glob('../cluster/*_results'))
            latest_file = max(list_of_files, key=os.path.getctime)
            DATA_ROOT =  latest_file+'/models/'

        saver = tf.train.Saver()
        saver.restore(tf_session, DATA_ROOT+'restore/{name}.ckpt'.format(name=CONFIG['file_prefix']))
    if train:
        opt = AdamOptimizer(0.1)

        def set_objective(_class, objective_str):
            #TODO: should just extend the optimize class at this point
            def minimize(self, model, session=None, var_list=None, feed_dict=None, maxiter=1000, initialize=False, anchor=True, step_callback=None, **kwargs):
                """
                Minimizes objective function of the model.
                :param model: GPflow model with objective tensor.
                :param session: Session where optimization will be run.
                :param var_list: List of extra variables which should be trained during optimization.
                :param feed_dict: Feed dictionary of tensors passed to session run method.
                :param maxiter: Number of run interation.
                :param initialize: If `True` model parameters will be re-initialized even if they were
                    initialized before for gotten session.
                :param anchor: If `True` trained variable values computed during optimization at
                    particular session will be synchronized with internal parameter values.
                :param step_callback: A callback function to execute at each optimization step.
                    The callback should accept variable argument list, where first argument is
                    optimization step number.
                :type step_callback: Callable[[], None]
                :param kwargs: This is a dictionary of extra parameters for session run method.
                """

                if model is None or not isinstance(model, gpflow.models.Model):
                    raise ValueError('The `model` argument must be a GPflow model.')

                opt = self.make_optimize_action(model,
                    session=session,
                    var_list=var_list,
                    feed_dict=feed_dict, **kwargs)

                self._model = opt.model
                self._minimize_operation = opt.optimizer_tensor

                session = model.enquire_session(session)
                with session.as_default():
                    for step in range(maxiter):
                        try:
                            opt()
                            if step_callback is not None:
                                step_callback(step)
                        except (KeyboardInterrupt, SystemExit):
                            print('STOPPING EARLY at {step}'.format(step=step))
                            break

                print('anchoring')
                if anchor:
                    opt.model.anchor(session)

            def make_optimize_tensor(self, model, session=None, var_list=None, **kwargs):
                """
                Make Tensorflow optimization tensor.
                This method builds optimization tensor and initializes all necessary variables
                created by optimizer.
                    :param model: GPflow model.
                    :param session: Tensorflow session.
                    :param var_list: List of variables for training.
                    :param kwargs: Dictionary of extra parameters passed to Tensorflow
                        optimizer's minimize method.
                    :return: Tensorflow optimization tensor or operation.
                """

                print('self: ', self)
                print('model: ', model)

                session = model.enquire_session(session)
                objective = getattr(model, objective_str)
                full_var_list = self._gen_var_list(model, var_list)
                # Create optimizer variables before initialization.
                with session.as_default():
                    minimize = self.optimizer.minimize(objective, var_list=full_var_list, **kwargs)
                    model.initialize(session=session)
                    self._initialize_optimizer(session)
                    return minimize

            setattr(_class, 'minimize', minimize)
            setattr(_class, 'make_optimize_tensor', make_optimize_tensor)

        #m.disable_base_hyperparameters()
        set_objective(AdamOptimizer, 'base_elbo')
        opt.minimize(m, step_callback=logger, maxiter=2000)

        if True:
            m.disable_base_elbo()
            #m.enable_base_hyperparameters()
            set_objective(AdamOptimizer, 'elbo')
            opt.minimize(m, step_callback=logger, maxiter=5000)

            m.enable_base_elbo()
            #m.enable_base_hyperparameters()
            set_objective(AdamOptimizer, 'elbo')
            opt.minimize(m, step_callback=logger, maxiter=2000)

        print('saving')
        saver = tf.train.Saver()
        save_path = saver.save(tf_session, "restore/{name}.ckpt".format(name=CONFIG['file_prefix']))
        print("restore/{name}.ckpt".format(name=CONFIG['file_prefix']))

    prediction(CONFIG, m, _OLD_X, xs, xs_grid, xs_fine_grid, vis_xs)
    

if __name__ == '__main__':
    i = 0
    if len(sys.argv) == 2:
        i = int(sys.argv[1])

    config = get_config()[i]

    if False:
        experiment_config = experiment_config.get_config()

        for name in experiment_config:
            config[name] = experiment_config[name]

    main(config)


