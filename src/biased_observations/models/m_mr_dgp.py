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
from _mr_dgp import MR_SE, MR_Linear, MR_KERNEL_PRODUCT, MR_MATERN_32
from _mr_dgp.mr_mixing_weights import MR_Average_Mixture, MR_Base_Only, MR_DGP_Only, MR_Variance_Mixing, MR_Variance_Mixing_1

from scipy.cluster.vq import kmeans2

import matplotlib.pyplot as plt

import sys
import os
import glob

np.random.seed(0)

def get_aggr_data(x, y):
    res = []
    for i in y:
        res.append(np.repeat(i, x.shape[1]))
    res = np.array(res).flatten()
    x = x.flatten()
    return x, res

def get_config():
    return  [
        {
            'name': 'CMGP-DGP-EXPERT',
            'file_prefix': 'mr_dgp',
            'model_name': 'mr_dgp',
            'experiment_id': 0,
            'num_outputs': 3,
            'output_names': ['R=1', 'R=2', 'R=3'],
            'kernel_type': 'sum',
            'order': 'normal',
            #'restore': False,
            'train': False,
            'use_f_z_pref': False,
            'f_only' : False,
            'z': None,
            'hierarchy_type': 'tree',
            #PLOTTING CONFIG
            'plot_order': 'top',
            'v_z_order': -11
        },

#        {
#            'name': 'MR-DGP-CASCADE',
#            'file_prefix': 'dgp_cascade',
#            'model_name': 'cascade_dgp_baseline',
#            'experiment_id': 0,
#            'num_outputs': 1,
#            'output_names': ['R=1'],
#            'kernel_type': 'sum',
#            'order': 'normal',
#            'restore': True,
#            'z': 30,
#            'f_only': True,
#            'hierarchy_type': 'dgp_cascade',
#            #PLOTTING CONFIG
#            'plot_order': 'any'
#        }
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

def get_sample_mean_var(ys, vs):
    ys = ys[:, :, 0, :]
    vs = vs[:, :, 0, :]
    mu = np.mean(ys, axis=0)
    sig = np.mean(vs+ys**2, axis=0)-np.mean(ys, axis=0)**2
    return mu, sig


def main(CONFIG, return_m=False, force_restore=False):
    print(CONFIG)
    CONFIG['num_layers'] = 2
    num_z = CONFIG['z']

    #[20, 5, 1], [20, 5, 1], [50, 1, 1]
    X = np.load('../data/data_x.npy', allow_pickle=True)
    Y = np.load('../data/data_y.npy', allow_pickle=True)


    xs = np.load('../data/data_xs.npy', allow_pickle=True) #200 , 1, 1

    #===========================SETUP MODEL===========================
    def get_inducing_points(X, num_z=None):
        if len(X.shape) == 3:
            X = X.reshape([X.shape[0]*X.shape[1], X.shape[2]])

        if num_z is None or num_z > X.shape[0]:
            Z = X
        else:
            Z = kmeans2(X, num_z, minit='points')[0] 
        return Z

    def get_kernel_product(K, active_dims=[0], lengthscales=None, variances=[1.0], name=''):
        if not isinstance(K, list):
            K = [K for i in range(len(active_dims))]

        if lengthscales is None:
            kernels = [K[i](input_dim=1, variance=variances[i], active_dims=[active_dims[i]], name=name+'_{i}'.format(i=i)) for i in range(len(active_dims))]
        else:
            kernels = [K[i](input_dim=1, lengthscales=lengthscales[i], variance=variances[i], active_dims=[active_dims[i]], name=name+'_{i}'.format(i=i)) for i in range(len(active_dims))]
        return gpflow.kernels.Product(kernels, name=name+'_product')

    def make_mixture(dataset, parent_mixtures = None, masks=None, name_prefix=''):
        base_kernel_ad = list(range(dataset[0][0].shape[-1]))
        base_kernel_ls = [0.1 for i in base_kernel_ad]
        base_kernel_v = [1.0 for i in base_kernel_ad]
        K_base_1 = get_kernel_product(MR_SE, active_dims=base_kernel_ad, lengthscales=base_kernel_ls, variances=base_kernel_v, name=name_prefix+'MR_SE_BASE_1')
        K_base_2 = get_kernel_product(MR_SE, active_dims=base_kernel_ad, lengthscales=base_kernel_ls, variances=base_kernel_v, name=name_prefix+'MR_SE_BASE_2')
        K_base_3 = get_kernel_product(MR_SE, active_dims=base_kernel_ad, lengthscales=base_kernel_ls, variances=base_kernel_v, name=name_prefix+'MR_SE_BASE_3')

        dgp_kernel_ad = [0]
        dgp_kernel_ls = [5.0]
        dgp_kernel_v = [1.0]

        K_dgp_1 = get_kernel_product(MR_SE, active_dims=dgp_kernel_ad, lengthscales=dgp_kernel_ls, variances=dgp_kernel_v, name=name_prefix+'MR_SE_DGP_1')
        K_dgp_2 = get_kernel_product(MR_SE, active_dims=dgp_kernel_ad, lengthscales=dgp_kernel_ls, variances=dgp_kernel_v, name=name_prefix+'MR_SE_DGP_2')
        #K_dgp_1 = get_kernel_product(MR_Linear, active_dims=dgp_kernel_ad, variances=dgp_kernel_v, name=name_prefix+'MR_SE_DGP_1')
        #K_dgp_2 = get_kernel_product(MR_Linear, active_dims=dgp_kernel_ad, variances=dgp_kernel_v, name=name_prefix+'MR_SE_DGP_2')
        K_parent_1 = None

        num_inducing = None
        base_Z = [get_inducing_points(dataset[0][0], num_inducing), get_inducing_points(dataset[1][0], num_inducing), get_inducing_points(dataset[2][0], num_inducing)]

        if len(dgp_kernel_ad) > 1:
            sliced_dataset = np.concatenate([np.expand_dims(dataset[0][0][:, 0, i-1], -1) for i in dgp_kernel_ad[1:]], axis=1)
            print(sliced_dataset)
            dgp_Z = get_inducing_points(np.concatenate([dataset[0][1], sliced_dataset], axis=1), num_inducing)

            sliced_dataset_2 = np.concatenate([np.expand_dims(dataset[0][0][:, 0, i-1], -1) for i in dgp_kernel_ad[1:]], axis=1)
            dgp_Z_2 = get_inducing_points(np.concatenate([dataset[0][1], sliced_dataset], axis=1), num_inducing)
        else:
            #num_inducing=10
            dgp_Z = get_inducing_points(dataset[0][1], num_inducing)
            dgp_Z_2 = get_inducing_points(dataset[0][1], num_inducing)

        def insert(D, col, i):
            col = np.expand_dims(col, -1)
            d_1 = D[:, :i]
            d_2 = D[:, i:]

            return np.concatenate([d_1,col,d_2],axis=1)

        dgp_Z = insert(dgp_Z, np.ones([dgp_Z.shape[0]]), 1)
        dgp_Z_2 = insert(dgp_Z_2, np.ones([dgp_Z_2.shape[0]]), 1)

        dgp_Z = [dgp_Z, dgp_Z_2]
        parent_Z = dgp_Z

        inducing_points = [base_Z, dgp_Z, parent_Z]
        noise_sigmas = [[0.1, 0.01, 0.01], [0.0001, 0.0001], [1.0]]
        minibatch_sizes = [dataset[0][0].shape[0], dataset[1][0].shape[0], dataset[2][0].shape[0]]

        m = MR_Mixture(
            datasets = dataset, 
            inducing_locations = inducing_points, 
            kernels = [[K_base_1, K_base_2, K_base_3], [K_dgp_1, K_dgp_2], [K_parent_1]], 
            noise_sigmas = noise_sigmas,
            minibatch_sizes = minibatch_sizes,
            #mixing_weight = MR_Base_Only(i=1), 
            mixing_weight = MR_Variance_Mixing(), 
            #mixing_weight = MR_DGP_Only(i=1), 
            parent_mixtures = parent_mixtures,
            masks=masks,
            num_samples=100,
            name=name_prefix+"MRDGP"
        )

        return m

    masks = []
    if True:
        for i in range(2):
            x1_min = np.min(X[i])
            x1_max = np.max(X[i])
            x3_min = np.min(X[2])
            x3_max = np.max(X[2])

            overlap_min = np.max([x1_min, x3_min])
            overlap_max = np.min([x1_max, x3_max])

            y_mask = ((X[2][:, 0, :] >= overlap_min) & (X[2][:, 0, :] <= overlap_max))
            y_mask = np.all(y_mask, axis=1)
            y_mask = y_mask
            masks.append(y_mask)

        masks.append(None) #Y[2] does not need a mask
        if False:
            j =1
            _x,_y = get_aggr_data(X[j],Y[j])
            plt.plot(_x, _y)
            plt.plot(np.ma.masked_array(X[2][:,0,:],mask=~masks[j]), np.ma.masked_array(Y[2],mask=~masks[j]))
            plt.show()
            print(masks[0])
            exit()


    dataset = [[X[2], Y[2]], [X[1], Y[1]],[X[0], Y[0]]]
    masks = [masks[2], masks[1],masks[0]]
    m1 = make_mixture(dataset, masks=masks, name_prefix='m1_')
    tf.local_variables_initializer()
    tf.global_variables_initializer()
    m1.compile()
    m = m1

    num_samples = 100

   #===========================SETUP TRAINING===========================
    tf_session = m.enquire_session()

    def logger(x):    
        if (logger.i % 10) == 0:
            session =  m.enquire_session()
            objective = m.objective.eval(session=session)
            print(logger.i, ': ', objective)

        logger.i+=1
    logger.i = 0

    train=CONFIG['train']
    restore=not train
    #restore=True
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

        if False:
            m.disable_base_hyperparameters()
            set_objective(AdamOptimizer, 'base_elbo')
            opt.minimize(m, step_callback=logger, maxiter=1000)

            if True:
                m.disable_base_elbo()
                #m.enable_base_hyperparameters()
                m.disable_dgp_hyperparameters()
                set_objective(AdamOptimizer, 'elbo')
                opt.minimize(m, step_callback=logger, maxiter=1000)

            if False:
                m.enable_base_elbo()
                m.enable_dgp_elbo()
                set_objective(AdamOptimizer, 'elbo')
                opt.minimize(m, step_callback=logger, maxiter=500)

        if True:
            m.set_base_gp_noise(False)
            set_objective(AdamOptimizer, 'base_elbo')
            opt.minimize(m, step_callback=logger, maxiter=2000)

            m.disable_base_elbo()
            m.set_dgp_gp_noise(False)
            set_objective(AdamOptimizer, 'elbo')
            opt.minimize(m, step_callback=logger, maxiter=2000)

            #m.enable_base_elbo()
            m.set_dgp_gp_noise(True)
            m.set_base_gp_noise(True)
            set_objective(AdamOptimizer, 'elbo')
            opt.minimize(m, step_callback=logger, maxiter=2000)

        if False:
            #m.enable_base_elbo()
            set_objective(AdamOptimizer, 'elbo')
            opt.minimize(m, step_callback=logger, maxiter=1000)


        print('saving')
        saver = tf.train.Saver()
        save_path = saver.save(tf_session, 'restore/{name}.ckpt'.format(name=CONFIG['file_prefix']))
        #saver.restore(tf_session, 'restore/{name}_{test}_{r}.ckpt'.format(name=CONFIG['file_prefix'], test=test_id, r=r))
        print("restore/{name}.ckpt".format(name=CONFIG['file_prefix']))


    #X_mu, X_sig = m.predict_y_experts(X[1][:, 0, :], num_samples)
    X_mu, X_sig = batch_predict(m, X[-1][:, 0, :], num_samples)
    X_mu, X_sig = get_sample_mean_var(X_mu, X_sig)
    pred_y = np.concatenate([X_mu, X_sig], axis=1)
    np.save('../results/{prefix}_y'.format(prefix=CONFIG['file_prefix']), pred_y)

    xs_mu, xs_sig = batch_predict(m, xs[:, 0, :], num_samples)
    xs_mu, xs_sig = get_sample_mean_var(xs_mu, xs_sig)
    pred_ys = np.concatenate([xs_mu, xs_sig], axis=1)
    np.save('../results/{prefix}_ys'.format(prefix=CONFIG['file_prefix']), pred_ys)


if __name__ == '__main__':
    i = 0
    if len(sys.argv) == 2:
        i = int(sys.argv[1])

    main(get_config()[i])
