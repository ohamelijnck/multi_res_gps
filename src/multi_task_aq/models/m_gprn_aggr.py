if False:
    import matplotlib
    import matplotlib.pyplot as plt

import sys
import numpy as np
from scipy.cluster.vq import kmeans2


import logging, os
import tensorflow as tf

#disable TF warnings
if True:
    logging.disable(logging.WARNING)
    os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
    tf.logging.set_verbosity(tf.logging.ERROR)


import _gprn as gprn

np.random.seed(0)

import sys
sys.path.append('../')
import experiment_config 
import setup_data


def get_config():
    return  [
        {
            'name': 'mr_gprn',
            'model_name': 'mr_gprn',
            'file_prefix': 'cmgp_gprn_aggr',
            'experiment_id': 0,
            'num_outputs': 1,
            'output_names': ['PM10', 'NO2'],
            'kernel_type': 'sum',
            'order': 'normal',
            'restore': False,
            'z': 20,
            #PLOTTING CONFIG
            'plot_order': 'top',
            'ignore': False,

            'exp_ignore_i': None
        }
    ]

def get_context(CONFIG, X, Y, test=None, r=1):
    num_datasets = X.shape[0]
    num_outputs = Y[0].shape[1]

    context = gprn.context.ContextFactory().create()

    t = not CONFIG['restore'] 
    context.train_flag=t
    context.restore_flag= not t
    context.save_image = False

    context.monte_carlo = False

    context.debug = False
    context.num_outputs = num_outputs
    context.num_latent = 1
    context.num_components = 1

    context.use_diag_covar = False
    context.use_diag_covar_flag = False

    context.train_inducing_points_flag = True

    context.whiten=True
    context.split_optimise=True
    context.jitter = 1e-4
    context.shuffle_seed = 0
    context.num_epochs = 2000
    context.seed = 0
    context.restore_location = 'restore/{name}_{test}_{r}.ckpt'.format(name=CONFIG['file_prefix'], test=test, r=r)

    #inv = lambda x: np.sqrt(x)
    inv = lambda x: np.log(x)
    sig = inv(0.1)
    ls = 0.5

    gprn.kernels.Matern32._id = -1
    gprn.kernels.SE._id = -1
    D = X[0].shape[-1]
    context.kernels = [
        {

            'f': [ gprn.kernels.MR_MATERN_32(num_dimensions=D, length_scale=inv(0.1)) for i in range(context.num_latent) ],
            #'f': [ gprn.kernels.SM(num_dimensions=1, num_components = 20, var_scale=inv(1/10.0), mean_scale=inv(1/0.1)) for i in range(context.num_latent) ],
            'w': [[gprn.kernels.MR_SE(num_dimensions=D, length_scale=inv(0.1)) for j in range(context.num_latent)] for i in range(context.num_outputs)]
            #'w': [[gprn.kernels.SM(num_dimensions=1, num_components = 20, var_scale=inv(1/10.0), mean_scale=inv(1/0.1)) for j in range(context.num_latent)] for i in range(context.num_outputs)]
        }, #r=0
    ]
    context.noise_sigmas = [
        #[sigma_arr, train_flag]
        [[inv(0.1) for i in range(context.num_outputs)], True],
        [[inv(0.1) for i in range(context.num_outputs)], True]
    ]

    return context

def get_dataset(CONFIG, X, Y, z_r):
    data = gprn.Dataset()
    num_data_sources = X.shape[0]

    for i in range(num_data_sources):
        x = np.array(X[i])
        y = np.array(Y[i])

        M = x.shape[1]

        b = 10
        b = b if b < x.shape[0] else y.shape[0]

        if CONFIG['exp_ignore_i'] is not None:
            if i == CONFIG['exp_ignore_i']: continue

        data.add_source_dict({
            'M': M,
            'x': x,
            'y': y,
            #'z': x,
            'batch_size': b,
            #'active_tasks': [[1], [0]]
            'active_tasks': [[1], [0]]
        })

    data.add_inducing_points(z_r);
    return data

def main(CONFIG, return_m=False, force_restore=False):
    if True:
        run(CONFIG, test = CONFIG['tests'][CONFIG['vis_test']], r = CONFIG['vis_iter'], return_m=return_m, force_restore=force_restore)
    else:
        for TEST in CONFIG['tests']:
            for r in TEST['resolutions']:
                #print(r)
                run(CONFIG, test = TEST, r = r, return_m=return_m, force_restore=force_restore)

def run(CONFIG, test=None, r=1, return_m=False, force_restore=False):
    test_id = test['id']
    print('test_id:', test_id)
    tf.reset_default_graph()
    print(CONFIG)
    CONFIG['num_layers'] = 2
    num_z = CONFIG['z']

    X = np.load('../data/data_x_{test_id}_{r}.npy'.format(test_id=test_id, r=r), allow_pickle=True)
    Y = np.load('../data/data_y_{test_id}_{r}.npy'.format(test_id=test_id, r=r), allow_pickle=True)


    print(np.min(X[0]), np.min(X[1]))
    print(np.max(X[0]), np.max(X[1]))

    #add empty Y columns 
    #Y[0] is task 1
    Y[0] = np.concatenate([np.empty(Y[0].shape), Y[0]], axis=1)
    Y[0][:, 0] = np.nan

    Y[1] = np.concatenate([Y[1], np.empty(Y[1].shape)], axis=1)
    Y[1][:, 1] = np.nan

    XS = np.load('../data/data_xs_{test_id}.npy'.format(test_id=test_id), allow_pickle=True)
    XS_VIS = np.load('../data/data_x_vis_{test_id}.npy'.format(test_id=test_id), allow_pickle=True)

    rs = lambda x: x.reshape(x.shape[0]*x.shape[1], x.shape[2])

    XS = rs(XS)
  
    i = 0
    x = X[i].reshape([X[i].shape[0]*X[i].shape[1], X[i].shape[2]])
    if num_z is None or num_z > x.shape[0]:
        z_x = x
    else:
        z_x = kmeans2(x,num_z, minit='points')[0]

    i = 0
    x = X[i].reshape([X[i].shape[0]*X[i].shape[1], X[i].shape[2]])
    if num_z is None or num_z > x.shape[0]:
        z_r = x
    else:
        z_r = kmeans2(x, num_z, minit='points')[0] 

    #z_r = X[i].reshape([X[i].shape[0]*X[i].shape[1], 1])

    Z = np.array([z_r, z_x])

    dataset = get_dataset(CONFIG, X, Y, z_r)
    context =  get_context(CONFIG, X, Y, test=test_id, r=1)

    #elbo_model =  gprn.models.GPRN_Aggr
    elbo_model =  gprn.models.GPRN_Aggr

    #context.parameters = gprn.composite_corrections.MagnitudeCorrection

    m = gprn.GPRN(
        model = elbo_model,
        context = context,
        data = dataset
    )

    m.optimise(context.train_flag, context.restore_flag)

    df_all, df_train, df_test = setup_data.get_site_1(test, root='../')

    y_pred, y_var = m.predict(rs(X[1]), r=1)

    #get task 0
    target_task = 0
    y_pred = np.expand_dims(y_pred[:, target_task], -1)
    y_var = np.expand_dims(y_var[:, target_task], -1)

    ys_pred, ys_var = m.predict(XS, r=1)
    ys_pred = np.expand_dims(ys_pred[:, target_task], -1)
    ys_var = np.expand_dims(ys_var[:, target_task], -1)

    y_vis_pred, y_vis_var = m.predict(rs(XS_VIS), r=1)
    y_vis_pred = np.expand_dims(y_vis_pred[:, target_task], -1)
    y_vis_var = np.expand_dims(y_vis_var[:, target_task], -1)


    if False:
        y_pred = setup_data.denormalise_wrt(y_pred, np.array(df_train['pm10']), sphere_flag=True)
        ys_pred = setup_data.denormalise_wrt(ys_pred, np.array(df_train['pm10']), sphere_flag=True)
        y_vis_pred = setup_data.denormalise_wrt(y_vis_pred, np.array(df_train['pm10']), sphere_flag=True)


    pred_y = np.concatenate([y_pred, y_var], axis=1)
    pred_ys = np.concatenate([ys_pred, ys_var], axis=1)
    pred_y_vis = np.concatenate([y_vis_pred, y_vis_var], axis=1)

    std = np.nanstd(np.array(df_train['pm10']), axis=0)
    y_var = y_var*std**2
    ys_var = ys_var*std**2
    y_vis_var = y_vis_var*std**2

    np.save('../results/{prefix}_y_{test_id}_{r}'.format(prefix=CONFIG['file_prefix'], r=r, test_id=test_id), pred_y)
    np.save('../results/{prefix}_ys_{test_id}_{r}'.format(prefix=CONFIG['file_prefix'], r=r, test_id=test_id), pred_ys)
    np.save('../results/{prefix}_y_vis_{test_id}_{r}'.format(prefix=CONFIG['file_prefix'], r=r, test_id=test_id), pred_y_vis)
 

if __name__ == '__main__':
    i = 0
    if len(sys.argv) == 2:
        i = int(sys.argv[1])

    config = get_config()[i]
    experiment_config = experiment_config.get_config()

    for name in experiment_config:
        config[name] = experiment_config[name]

    main(config)

