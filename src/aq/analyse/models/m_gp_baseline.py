import matplotlib
import matplotlib.pyplot as plt

import _gprn as gprn

import sys
import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans2
import tensorflow as tf


np.random.seed(0)


def get_config():
    return  [
        {
            'name': 'CENTER-POINT-BASELINE',
            'pretty_name': 'Center Point Baseline',
            'model_name': 'center_baseline' ,
            'file_prefix': 'center_point_baseline',
            'experiment_id': 0,
            'config_id': 0,
            'num_outputs': 1,
            'output_names': ['R=1', 'R-Aggr'],
            'restore': False,
            'ignore': True,
            'plot_fine_grid': True,
            'plot_fine_grid_order': 2,
            #PLOTTING CONFIG
            'plot_order': 'top',

            #EXP CONFIG
            'exp_ignore_i': None
        },
    ]


def get_context(CONFIG, X, Y):
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
    context.jitter = 1e-5
    context.shuffle_seed = 0
    context.num_epochs = 2000
    context.seed = 0
    context.restore_location = 'restore/{name}.ckpt'.format(name=CONFIG['file_prefix'])

    #inv = lambda x: np.sqrt(x)
    inv = lambda x: np.log(x)
    sig = inv(0.1)
    ls = 0.5

    gprn.kernels.Matern32._id = -1
    gprn.kernels.SE._id = -1
    context.kernels = [
        {
            'f': [gprn.kernels.SE(num_dimensions=X[0].shape[-1], length_scale=inv(0.1)) for i in range(context.num_latent)],
        }, #r=0
    ]
    context.noise_sigmas = [
        #[sigma_arr, train_flag]
        [[inv(0.1) for i in range(context.num_outputs)], True],
        [[inv(0.1) for i in range(context.num_outputs)], True]
    ]

    return context

def denormalise_wrt(x, y, sphere_flag=False):
    sphere = 1.0
    if sphere_flag:
        sphere = np.nanstd(y, axis=0)

    return (x*sphere)+np.nanmean(y,axis=0)

def denormalise(pred_y):
    raw_x = pd.read_csv('data/data_x.csv') #id, time, x, y, covs
    raw_train_y = np.array(raw_x['val'])
    return denormalise_wrt(pred_y, raw_train_y, sphere_flag=False)

def prediction(CONFIG, m, X, xs, xs_grid, xs_small_grid, xs_small_fine_grid, plot_grid=False):
    rs = lambda x: x.reshape([x.shape[0]*x.shape[1], x.shape[2]])

    r = 0
    if CONFIG['exp_ignore_i'] == None:
        r = 1

    y_0, y_var_0 = m.predict(rs(X[1]), r=r)
    ys_0, ys_var_0 = m.predict(xs, r=r)

    #y_0 = np.expand_dims(denormalise(y_0[:, 0]), -1)
    #ys_0 = np.expand_dims(denormalise(ys_0[:, 0]), -1)


    pred_y = np.concatenate([y_0, y_var_0], axis=1)
    pred_ys = np.concatenate([ys_0, ys_var_0], axis=1)

    np.save('results/{prefix}_y'.format(prefix=CONFIG['file_prefix']), pred_y)
    np.save('results/{prefix}_ys'.format(prefix=CONFIG['file_prefix']), pred_ys)

    if plot_grid:
        ys_grid_0, ys_var_grid_0 = m.predict(xs_small_grid, r=r)
        #ys_grid_0 = np.expand_dims(denormalise(ys_grid_0[:, 0]), -1)
        pred_ys_grid = np.concatenate([ys_grid_0, ys_var_grid_0], axis=1)
        np.save('results/{prefix}_ys_small_grid'.format(prefix=CONFIG['file_prefix']), pred_ys_grid)

        ys_fine_grid_0, ys_var_fine_grid_0 = m.predict(xs_small_fine_grid, r=r)
        pred_ys_fine_grid = np.concatenate([ys_fine_grid_0, ys_var_fine_grid_0], axis=1)
        np.save('results/{prefix}_ys_small_fine_grid'.format(prefix=CONFIG['file_prefix']), pred_ys_fine_grid)


def get_dataset(CONFIG, X, Y, z_r):
    data = gprn.Dataset()
    num_data_sources = X.shape[0]

    for i in range(num_data_sources):
        x = np.array(X[i])
        y = np.array(Y[i])

        M = x.shape[1]
        print(M)

        b = 300
        b = b if b < x.shape[0] else None

        if CONFIG['exp_ignore_i'] is not None:
            if i == CONFIG['exp_ignore_i']: continue

        data.add_source_dict({
            'M': M,
            'x': x,
            'y': y,
            #'z': x,
            'batch_size': b
        })

    data.add_inducing_points(z_r);
    return data


def main(CONFIG, return_m=False, force_restore=False):
    X = np.load('data/data_with_features/processed_x_train.npy', allow_pickle=True) #[SAT, LAQN]
    Y = np.load('data/data_with_features/processed_y_train.npy', allow_pickle=True)

    xs = np.load('data/data_with_features/processed_x_test.npy', allow_pickle=True) #100 , 50
    xs_grid = np.load('data/data_with_features/processed_x_test_grid.npy', allow_pickle=True) #100 , 50
    xs_small_grid = np.load('data/data_with_features/processed_x_test_small_grid.npy', allow_pickle=True)
    xs_small_fine_grid = np.load('data/data_with_features/processed_x_test_small_fine_grid.npy', allow_pickle=True)

    X[0] = np.array(X[0])
    X[1] = np.array(X[1])
    Y[0] = np.array(Y[0])
    Y[1] = np.array(Y[1])



    center_mean = lambda x: np.expand_dims(np.mean(x, axis=1), 1)
    Y[0] = center_mean(Y[0])
    X[0] = center_mean(X[0])

    print(X[0].shape)
    print(X[1].shape)
    print(Y[0].shape)
    print(Y[1].shape)


    #ys = np.load('data/data_ys.npy')

    num_z = 400

    i = 1
    if num_z is None:
        z_r = X[i].reshape([X[i].shape[0]*X[i].shape[1], X[i].shape[2]])
    else:
        z_r = kmeans2(X[i].reshape([X[i].shape[0]*X[i].shape[1], X[i].shape[2]]), num_z, minit='points')[0] 
    #z_r = X[i].reshape([X[i].shape[0]*X[i].shape[1], X[i].shape[2])

    dataset = get_dataset(CONFIG, X, Y, z_r)
    context = get_context(CONFIG, X, Y)

    elbo_model =  gprn.models.GPAggr

    #context.parameters = gprn.composite_corrections.MagnitudeCorrection

    m = gprn.GPRN(
        model = elbo_model,
        context = context,
        data = dataset
    )

    m.optimise(context.train_flag, context.restore_flag)

    prediction(CONFIG, m, X, xs, xs_grid, xs_small_grid, xs_small_fine_grid, plot_grid=True)

if __name__ == '__main__':
    i = 0
    if len(sys.argv) == 2:
        i = int(sys.argv[1])

    main(get_config()[i])


