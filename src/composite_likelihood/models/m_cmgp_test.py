import matplotlib
import matplotlib.pyplot as plt

import gprn

import sys
import numpy as np
from scipy.cluster.vq import kmeans2
import tensorflow as tf


np.random.seed(0)


def get_config():
    return  [
        {
            'name': 'GP-AGGR-1',
            'file_prefix': 'gp_aggr',
            'model_name': 'vbagg',
            'experiment_id': 0,
            'num_outputs': 1,
            'output_names': ['R=1', 'R-Aggr'],
            'restore': False,
            'ignore': True,
            #PLOTTING CONFIG
            'plot_order': 'top',
            #'plot_var_type': 'error_bar',
            'composite_w': False

        },
        {
            'name': 'GP-AGGR-CORRECTED',
            'file_prefix': 'gp_aggr_corrected',
            'model_name': 'mr_gp',
            'experiment_id': 0,
            'num_outputs': 1,
            'output_names': ['R=1', 'R-Aggr'],
            'restore': False,
            'ignore': True,
            #PLOTTING CONFIG
            'plot_order': 'any',
            #'plot_var_type': 'error_bar',
            'composite_w': True

        }
    ]

def get_dataset(X, Y, z_r):
    data = gprn.Dataset()
    num_data_sources = X.shape[0]

    #for i in range(num_data_sources):
    for i in [0, 1]:
        x = X[i]
        y = Y[i]
        print('dataset: ', i, ' ',  x.shape)

        M = x.shape[1]
        print(x.shape)
        exit()

        data.add_source_dict({
            'M': M,
            'x': x,
            'y': y,
            'z': z_r,
            'batch_size': None

        })


    #data.add_inducing_points(z_r);
    return data

def get_context(CONFIG, X, Y):
    num_datasets = X.shape[0]
    num_outputs = Y[0].shape[1]

    context = gprn.context.ContextFactory().create()

    t = True 
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

    context.train_inducing_points_flag = False
    context.plot_posterior = False

    context.whiten=True
    context.jitter = 1e-4
    context.shuffle_seed = 0
    context.num_epochs = 10000
    context.seed = 0
    context.restore_location = 'restore/{name}.ckpt'.format(name=CONFIG['file_prefix'])

    context.save_parameters = True
    context.save_parameters_location = 'saved_params.csv'

    #inv = lambda x: np.sqrt(x)
    inv = lambda x: np.log(x)
    sig = inv(0.1)
    ls = 0.5

    gprn.kernels.Matern32._id = -1
    gprn.kernels.SE._id = -1
    context.kernels = [
        {
            'f': [gprn.kernels.SE(num_dimensions=1, length_scale=inv(0.1)) for i in range(context.num_latent)],
        }, #r=0
    ]
    context.noise_sigmas = [
        #[sigma_arr, train_flag]
        [[inv(0.1) for i in range(context.num_outputs)], True] for j in range(num_datasets)
    ]

    return context

def prediction(CONFIG, m, X, xs):
    rs = lambda x: x.reshape([x.shape[0]*x.shape[1], x.shape[2]])

    #r = 1
    r = 0
    y_0, y_var_0 = m.predict(rs(X[1]), r=r)
    ys_0, ys_var_0 = m.predict(rs(xs), r=r)

    y_1, y_var_1 = m.predict(rs(X[1]), r=r)
    ys_1, ys_var_1 = m.predict(rs(xs), r=r)

    pred_y = np.concatenate([y_0, y_var_0, y_1,y_var_1], axis=1)
    pred_ys = np.concatenate([ys_0, ys_var_0, ys_1,ys_var_1], axis=1)

    np.save('../results/{prefix}_y'.format(prefix=CONFIG['file_prefix']), pred_y)
    np.save('../results/{prefix}_ys'.format(prefix=CONFIG['file_prefix']), pred_ys)



def main(CONFIG, return_m=False, force_restore=False):
    X = np.load('../data/data_x.npy', allow_pickle=True)
    Y = np.load('../data/data_y.npy', allow_pickle=True)


    xs = np.load('../data/data_xs.npy', allow_pickle=True) #100 , 50
    ys = np.load('../data/data_ys.npy', allow_pickle=True)

    print(X[0].shape)
    print(X[1].shape)


    i = 0
    #z_r = kmeans2(X[i].reshape([X[i].shape[0]*X[i].shape[1], 1]), 10, minit='points')[0] 
    z_r = X[i].reshape([X[i].shape[0]*X[i].shape[1], 1])

    dataset = get_dataset(X, Y, z_r)
    context = get_context(CONFIG, X, Y)

    elbo_model =  gprn.models.GPAggr

    if CONFIG['composite_w']:
        #context.parameters = gprn.composite_corrections.MagnitudeCorrection
        #context.parameters = gprn.composite_corrections.CurvatureCorrection
        pass

    m = gprn.GPRN(
        model = elbo_model,
        context = context,
        data = dataset
    )

    m.optimise(context.train_flag, context.restore_flag)

    prediction(CONFIG, m, X, xs)

if __name__ == '__main__':
    i = 0
    if len(sys.argv) == 2:
        i = int(sys.argv[1])

    main(get_config()[i])
