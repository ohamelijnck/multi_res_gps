import matplotlib 
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from SpaceTimeVisualise import SpaceTimeVisualise

import glob
import os

PLOT_FINE_GRID = False

USE_RESULTS_FROM_CLUSTER = True

DATA_ROOT = ''
if USE_RESULTS_FROM_CLUSTER:
    list_of_files = (glob.glob('cluster/*_results'))
    latest_file = max(list_of_files, key=os.path.getctime)
    #latest_file = 'cluster/27_11_2019_18_00_00_results'
    DATA_ROOT =  latest_file+'/'

if True:
    #file_prefix= 'cmgp_aggr'
    #file_prefix= 'cmgp_vbagg'
    file_prefix= 'cmgp_gprn_aggr'
    #file_prefix= 'center_point_baseline'
    #file_prefix= 'cmgp_aggr_sat_only'
    #file_prefix= 'cmgp_aggr_laqn_only'
    #file_prefix= 'cmgp_gprn_aggr_corrected'
    PLOT_FULL_GRID = False
    PLOT_FINE_GRID = True
else:
    file_prefix= 'cmgp_dgp'
    file_prefix= 'cmgp_dgp_expert_1'
    PLOT_FULL_GRID = False
    PLOT_FINE_GRID = True

#============================GET DATA=================================
y_pred = np.load(DATA_ROOT+'results/{file_prefix}_y.npy'.format(file_prefix=file_prefix)) #ys, ys_var
ys_pred = np.load(DATA_ROOT+'results/{file_prefix}_ys.npy'.format(file_prefix=file_prefix)) #ys, ys_var
raw_x = pd.read_csv('data/data_with_features/processed_data_x.csv') #id, time, x, y, covs
raw_sat_x = pd.read_csv('data/data_with_features/processed_data_sat_x.csv') #id, time, x, y, covs
raw_xs = pd.read_csv('data/data_with_features/processed_data_xs.csv') #id, time, x, y, covs

print(raw_x.shape, ', ', y_pred.shape)
print(raw_xs.shape, ', ', ys_pred.shape)

print(raw_x.columns)
val_col = 'val'
column_names = ['id', 'epoch', 'lon', 'lat', 'datetime', val_col]

print('y_pred.shape: ', y_pred.shape)
print(raw_x.shape)
train_df = np.concatenate([np.array(raw_x[column_names]), y_pred], axis=1)
test_df = np.concatenate([np.array(raw_xs[column_names]), ys_pred], axis=1)

train_df = np.concatenate([train_df,test_df], axis=0)
train_sat_df =np.array(raw_sat_x[column_names])


if PLOT_FULL_GRID:
    ys = np.load(DATA_ROOT+'results/{file_prefix}_ys_grid.npy'.format(file_prefix=file_prefix)) #ys, ys_var
    raw_grid_xs = pd.read_csv('data/data_with_features/processed_data_x_test_grid.csv') #id, time, x, y, covs
    #grid_test_df = np.concatenate([np.array(raw_grid_xs[['id', 'epoch', 'lon', 'lat', 'datetime']]), np.expand_dims(np.repeat([None], ys.shape[0]), -1), ys], axis=1)
    #ys = np.expand_dims(np.array(raw_grid_xs['avg_ratio_avg_100']), -1)
else:
    print(file_prefix)
    if PLOT_FINE_GRID:
        ys = np.load(DATA_ROOT+'results/{file_prefix}_ys_small_fine_grid.npy'.format(file_prefix=file_prefix)) #ys, ys_var
        raw_grid_xs = pd.read_csv('data/data_with_features/processed_data_x_test_small_fine_grid.csv') #id, time, x, y, covs
    else:
        ys = np.load(DATA_ROOT+'results/{file_prefix}_ys_small_grid.npy'.format(file_prefix=file_prefix)) #ys, ys_var
        raw_grid_xs = pd.read_csv('data/data_with_features/processed_data_x_test_small_grid.csv') #id, time, x, y, covs
        print(raw_grid_xs.shape, ys.shape)

ys_var = np.expand_dims(ys[:, 1], -1)
ys = np.expand_dims(ys[:, 0], -1)
observed = np.expand_dims(np.repeat([None], ys.shape[0]), -1)
grid_test_df = np.concatenate([np.array(raw_grid_xs[['id', 'epoch', 'lon', 'lat', 'datetime']]), observed, ys, ys_var], axis=1)


if False:
    data_save = np.concatenate([np.array(raw_grid_xs[['id', 'x', 'y', 'datetime']]), ys], axis=1)
    df = pd.DataFrame(data=data_save, columns=['id', 'x', 'y', 'datetime', 'no2', 'var'])
    df.to_csv('results/results_df.csv', index=False)

    data_df = raw_grid_xs


visualise = SpaceTimeVisualise(train_df, grid_test_df, sat_df = train_sat_df, test_start = np.min(raw_xs['epoch']))
#visualise = SpaceTimeVisualise(train_df, grid_test_df, sat_df = None, test_start = np.min(raw_xs['epoch']))
visualise.show()
