import matplotlib 
matplotlib.use("Qt5agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from SpaceTimeVisualise import SpaceTimeVisualise

if True:
    file_prefix= 'cmgp_aggr'
    PLOT_GRID = True
else:
    file_prefix= 'cmgp_dgp'
    PLOT_GRID = False

#============================GET DATA=================================
y_pred = np.load('results/{file_prefix}_y.npy'.format(file_prefix=file_prefix)) #ys, ys_var
ys_pred = np.load('results/{file_prefix}_ys.npy'.format(file_prefix=file_prefix)) #ys, ys_var
raw_x = pd.read_csv('data/processed_data_x.csv') #id, time, x, y, covs
raw_xs = pd.read_csv('data/processed_data_xs.csv') #id, time, x, y, covs
raw_grid_xs = pd.read_csv('data/processed_data_x_test_grid.csv') #id, time, x, y, covs

xs = raw_grid_xs[raw_grid_xs['epoch']==np.min(raw_grid_xs['epoch'])]

cols = xs.columns[7:-1]

val_col = 'total_road_length_1c'


s = xs
min_x = np.min(s['lon'])
min_y = np.min(s['lat'])
max_x = np.max(s['lon'])
max_y = np.max(s['lat'])

x_train, y_train  = np.array(s['lon']).astype(np.float32), np.array(s['lat']).astype(np.float32)
s = np.c_[x_train, y_train]
grid_index = np.lexsort((s[:, 0], s[:, 1]))
s = s[grid_index, :]
n = int(np.sqrt(x_train.shape[0]))

def plot(ax, xs, val_col):
    z_train =np.array(xs[val_col]).astype(np.float32) 
    z_train = z_train[grid_index]
    z_train = (z_train).reshape(n, n)
    grid_plot = ax.imshow(z_train, origin='lower',  aspect='auto', extent=[min_x, max_x, min_y, max_y])
    ax.set_title(val_col, fontsize=5)

I = int(np.ceil(np.sqrt(len(cols))))
J = int(I)
fig, ax = plt.subplots(I, J, sharex='col', sharey='row')

cur_id = 0
for i in range(I):
    for j in range(J):
        a = ax[i][j]
        if cur_id >= len(cols): continue
        plot(a, xs, cols[cur_id])
        cur_id = cur_id + 1

plt.show()
exit()


print(raw_x.columns)
val_col = 'total_road_length_1c'
val_col = 'max_building_height_100'

column_names = ['id', 'epoch', 'lon', 'lat', 'datetime', val_col]

y_pred = np.expand_dims(np.array(raw_x[val_col]), -1)
y_var = np.expand_dims(np.zeros(raw_x.shape[0]), -1)
ys_pred = np.expand_dims(np.array(raw_xs[val_col]), -1)
ys_var = np.expand_dims(np.zeros(raw_xs.shape[0]), -1)

train_df = np.concatenate([np.array(raw_x[column_names]), y_pred, y_var], axis=1)
test_df = np.concatenate([np.array(raw_xs[column_names]), ys_pred, ys_var], axis=1)

train_df = np.concatenate([train_df,test_df], axis=0)


if PLOT_GRID:
    ys = np.load('results/{file_prefix}_ys_grid.npy'.format(file_prefix=file_prefix)) #ys, ys_var
    ys_var = np.expand_dims(ys[:, 1], -1)
    ys = np.expand_dims(ys[:, 0], -1)
    observed = np.expand_dims(np.repeat([None], ys.shape[0]), -1)
    raw_grid_xs = pd.read_csv('data/processed_data_x_test_grid.csv') #id, time, x, y, covs
    #grid_test_df = np.concatenate([np.array(raw_grid_xs[['id', 'epoch', 'lon', 'lat', 'datetime']]), np.expand_dims(np.repeat([None], ys.shape[0]), -1), ys], axis=1)
    ys = np.expand_dims(np.array(raw_grid_xs[val_col]), -1)
    grid_test_df = np.concatenate([np.array(raw_grid_xs[['id', 'epoch', 'lon', 'lat', 'datetime']]), observed, ys, ys_var], axis=1)
else:
    grid_test_df = None

if False:
    data_save = np.concatenate([np.array(raw_grid_xs[['id', 'x', 'y', 'datetime']]), ys], axis=1)
    df = pd.DataFrame(data=data_save, columns=['id', 'x', 'y', 'datetime', 'no2', 'var'])
    df.to_csv('results/results_df.csv', index=False)

    data_df = raw_grid_xs


visualise = SpaceTimeVisualise(train_df, grid_test_df, test_start = np.min(raw_xs['epoch']))
visualise.show()

