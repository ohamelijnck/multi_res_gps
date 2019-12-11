import matplotlib
matplotlib.use("Qt5agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

data_x = pd.read_csv('data/data_x.csv')
sat_data_x = pd.read_csv('data/sat_data_x.csv')
data_xs = pd.read_csv('data/data_xs.csv')

prediction_y = np.load('results/cmgp_aggr_y.npy')
prediction_ys = np.load('results/cmgp_aggr_ys.npy')

data_x['y_pred'] = prediction_y[:, 0]
data_x['y_var'] = prediction_y[:, 1]

data_xs['y_pred'] = prediction_ys[:, 0]
data_xs['y_var'] = prediction_ys[:, 1]

_d = data_x[data_x['site_id'] == 10]
_ds = data_xs[data_xs['site_id'] == 10]

plt.plot(_d['epoch'], _d['y_pred'])
plt.plot(_ds['epoch'], _ds['y_pred'])

plt.scatter(_d['epoch'], _d['no2'])
plt.scatter(_ds['epoch'], _ds['no2'])

plt.scatter(sat_data_x['epoch'], sat_data_x['no2'])

plt.show()

