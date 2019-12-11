import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#=========================================Helper Functions=========================================
def get_aggr_data(x, y):
    res = []
    for i in y:
        res.append(np.repeat(i, x.shape[1]))
    res = np.array(res).flatten()
    x = x.flatten()
    return x, res

#=========================================Load Data=========================================

X = np.load('data/data_x.npy', allow_pickle=True) #Points, Aggr data
Y = np.load('data/data_y.npy', allow_pickle=True)
XS = np.load('data/data_xs.npy', allow_pickle=True)
YS = np.load('data/data_ys.npy', allow_pickle=True)

labels = [
    'Observed Y',
    'Observed Y'
]

color_palette = sns.color_palette()

#=========================================Plot=========================================

num_datasets = len(X)

sns.set(color_codes=True)
fig, ax = plt.subplots(figsize=(20, 8))


a = 0
x_0, y_0 = get_aggr_data(X[0], Y[0])
plt.plot(x_0, y_0, c=color_palette[0], label=labels[0])

x_1, y_1 = get_aggr_data(X[1], Y[1])
plt.scatter(
    x_1,
    y_1, 
    marker = 'x',
    c=[color_palette[1]],
    label=labels[1]
)

plt.legend()
plt.show()
