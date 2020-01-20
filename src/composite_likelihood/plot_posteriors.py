import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import glob

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

list_of_files = (glob.glob('cluster/*_results'))
latest_file = max(list_of_files, key=os.path.getctime)
ROOT = latest_file+'/'
ROOT = ''

#=========================================Plot=========================================

num_datasets = len(X)


def plot_observations():
    a = 0
    x_0, y_0 = get_aggr_data(X[0], Y[0])
    c = 'black'
    plt.plot(x_0, y_0, c=c, label=labels[0])

    x_1, y_1 = get_aggr_data(X[1], Y[1])
    plt.scatter(
        x_1,
        y_1, 
        marker = 'x',
        c=[c],
        label=labels[1],
        zorder=10.0
    )

def plot_single_gp():
    ys = np.load(ROOT+'results/single_gp_ys.npy', allow_pickle=True)
    c = color_palette[0]
    xs = XS.flatten()
    ys_var = 2*np.sqrt(ys[:, 3])
    ys_mean = ys[:, 2]

    plt.fill_between(xs, ys_mean+ys_var, ys_mean-ys_var, facecolor=c, alpha=0.4)
    ax.plot(xs, ys_mean+ys_var,  c=c, alpha=0.5)
    ax.plot(xs, ys_mean-ys_var, c=c, alpha=0.5)

    plt.plot(xs, ys_mean, c=c,  linestyle='dashed', linewidth=3, label='True Likelihood')
    print(ys.shape)

def plot_gp_aggr():
    ys = np.load(ROOT+'results/gp_aggr_ys.npy', allow_pickle=True)
    c = color_palette[2]
    xs = XS.flatten()
    ys_var = 2*np.sqrt(ys[:, 1])
    ys_mean = ys[:, 0]

    plt.fill_between(xs, ys_mean+ys_var, ys_mean-ys_var, facecolor=c, alpha=0.4)
    ax.plot(xs, ys_mean+ys_var,  c=c, alpha=0.5)
    ax.plot(xs, ys_mean-ys_var, c=c, alpha=0.5)

    plt.plot(xs, ys_mean, c=c, linewidth=3, linestyle='dotted', marker='v', markevery=10, markersize=10, label='VBAgg')
    print(ys.shape)

def plot_dgp():
    ys = np.load(ROOT+'results/mr_dgp_ys.npy', allow_pickle=True)
    c = color_palette[1]
    xs = XS.flatten()
    ys_var = 2*np.sqrt(ys[:, 1])
    ys_mean = ys[:, 0]

    plt.fill_between(xs, ys_mean+ys_var, ys_mean-ys_var, facecolor=c, alpha=0.4)
    ax.plot(xs, ys_mean+ys_var,  c=c, alpha=0.5)
    ax.plot(xs, ys_mean-ys_var, c=c, alpha=0.5)

    plt.plot(xs, ys_mean, c=c, linewidth=3, linestyle='dotted', marker='v', markevery=10, markersize=10, label='DGP')

def plot_gp_aggr_corrected():
    ys = np.load(ROOT+'results/gp_aggr_corrected_ys.npy', allow_pickle=True)
    c = color_palette[3]
    xs = XS.flatten()
    ys_var = 2*np.sqrt(ys[:, 1])
    ys_mean = ys[:, 0]

    if False:
        plt.fill_between(xs, ys_mean+ys_var, ys_mean-ys_var, facecolor=c, alpha=0.3)
        ax.plot(xs, ys_mean+ys_var,  c=c, alpha=0.4)
        ax.plot(xs, ys_mean-ys_var, c=c, alpha=0.4)

    err = plt.errorbar(
        xs, 
        ys_mean,
        fmt="none", 
        yerr=ys_var,
        errorevery=15,
        capsize=5,
        linewidth=3.0,
        linestyle='-',
        c=c
    )

    plt.plot(xs, ys_mean, c=c, linewidth=3, label='MR-GP')
    print(ys.shape)


sns.set(color_codes=True)
fig, ax = plt.subplots(figsize=(12, 8))

plot_single_gp()
#plot_gp_aggr_corrected()
#plot_dgp()
plot_gp_aggr()

plot_observations()

plt.xlim(3, 10)
plt.ylim(-1.5, 6.1)

plt.legend(prop={'size': 15})
plt.tight_layout()

plt.savefig('vis/compare_corrected.pgf', bbox_inches = 'tight', pad_inches = 0)
plt.savefig('vis/compare_corrected.png', bbox_inches = 'tight', pad_inches = 0)
plt.show()
