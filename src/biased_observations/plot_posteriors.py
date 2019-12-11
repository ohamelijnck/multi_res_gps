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

color_palette = sns.color_palette('colorblind')

var_flag=True


#=========================================Plot=========================================

num_datasets = len(X)


def plot_observations():
    a = 0
    x_0, y_0 = get_aggr_data(X[0], Y[0])
    plt.plot(x_0, y_0, c='black', label='', linewidth=3)

    x_1, y_1 = get_aggr_data(X[1], Y[1])
    plt.plot(
        x_1,
        y_1, 
        c='black',
        marker='s',
        markersize=5,
        markevery=3,
        label='',
        linewidth=3,
    )

    x_2, y_2 = get_aggr_data(X[2], Y[2])
    plt.scatter(
        x_2,
        y_2, 
        marker = 'x',
        c='black',
        label='',
        zorder=10,
        s=100
    )

    plt.scatter(
        XS,
        YS, 
        marker = 'x',
        c='grey',
        label='',
        s=100
    )


def plot_mr_dgp():
    ys = np.load('results/mr_dgp_ys.npy', allow_pickle=True)
    i = ys.shape[1]-2
    #for i in range(int(ys.shape[1]/2)):
    print(int(ys.shape[1]/2))
    i = 0 #last layer
    c = color_palette[0]
    xs = XS.flatten()

    ys_mean = ys[:, i*2]
    ys_var = 2*np.sqrt(ys[:, (i*2)+1])

    if var_flag:
        plt.fill_between(xs, ys_mean+ys_var, ys_mean-ys_var, facecolor=c, alpha=0.4)

        ax.plot(xs, ys_mean+ys_var,  c=c, alpha=0.5)
        ax.plot(xs, ys_mean-ys_var, c=c, alpha=0.5)

    plt.plot(xs, ys_mean, c=c,  linestyle='dashed', linewidth=5, label='MR-DGP'.format(l=i))
    print(ys.shape)

def plot_vbagg():
    ys = np.load('results/gp_aggr_ys.npy', allow_pickle=True)
    c_i = 1
    i = 0
    print(int(ys.shape[1]/2))
    c = color_palette[c_i+i]
    xs = XS.flatten()

    ys_mean = ys[:, i*2]
    print(ys[:, (i*2)+1])
    ys_var = 2*np.sqrt(ys[:, (i*2)+1])

    if var_flag:
        plt.fill_between(xs, ys_mean+ys_var, ys_mean-ys_var, facecolor=c, alpha=0.4)
        ax.plot(xs, ys_mean+ys_var,  c=c, alpha=0.5)
        ax.plot(xs, ys_mean-ys_var, c=c, alpha=0.5)

    plt.plot(xs, ys_mean, c=c,  linestyle='dashed', marker='^', markersize=15, markevery=4, linewidth=5, label='VBagg'.format(l=i))
    print(ys.shape)

def plot_mr_gprn():
    ys = np.load('results/gprn_aggr_ys.npy', allow_pickle=True)
    c_i = 2
    i = 0
    print(int(ys.shape[1]/2))
    c = color_palette[c_i+i]
    xs = XS.flatten()

    ys_mean = ys[:, i*2]
    print(ys[:, (i*2)+1])
    ys_var = 2*np.sqrt(ys[:, (i*2)+1])

    if var_flag:
        plt.fill_between(xs, ys_mean+ys_var, ys_mean-ys_var, facecolor=c, alpha=0.4)
        ax.plot(xs, ys_mean+ys_var,  c=c, alpha=0.5)
        ax.plot(xs, ys_mean-ys_var, c=c, alpha=0.5)

    #plt.plot(xs, ys_mean, c=c,  linestyle='dotted', linewidth=3, label='MR-GPRN'.format(l=i))
    plt.plot(xs, ys_mean, c=c,  linestyle='solid', linewidth=5, label='MR-GPRN')
    print(ys.shape)

def plot_mr_dgp_cascade():
    ys = np.load('results/dgp_cascade_ys.npy', allow_pickle=True)
    c_i = 1
    #for i in range(int(ys.shape[1]/2)):
    i = 0 #last layer
    print(int(ys.shape[1]/2))
    c = color_palette[c_i+i]
    xs = XS.flatten()

    ys_mean = ys[:, i*2]
    print(ys[:, (i*2)+1])
    ys_var = 2*np.sqrt(ys[:, (i*2)+1])

    if var_flag:
        plt.fill_between(xs, ys_mean+ys_var, ys_mean-ys_var, facecolor=c, alpha=0.4)
        ax.plot(xs, ys_mean+ys_var,  c=c, alpha=0.5)
        ax.plot(xs, ys_mean-ys_var, c=c, alpha=0.5)

    plt.plot(xs, ys_mean, c=c,  linestyle='dotted', linewidth=5, label='DGP-Cascade'.format(l=i))
    print(ys.shape)


sns.set(color_codes=True)
fig, ax = plt.subplots(figsize=(12, 8))

plot_mr_dgp()
plot_vbagg()
plot_mr_gprn()
plot_mr_dgp_cascade()
plot_observations()

plt.xlim(1, 17)
#plt.ylim(-1.0, 6.4)
#plt.ylim(-0.5, 6.0)
plt.ylim(-0.5, 5.5)

plt.legend(prop={'size': 20}, loc='upper right', fancybox=True, framealpha=0.8)
plt.tight_layout()

plt.savefig('vis/compare_baseline.pgf', bbox_inches = 'tight', pad_inches = 0)
plt.savefig('vis/compare_baseline.png', bbox_inches = 'tight', pad_inches = 0)
plt.show()
