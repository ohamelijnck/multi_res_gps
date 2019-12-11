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

test_id = 0
r=2

X = np.load('data/data_x_{test_id}_2.npy'.format(test_id=test_id), allow_pickle=True)  #Aggr data , Points
Y = np.load('data/data_y_{test_id}_2.npy'.format(test_id=test_id), allow_pickle=True)
XS = np.load('data/data_xs_{test_id}.npy'.format(test_id=test_id), allow_pickle=True)
YS = np.load('data/data_ys_{test_id}.npy'.format(test_id=test_id), allow_pickle=True)


DATA_VIS = pd.read_csv('data/data_vis_raw_{test_id}.csv'.format(test_id=test_id))
print(DATA_VIS.columns)


DATA_VIS['date'] = pd.to_datetime(DATA_VIS['date'] )

print(X[0].shape)
print(X[1].shape)

labels = [
    'Observed Y',
    'Observed Y'
]

color_palette = sns.color_palette()

#=========================================Plot=========================================

num_datasets = len(X)

def _plot_observations():
    a = 0
    x_0, y_0 = get_aggr_data(X[0], Y[0])

    plt.plot(x_0, y_0, c=color_palette[0], label='')

    x_0, y_0 = get_aggr_data(XS, YS)
    plt.scatter(x_0, y_0, c=[color_palette[0]], label='')

    x_2, y_2 = get_aggr_data(X[1], Y[1])
    plt.scatter(
        x_2,
        y_2, 
        marker = 'x',
        c=[color_palette[1]],
        label=''
    )


def plot_observations():
    plt.plot(DATA_VIS['date'], DATA_VIS['pm10'])
    plt.plot(DATA_VIS['date'], DATA_VIS['pm25'])


def plot_mr_dgp():
    ys = np.load('results/cmgp_dgp_expert_1_y_vis_{test_id}_{r}.npy'.format(test_id=test_id,r=r), allow_pickle=True)
    c = color_palette[0]
    xs = DATA_VIS['date']
    

    ys_mean = ys[:, 0]
    ys_var = 2*np.sqrt(ys[:, 1])

    plt.fill_between(xs, ys_mean+ys_var, ys_mean-ys_var, facecolor=c, alpha=0.4)
    ax.plot(xs, ys_mean+ys_var,  c=c, alpha=0.5)
    ax.plot(xs, ys_mean-ys_var, c=c, alpha=0.5)

    plt.plot(xs, ys_mean, c=c,  linestyle='dashed', linewidth=3, label='mgp')
    print(ys.shape)

def plot_mr_gprn():
    ys = np.load('results/cmgp_gprn_aggr_y_vis_{test_id}_{r}.npy'.format(test_id=test_id, r=r), allow_pickle=True)
    c = color_palette[1]
    xs = DATA_VIS['date']

    ys_mean = ys[:, 0]
    ys_var = 2*np.sqrt(ys[:, 1])

    plt.fill_between(xs, ys_mean+ys_var, ys_mean-ys_var, facecolor=c, alpha=0.4)
    ax.plot(xs, ys_mean+ys_var,  c=c, alpha=0.5)
    ax.plot(xs, ys_mean-ys_var, c=c, alpha=0.5)

    plt.plot(xs, ys_mean, c=c,  linestyle='dashed', linewidth=3, label='gprn')
    print(ys.shape)

def plot_center_point():
    ys = np.load('results/center_point_gprn_y_vis_{test_id}_{r}.npy'.format(test_id=test_id, r=r), allow_pickle=True)
    c = color_palette[2]
    xs = DATA_VIS['date']

    ys_mean = ys[:, 0]
    ys_var = 2*np.sqrt(ys[:, 1])

    plt.fill_between(xs, ys_mean+ys_var, ys_mean-ys_var, facecolor=c, alpha=0.4)
    ax.plot(xs, ys_mean+ys_var,  c=c, alpha=0.5)
    ax.plot(xs, ys_mean-ys_var, c=c, alpha=0.5)

    plt.plot(xs, ys_mean, c=c,  linestyle='dashed', linewidth=3, label='center')
    print(ys.shape)

sns.set(color_codes=True)
fig, ax = plt.subplots(figsize=(12, 8))

plot_observations()
plot_mr_dgp()
plot_mr_gprn()
plot_center_point()

#plt.xlim(3, 10)
#plt.ylim(-1.5, 6.1)

plt.legend(prop={'size': 15})
plt.tight_layout()

#plt.savefig('vis/compare_corrected.pgf', bbox_inches = 'tight', pad_inches = 0)
#plt.savefig('vis/compare_corrected.png', bbox_inches = 'tight', pad_inches = 0)
plt.show()
