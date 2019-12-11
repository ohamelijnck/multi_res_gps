import numpy as np
import os

np.random.seed(0)

def get_aggr_x(x, r):
    N = x.shape[0]

    step = r
    current_index = 0
    aggr_x = []
    for i in range(int(N/r)):
        next_index = current_index + step
        aggr_x.append(x[current_index:next_index])
        current_index = next_index
    return np.array(aggr_x)

def get_aggr_y(y, r):
    mu = []
    N = y.shape[0]
    step = r
    current_index = 0
    
    for i in range(int(N/r)):
        next_index = current_index + step
        mu.append((1/step)*np.sum(y[current_index:next_index]))
        current_index = next_index

    mu = np.array(mu)
    return mu

def get_f(x):
    sig = 0.5
    y = 5*np.sin(x)**2 +np.random.randn(x.shape[0])*sig
    #y = np.sin(x)**2 + np.random.randn(x.shape[0])*0.01
    return y

N = 100
x = np.linspace(-2, 15, N)
#x = np.linspace(0,5, N)
#x = np.linspace(10,15, N)
y = get_f(x)

xs= np.expand_dims(np.linspace(-2,15, 500), -1)
ys= np.expand_dims(y, -1)

r = 2
x_r = get_aggr_x(x, r)
y_r = get_aggr_y(y, r)

scale = 1.0
x_r = np.expand_dims(x_r, -1)
y_r = scale*np.expand_dims(y_r, -1)

#x = np.linspace(-2, 10, N)
#y = np.expand_dims(get_f(x), -1)

x = x
y = np.expand_dims(y, -1)


X = np.array([x_r, np.expand_dims(np.expand_dims(x, -1), -1)])
Y = np.array([y_r, y])
xs = np.expand_dims(xs, -1)
ys = ys

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

#create data dir
ensure_dir('data')
ensure_dir('results')
ensure_dir('vis')
ensure_dir('models/restore')

np.save('data/data_x', X)
np.save('data/data_y', Y)
np.save('data/data_xs', xs)
np.save('data/data_ys', ys)


