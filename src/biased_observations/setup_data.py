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
    y = 5*np.sin(x)**2 +np.random.randn(x.shape[0])*0.1
    #y = np.sin(x)**2 + np.random.randn(x.shape[0])*0.01
    return y

shift_x = 10
N = 50
#main data
x = np.linspace(shift_x+(-4), shift_x+3, N)


y = np.expand_dims(get_f(x), -1)
x = np.expand_dims(np.expand_dims(x, -1),-1)

xs= np.linspace(shift_x+(-20), shift_x+10, 200)
ys = np.expand_dims(get_f(xs), -1)
xs= np.expand_dims(np.expand_dims(xs, -1),-1)


#aggr 1
#x = np.linspace(-10, 10, N)
r = 5
scale = 0.5
x_1 = np.linspace(shift_x+(-20),shift_x+0, 2*N)
y_1 = scale*get_f(x_1)
x_1 = np.expand_dims(get_aggr_x(x_1, r),-1)
y_1 = np.expand_dims(get_aggr_y(y_1, r),-1)

#aggr 1
#x = np.linspace(-10, 10, N)
r = 5
scale = 0.3
x_2 = np.linspace(shift_x+(0),shift_x+10, 2*N)
y_2 = scale*get_f(x_2)
x_2 = np.expand_dims(get_aggr_x(x_2, r),-1)
y_2 = np.expand_dims(get_aggr_y(y_2, r),-1)


print(x_1.shape, x_2.shape, x.shape)
print(y_1.shape, y_2.shape, y.shape)
X = np.array([x_2, x_1, x])
Y = np.array([y_2, y_1, y])

print(X)

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



