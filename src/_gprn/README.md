Code for MR-GPRN
=================

This is the code base for the paper 'Multi-resolution Multi Task Gaussian Processes' [1].

# A simple MR-GP model

## Data

Data is organised through `gprn.Dataset`. For each dataset/observation process as datasource is added to an Dataset object:

```python
    data = gprn.Dataset()
    num_data_sources = X.shape[0]

    for i in range(num_data_sources):
        # N = number of data observations
        # M = number of aggregation points
        # P = number of tasks
        # D = input dimensions
        x = np.array(X[i]) # N x M x D
        y = np.array(Y[i]) # N x P

        M = x.shape[1]

        #batch size
        b = 300
        b = b if b < x.shape[0] else x.shape[0]


        data.add_source_dict({
            'M': M,
            'x': x,
            'y': y,
            #'z': x,
            'batch_size': b
        })

    data.add_inducing_points(z_r);
```

### Adding inducing points

This can either be done by:

```python
    data.add_source_dict({
        'M': M,
        'x': x,
        'y': y,
        'z': z, #inducing points
        'batch_size': b
    })
```

or by calling

```python
    data.add_inducing_points(z);
```

as above.


### Missing Data
Missing data is support. In `y` this is denoted by `np.NaN`.

### Multi-task data

Multi-task data is support by simply stacking y into a matrix of size `N x P`. Not all observation processes will observe all tasks and so `active_tasks` is a list where for each dataset there is list of the active task indexes.

```python
    data.add_source_dict({
        'M': M,
        'x': x # N x M x D,
        'y': y # N x P,
        'batch_size': b,
        'active_tasks': [[1], [0]]
    })
```
## Context

Model settings are organised through `gprn.Context`. All settings from the number of iteration epochs, the amount of jitter, the kernels, initial hyperparameters are.

```python
    num_datasets = X.shape[0]
    num_outputs = Y[0].shape[1]

    context = gprn.context.ContextFactory().create()
    #Train the model flag
    context.train_flag=True 
    #Use previously trained model flag
    context.restore_flag= False 
    context.save_image = False


    #Use monte carlo for predictions
    context.monte_carlo = False

    context.debug = False
    context.num_outputs = num_outputs
    context.num_latent = 1
    context.num_components = 1

    context.use_diag_covar = False
    context.use_diag_covar_flag = False

    
    context.train_inducing_points_flag = True

    context.whiten=True
    #Optimise hyper and variational parametes in EM style
    context.split_optimise=True
    context.jitter = 1e-4
    context.shuffle_seed = 0
    context.num_epochs = 2000
    context.seed = 0
    context.restore_location = 'restore/{name}_{test}_{r}.ckpt'.format(name=CONFIG['file_prefix'], test=test, r=r)

```
## MR-GP
Kernels only need to be specified for both `f`.

 ```python
    context.kernels = [
    {
        'f': [gprn.kernels.SE(num_dimensions=X[0].shape[-1], length_scale=inv(0.1)) for i in range(context.num_latent)],
    }, #r=0
    ]
```
## MR-GPRN
Kernels need to be specified for both `f` and `W`.

 ```python
    context.kernels = [
    {

        'f': [ gprn.kernels.SE(num_dimensions=D, length_scale=inv(0.1)) for i in range(context.num_latent) ],
        'w': [[gprn.kernels.SE(num_dimensions=D, length_scale=inv(3.0)) for j in range(context.num_latent)] for i in range(context.num_outputs)]
    }, #r=0
    ]
```
## MR-Kernels

When using MR-GPRN, MR-GP to allow for vectorised operations one needs to use MR-SE. MR-SE does not support ARD directly, but this can be achieved by simply getting the product of multilple MR-SE kernels over the input dimensions.

```python
def get_prod(D=1, init_vars = []):
    k_arr = []
    include_arr = []
    for i in range(D):
        include_arr.append([i])
        k_arr.append(gprn.kernels.MR_SE(num_dimensions=1, length_scale=init_vars[0]))

    return gprn.kernels.Product(k_arr, include_arr=include_arr)

context.kernels = [{
    'f': [get_prod(D=X[0].shape[-1], init_vars=[inv(0.1)]) for i in range(context.num_latent)],
    'w': [[get_prod(D=X[0].shape[-1], init_vars=[inv(0.1)]) for j in range(context.num_latent)] for i in range(context.num_outputs)]
}]
```


