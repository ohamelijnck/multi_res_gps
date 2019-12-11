import numpy as np
import pandas as pd
import os
from db import DB

import experiment_config

def fix_types(df, names, types):
    for i in range(len(names)):
        name = names[i]
        t = types[i]
        df[name] = df[name].astype(t)
    return df


EXPERIMENT_CONFIG = experiment_config.get_config()
vis_id = EXPERIMENT_CONFIG['vis_iter']

#===============================================LOAD DATA WITHOUT FEATURES===============================================
raw_x = pd.read_csv('data/data_x.csv') #id, time, x, y, covs
raw_xs = pd.read_csv('data/data_xs.csv') #id, time, x, y, covs
raw_grid_xs = pd.read_csv('data/data_x_test_grid.csv') #id, time, x, y, covs
raw_fine_grid_xs = pd.read_csv('data/data_x_test_fine_grid.csv') #id, time, x, y, covs
sat_x = pd.read_csv('data/sat_data_x.csv') #id, time, x, y, covs

column_names = ['src', 'id', 'datetime', 'epoch', 'lat', 'lon', 'val']
column_names_no_val = column_names[:-1]
column_types = [np.int, np.int, np.str, np.int, np.float64, np.float64, np.float64]

raw_x = fix_types(raw_x, column_names, column_types)
raw_grid_xs = fix_types(raw_grid_xs, column_names, column_types)
raw_fine_grid_xs = fix_types(raw_fine_grid_xs, column_names, column_types)
sat_x = fix_types(sat_x, column_names, column_types)
raw_xs = fix_types(raw_xs, column_names, column_types)

#===============================================LOAD DATA WITH FEATURES - NEEDS TO BE MATCHED TO RAW DATA ABOVE===============================================

all_data = pd.read_csv('data/processed_data/data_with_features.csv') #id', 'src', 'lat', 'lon', 'datetime', *
print(all_data.columns)

def get_features(df):
    #x = ['epoch', 'lat', 'lon', 'min_distance_to_road_100', 'avg_ratio_avg_100', 'total_flat_area_1c']
    #x = ['epoch',  'lat', 'lon', 'total_road_length_1c', 'max_building_height_100']
    x = ['epoch',  'total_road_length_1c', 'max_building_height_100']
    x = ['epoch', 'lat', 'lon', 'total_flat_area_1c', 'total_a_road_length_1c', 'avg_ratio_avg_1c']
    #x = ['epoch', 'lat', 'lon']
    #x = ['epoch', 'total_flat_area_1c', 'total_a_road_length_1c', 'avg_ratio_avg_1c']
    #x = ['epoch',  'avg_ratio_avg_100']
    #x = ['epoch', 'lat', 'lon']
    return np.array(df[x])

def get_targets(df):
    y = ['val']
    return np.array(df[y])

def denormalise_wrt(x, y, sphere_flag=False):
    sphere = 1.0
    if sphere_flag:
        sphere = np.nanstd(y, axis=0)

    return (x*sphere)+np.nanmean(y,axis=0)

def normalise(x, sphere_flag=False):
    sphere = 1.0
    if sphere_flag:
        sphere = np.nanstd(x, axis=0)

    return (x - np.nanmean(x, axis=0))/sphere

def normalise_wrt(x, y, sphere_flag=False):
    sphere = 1.0
    if sphere_flag:
        sphere = np.nanstd(y, axis=0)
    return (x - np.nanmean(y, axis=0))/sphere

def collect_sources(target_df, srcs, apply_fn=lambda a: a, col='src', use_numpy=False):
    df = None
    for s in srcs:
        _df = target_df[target_df[col] == s]

        _df = apply_fn(_df)

        if use_numpy:
            df = _df if df is None else np.concatenate([df, _df])
        else:
            df = _df if df is None else pd.concat([df, _df])
    return df

raw_laqn_train_x = collect_sources(all_data, np.unique(raw_x['src']), apply_fn=get_features, use_numpy=True)
raw_laqn_train_y = collect_sources(all_data, np.unique(raw_x['src']), apply_fn=get_targets, use_numpy=True)

DISCRETISE_SIZE = 10
NORM_TARGET=True
norm_lambda  =  lambda a: normalise_wrt(a, raw_laqn_train_x, sphere_flag=False)

if NORM_TARGET:
    #norm_y_wrt_lambda  =  lambda a, b: normalise_wrt(a, b, sphere_flag=False)
    norm_y_wrt_lambda  =  lambda a, b: a
    norm_wrt_lambda  =  lambda a, b: normalise_wrt(a, b, sphere_flag=True)
else:
    norm_y_wrt_lambda  =  lambda a, b: a
    norm_wrt_lambda  =  lambda a, b: a

rs = lambda a : np.reshape(a, [int(a.shape[0]/DISCRETISE_SIZE**2), DISCRETISE_SIZE**2, a.shape[1]])

get_single = lambda a : np.expand_dims(a[0, :], -1)

laqn_train_df = collect_sources(all_data, np.unique(raw_x['src']))
laqn_test_df = collect_sources(all_data, np.unique(raw_xs['src']))
grid_points_df = collect_sources(all_data, np.unique(raw_grid_xs['src']))
fine_grid_points_df = collect_sources(all_data, np.unique(raw_fine_grid_xs['src']))

vis_df = pd.concat([laqn_train_df[laqn_train_df['id']==vis_id], laqn_test_df[laqn_test_df['id']==vis_id]], axis=0)

date_slice = None
for t in EXPERIMENT_CONFIG['tests']:
    s = (pd.to_datetime(grid_points_df['datetime']).between(pd.Timestamp(t['start_test_date']), pd.Timestamp(t['end_test_date'])))         
    if date_slice is None:
        date_slice = s
    else:
        date_slice = date_slice | s

small_grid_points_df = grid_points_df[date_slice]

fine_date_slice = None
for t in EXPERIMENT_CONFIG['tests']:
    s = (pd.to_datetime(fine_grid_points_df['datetime']).between(pd.Timestamp(t['start_test_date']), pd.Timestamp(t['end_test_date'])))         
    if fine_date_slice is None:
        fine_date_slice = s
    else:
        fine_date_slice = fine_date_slice | s

small_fine_grid_points_df = fine_grid_points_df[fine_date_slice]

sat_train_df = collect_sources(all_data, np.unique(sat_x['src']))

processed_discretise_sat_points_x = collect_sources(all_data, np.unique(sat_x['src']), lambda a: rs(norm_wrt_lambda(get_features(a), raw_laqn_train_x)), use_numpy=True)
processed_discretise_sat_points_y = collect_sources(all_data, np.unique(sat_x['src']), lambda a: norm_y_wrt_lambda(get_single(get_targets(a)), raw_laqn_train_y), use_numpy=True)

processed_laqn_train_x = norm_wrt_lambda(get_features(laqn_train_df), raw_laqn_train_x)
processed_laqn_train_y = norm_y_wrt_lambda(get_targets(laqn_train_df), raw_laqn_train_y)

processed_laqn_test_x = norm_wrt_lambda(get_features(laqn_test_df), raw_laqn_train_x)
processed_laqn_test_y = norm_y_wrt_lambda(get_targets(laqn_test_df), raw_laqn_train_y)

processed_vis_x = norm_wrt_lambda(get_features(vis_df), raw_laqn_train_x)
processed_vis_y = norm_y_wrt_lambda(get_targets(vis_df), raw_laqn_train_y)

processed_grid_points = norm_wrt_lambda(get_features(grid_points_df), raw_laqn_train_x)
processed_small_grid_points = norm_wrt_lambda(get_features(small_grid_points_df), raw_laqn_train_x)

processed_fine_grid_points = norm_wrt_lambda(get_features(fine_grid_points_df), raw_laqn_train_x)
processed_small_fine_grid_points = norm_wrt_lambda(get_features(small_fine_grid_points_df), raw_laqn_train_x)

#===============================================SAVE DATA===============================================

print('processed_laqn_train_x: ',processed_laqn_train_x.shape)
print('processed_laqn_train_y: ', processed_laqn_train_y.shape)

processed_X = [processed_discretise_sat_points_x, np.expand_dims(processed_laqn_train_x, 1)]
processed_Y = [processed_discretise_sat_points_y, processed_laqn_train_y]

print('processed_X[0]: ', processed_X[0].shape)
print('processed_X[1]: ', processed_X[1].shape)
print('processed_Y[0]: ', processed_Y[0].shape)
print('processed_Y[1]: ', processed_Y[1].shape)
print('processed_vis_x: ', processed_vis_x.shape)
print('processed_vis_y: ', processed_vis_y.shape)

processed_XS = processed_laqn_test_x
processed_YS = processed_laqn_test_y

np.save('data/data_with_features/processed_x_train', processed_X)
np.save('data/data_with_features/processed_y_train', processed_Y)

np.save('data/data_with_features/processed_x_test', processed_XS)
np.save('data/data_with_features/processed_y_test', processed_YS)

np.save('data/data_with_features/processed_vis_x_{id}'.format(id=vis_id), processed_vis_x)
np.save('data/data_with_features/processed_vis_y_{id}'.format(id=vis_id), processed_vis_y)

np.save('data/data_with_features/processed_x_test_grid', processed_grid_points)
np.save('data/data_with_features/processed_x_test_small_grid', processed_small_grid_points)

np.save('data/data_with_features/processed_x_test_fine_grid', processed_fine_grid_points)
np.save('data/data_with_features/processed_x_test_small_fine_grid', processed_small_fine_grid_points)

vis_df.to_csv('data/data_with_features/data_vis_raw_{id}.csv'.format(id=vis_id), index=False)

laqn_train_df.to_csv('data/data_with_features/processed_data_x.csv', index=False)
sat_train_df.to_csv('data/data_with_features/processed_data_sat_x.csv', index=False)
laqn_test_df.to_csv('data/data_with_features/processed_data_xs.csv', index=False)
grid_points_df.to_csv('data/data_with_features/processed_data_x_test_grid.csv', index=False)
small_grid_points_df.to_csv('data/data_with_features/processed_data_x_test_small_grid.csv', index=False)

fine_grid_points_df.to_csv('data/data_with_features/processed_data_x_test_fine_grid.csv', index=False)
small_fine_grid_points_df.to_csv('data/data_with_features/processed_data_x_test_small_fine_grid.csv', index=False)












