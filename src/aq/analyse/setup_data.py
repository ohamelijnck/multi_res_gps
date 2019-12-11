import matplotlib
matplotlib.use("Qt5agg")
import matplotlib.pyplot as plt

import datetime

import numpy as np
import pandas as pd
import os
import copy

import experiment_config

"""

"""

EXPERIMENT_CONFIG = experiment_config.get_config()

#===============================================EXPERIMENT SETTINGS===============================================
EXP_START_TRAIN_DATE = EXPERIMENT_CONFIG['start_train_date']
EXP_END_TRAIN_DATE = EXPERIMENT_CONFIG['end_train_date']

EXP_START_TEST_DATE = EXPERIMENT_CONFIG['start_test_date']
EXP_END_TEST_DATE = EXPERIMENT_CONFIG['end_test_date']

GRID_START =  EXP_START_TRAIN_DATE
GRID_END =  EXP_END_TEST_DATE

NORMALISE = False

DISCRETISE_SIZE = 10

DATA_SRC_COL = 'src'

class Counter(object):
    def __init__(self):
        self.num_sources = 0

    def add_source(self):
        val = self.num_sources
        self.num_sources += 1
        return copy.copy(val)

sources = Counter()


#===============================================HELPER FUNCTIONS===============================================

def get_grid_in_region(x1,x2,y1,y2, n):
    A = np.linspace(x1, x2, n)
    B = np.linspace(y1, y2, n)
    g = [[a, b] for b in B for a in A]
    return np.array(g)

def to_epoch(col):
    return col.astype(np.int64) // 10**9

def int_to_padded_str(col, zfill=1):
    return col.apply(lambda x: str(int(x)).zfill(zfill))

def discretise_sat(df):
    decimal=4
    total_x = None
    total_y= None
    for i in range(len(df)):
        r = df.iloc[i]
        lat= np.round(r['lat'],decimal)
        lon= np.round(r['lon'],decimal)
        g = get_grid_in_region(lat -0.05,lat+0.05, lon-0.05,lon+0.05, DISCRETISE_SIZE)
        t = r['epoch']
        
        row = np.array([[[t, x[0], x[1]] for x in g]])
        y = np.array([[r['no2']]])
        
        total_x = row if total_x is None else np.concatenate([total_x, row],axis=0)
        total_y = y if total_y is None else np.concatenate([total_y, y],axis=0)
    return total_x, total_y


rs = lambda x: x.reshape([x.shape[0]*x.shape[1], x.shape[2]])

def point_in_region(p_lat, p_lon, centers, w=0.05):
    for c in centers:
        c_lat = c[0]
        c_lon = c[1]

        if ((c_lat-w) <= p_lat) and  (p_lat <= (c_lat+w)):
            if ((c_lon-w) <= p_lon) and  (p_lon <= (c_lon+w)):
                return True
    return False

def get_unique_pairs(df, col_1, col_2, decimal=5):
    tmp_df = df
    tmp_df[col_1] = tmp_df[col_1].round(decimals=decimal)
    tmp_df[col_2] = tmp_df[col_2].round(decimals=decimal)
    a = [[lat, np.unique(tmp_df[tmp_df[col_1] == lat][col_2])] for lat in np.unique(tmp_df[col_1])]
    a = [[x[0], y] for x in a for y in x[1]]
    a = np.array(a)
    return a

def get_datetime_from_epoch(col):
    #TODO: check timezones
    return col.apply(lambda epoch: datetime.datetime.fromtimestamp(epoch).strftime('%Y-%m-%d %H:%M:%S'))

#===============================================LOAD LAQN DATA===============================================

laqn_df = pd.read_csv('data/raw_data/laqn_data.csv')

LAQN_DATE_COL = 'date'
LAQN_VAL_COL = 'no2'
LAQN_LAT_COL = 'latitude'
LAQN_LON_COL = 'longitude'
LAQN_ID_COL = 'site_id'


sitecode_with_largest_no2 = laqn_df.iloc[laqn_df[LAQN_VAL_COL].idxmax()]['sitecode']
#df_max = df[df['sitecode']==sitecode_with_largest_no2]

laqn_df[LAQN_DATE_COL] =  pd.to_datetime(laqn_df[LAQN_DATE_COL])
laqn_df['epoch'] = to_epoch(laqn_df[LAQN_DATE_COL])

#===============================================LOAD SAT DATA===============================================

SAT_DATE_COL = 'datetime'
SAT_VAL_COL = 'no2'
SAT_LAT_COL = 'lat'
SAT_LON_COL = 'lon'
SAT_ID_COL = 'id'

sat_df = pd.read_csv('data/raw_data/sat_data.csv')

tmp_col = sat_df['date'].astype(np.int).astype(str)+int_to_padded_str(sat_df['hour'], 2)
sat_df[SAT_DATE_COL] = pd.to_datetime(tmp_col, format='%Y%m%d%H')
sat_df = sat_df.sort_values(by=SAT_DATE_COL, ascending=True)
sat_df['epoch'] = to_epoch(sat_df[SAT_DATE_COL])
sat_df[SAT_VAL_COL] = sat_df['no2']*1000000000 #convert to the correct units

sat_centers = get_unique_pairs(sat_df, SAT_LAT_COL, SAT_LON_COL)

#===============================================PROCESSES LAQN DATA===============================================

laqn_df = laqn_df[laqn_df[LAQN_DATE_COL].between(pd.Timestamp(EXP_START_TRAIN_DATE), pd.Timestamp(EXP_END_TEST_DATE))]


def fn(row):
    lat = row[LAQN_LAT_COL]
    lon = row[LAQN_LON_COL]
    return point_in_region(lat, lon, sat_centers)

laqn_df = laqn_df[laqn_df.apply(fn, axis=1)]

laqn_train_df = laqn_df[laqn_df[LAQN_DATE_COL].between(pd.Timestamp(EXP_START_TRAIN_DATE), pd.Timestamp(EXP_END_TRAIN_DATE))]
laqn_test_df = laqn_df[laqn_df[LAQN_DATE_COL].between(pd.Timestamp(EXP_START_TEST_DATE), pd.Timestamp(EXP_END_TEST_DATE))]


def laqn_get_data_df(df):
    x = df[[LAQN_ID_COL, LAQN_DATE_COL, 'epoch',LAQN_LAT_COL, LAQN_LON_COL, LAQN_VAL_COL]]
    y = df[[LAQN_VAL_COL]]
    return x, y

def laqn_get_data_matrix(df):
    return np.expand_dims(df[['epoch',LAQN_LAT_COL, LAQN_LON_COL]], axis=1)


laqn_train_x_df, laqn_train_y_df = laqn_get_data_df(laqn_train_df)
laqn_test_x_df, laqn_test_y_df = laqn_get_data_df(laqn_test_df)

laqn_train_x_df[DATA_SRC_COL] = sources.add_source()
laqn_test_x_df[DATA_SRC_COL] = sources.add_source()

#===============================================PROCESS SAT DATA===============================================

#satellite data is between the start train date to end test date because we are evaluating how the model can predict the LAQN data not the satelitte data
sat_train_df = sat_df[sat_df[SAT_DATE_COL].between(pd.Timestamp(EXP_START_TRAIN_DATE), pd.Timestamp(EXP_END_TEST_DATE))]
sat_train_x, sat_train_y = discretise_sat(sat_train_df)

#satellite data is an area source, need to discritise the area
discretise_points = None
for i in range(sat_train_x.shape[0]):
    row = sat_train_x[i, :]
    row_y = np.tile(sat_train_y[i, :], [row.shape[0], 1])
    _src = sources.add_source()

    #add srcs
    row = np.concatenate([row, np.tile(_src, [row.shape[0], 1])], axis=1)
    #add ids
    row = np.concatenate([row, np.expand_dims(range(row.shape[0]), -1), row_y], axis=1)

    discretise_points = row if discretise_points is None else np.concatenate([discretise_points, row], axis=0)

discretise_points_df = pd.DataFrame(discretise_points, columns=['epoch', LAQN_LAT_COL, LAQN_LON_COL, DATA_SRC_COL, LAQN_ID_COL, LAQN_VAL_COL])
discretise_points_df[LAQN_DATE_COL] = get_datetime_from_epoch(discretise_points_df['epoch'])

#===============================================PREDICTION GRID===============================================

GRID_SIZE = 100
TOTAL_SIZE = GRID_SIZE**2

grid_min_lat, grid_max_lat = np.min(sat_df[SAT_LAT_COL]-0.05), np.max(sat_df[SAT_LAT_COL]+0.05)
grid_min_lon, grid_max_lon = np.min(sat_df[SAT_LON_COL])-0.05, np.max(sat_df[SAT_LON_COL]+0.05)

lats = np.linspace(grid_min_lat, grid_max_lat, GRID_SIZE)
lons = np.linspace(grid_min_lon, grid_max_lon, GRID_SIZE)
epochs = np.unique(sat_df[sat_df[SAT_DATE_COL].between(pd.Timestamp(GRID_START), pd.Timestamp(GRID_END))]['epoch'])


#get a matrix with the 'cross product' of all the lat,lon,time points
_grid_points = np.array([[lat, lon]  for lat in lats for lon in lons])
grid_points = np.concatenate([np.expand_dims(range(TOTAL_SIZE), -1).astype(np.int),_grid_points], axis=1)
epochs_tiled = np.expand_dims(np.tile(np.expand_dims(epochs, -1), [1, TOTAL_SIZE]).flatten(), -1)
grid_points = np.concatenate([epochs_tiled, np.tile(grid_points, [epochs.shape[0], 1])], axis=1)
grid_points_df = pd.DataFrame(grid_points, columns=['epoch',LAQN_ID_COL, LAQN_LAT_COL, LAQN_LON_COL])


grid_points_df[LAQN_DATE_COL]= get_datetime_from_epoch(grid_points_df['epoch'])
grid_points_df[LAQN_VAL_COL]= None #prediction area - we don't know the value of AQ here
grid_points_df[DATA_SRC_COL]= sources.add_source()


grid_points = np.array(grid_points_df[['epoch',LAQN_LAT_COL, LAQN_LON_COL]])

#===============================================PREDICTION FINER GRID===============================================

GRID_SIZE = 100
TOTAL_SIZE = GRID_SIZE**2

fine_grid_min_lat, fine_grid_max_lat = 51.47, 51.55
fine_grid_min_lon, fine_grid_max_lon = -0.14, -0.085

lats = np.linspace(fine_grid_min_lat, fine_grid_max_lat, GRID_SIZE)
lons = np.linspace(fine_grid_min_lon, fine_grid_max_lon, GRID_SIZE)
epochs = np.unique(sat_df[sat_df[SAT_DATE_COL].between(pd.Timestamp(GRID_START), pd.Timestamp(GRID_END))]['epoch'])


#get a matrix with the 'cross product' of all the lat,lon,time points
_grid_points = np.array([[lat, lon]  for lat in lats for lon in lons])
fine_grid_points = np.concatenate([np.expand_dims(range(TOTAL_SIZE), -1).astype(np.int),_grid_points], axis=1)
epochs_tiled = np.expand_dims(np.tile(np.expand_dims(epochs, -1), [1, TOTAL_SIZE]).flatten(), -1)
fine_grid_points = np.concatenate([epochs_tiled, np.tile(fine_grid_points, [epochs.shape[0], 1])], axis=1)
fine_grid_points_df = pd.DataFrame(fine_grid_points, columns=['epoch',LAQN_ID_COL, LAQN_LAT_COL, LAQN_LON_COL])


fine_grid_points_df[LAQN_DATE_COL]= get_datetime_from_epoch(fine_grid_points_df['epoch'])
fine_grid_points_df[LAQN_VAL_COL]= None #prediction area - we don't know the value of AQ here
fine_grid_points_df[DATA_SRC_COL]= sources.add_source()


fine_grid_points = np.array(fine_grid_points_df[['epoch',LAQN_LAT_COL, LAQN_LON_COL]])

#===============================================SAVE DATA===============================================

#make sure all dataframes have the same columns, in the same order, with the same names
column_names = ['src', 'id', 'datetime', 'epoch', 'lat', 'lon', 'val']
column_types = [np.str, np.int, np.str, np.int, np.float64, np.float64, np.float64]


laqn_train_x_df = laqn_train_x_df[[DATA_SRC_COL, LAQN_ID_COL, LAQN_DATE_COL, 'epoch', LAQN_LAT_COL, LAQN_LON_COL, LAQN_VAL_COL]]
laqn_test_x_df = laqn_test_x_df[[DATA_SRC_COL, LAQN_ID_COL, LAQN_DATE_COL, 'epoch', LAQN_LAT_COL, LAQN_LON_COL, LAQN_VAL_COL]]
discretise_points_df = discretise_points_df[[DATA_SRC_COL, LAQN_ID_COL, LAQN_DATE_COL, 'epoch', LAQN_LAT_COL, LAQN_LON_COL, LAQN_VAL_COL]]
grid_points_df = grid_points_df[[DATA_SRC_COL, LAQN_ID_COL, LAQN_DATE_COL, 'epoch', LAQN_LAT_COL, LAQN_LON_COL, LAQN_VAL_COL]]

fine_grid_points_df = fine_grid_points_df[[DATA_SRC_COL, LAQN_ID_COL, LAQN_DATE_COL, 'epoch', LAQN_LAT_COL, LAQN_LON_COL, LAQN_VAL_COL]]


laqn_train_x_df.to_csv('data/data_x.csv', index=False, header=column_names)
laqn_test_x_df.to_csv('data/data_xs.csv', index=False, header=column_names)

discretise_points_df.to_csv('data/sat_data_x.csv', index=False, header=column_names)
grid_points_df.to_csv('data/data_x_test_grid.csv', index=False, header=column_names)
fine_grid_points_df.to_csv('data/data_x_test_fine_grid.csv', index=False, header=column_names)

