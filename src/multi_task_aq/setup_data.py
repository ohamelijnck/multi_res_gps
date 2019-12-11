import numpy as np
import pandas as pd

import experiment_config 


CONFIG = experiment_config.get_config()

#hourly_site_2 = pd.read_csv('../data/laqn_data_hourly_site_2.csv') #'lat', 'lon', 'temperature', 'windspeed', 'windbearing', 'humidity', 'pressure', 'site_id', 'sitecode', 'date', 'sitetype', 'latitude', 'longitude', 'no2', 'o3', 'pm10_raw', 'pm10', 'pm25', 'geom'

START_DATE='2018-06-15'
END_DATE='2018-06-28'


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

def to_epoch(col):
    return (col.astype(np.int64) // 10**9).astype(np.float64)

def setup_df(df):
    df['date'] =  pd.to_datetime(df['date'])
    df['epoch'] = to_epoch(df['date'])

    print(np.min(df['date']), np.max(df['date']))
    df = df[df['date'].between(pd.Timestamp(START_DATE), pd.Timestamp(END_DATE))]
    return df

def get_X(df):
    #return np.array(df[['scaled_epoch',  'scaled_windspeed']])
    return np.array(df[['scaled_epoch']])

def get_Y(df, col='no2', scaled=True):
    scaled=CONFIG['scale_transform']
    if scaled:
        return np.array( df[['scaled_'+col]])
    return np.array( df[[col]])
    

def get_aggr(df, col, _type, _size, return_raw=False):
    #1 hour is the base resolution
    #_type = (DAY | HOUR)
    total_x = []
    total_y = []
    if _type is 'DAY':
        years = np.unique(df['date'].dt.year)
        for y in years:
            months = np.unique(df[df['date'].dt.year==y]['date'].dt.month)
            for m in months:
                days = np.unique(df[df['date'].dt.month==m]['date'].dt.day)

                for d in days:
                    _df = df[(df['date'].dt.year==y) & (df['date'].dt.month==m) & (df['date'].dt.day==d)]

                    if _df.shape[0] != 24:
                        #only want whole days
                        print('SKIPPING: ', y, m , d)
                        continue

                    if return_raw:
                        total_x.append(_df)
                        total_y.append(np.mean(_df, axis=0))
                    else:
                        total_x.append(get_X(_df))
                        total_y.append(np.mean(get_Y(_df, col), axis=0))


        if return_raw:
            total_x = pd.concat(total_x)
            total_y = pd.concat(total_y)
        else:
            total_x = np.array(total_x)
            total_y = np.array(total_y)

        return total_x, total_y
    else:
        stacked_x = []
        aggr_y = []
        cur_date = np.min(df['date'])
        max_date = np.max(df['date'])
        print('min', cur_date, max_date)
        while cur_date < max_date:
            print(cur_date)
            next_date = cur_date +   pd.Timedelta(hours=_size)
            _df = df[(df['date'] >= cur_date) &  (df['date'] < next_date)]

            if np.shape(_df)[0] != _size: 
                cur_date = next_date
                continue

            if return_raw:
                _df[col] = _df[col].mean(axis=0)
                _df_x = _df
                y_aggr = _df
            else:
                _df_x = get_X(_df)
                _df_y = get_Y(_df, col)
                y_aggr = np.mean(_df_y, axis=0)

            aggr_y.append(y_aggr)
            stacked_x.append(_df_x)

            cur_date = next_date


        if return_raw:
            stacked_x = pd.concat(stacked_x)
            aggr_y = pd.concat(aggr_y)
        else:
            stacked_x = np.array(stacked_x)
            aggr_y = np.array(aggr_y)

        return stacked_x, aggr_y



def get_hourly(df, col, scaled=True):
    total_x = np.expand_dims(get_X(df), 1)
    total_y = get_Y(df, col, scaled=scaled)
    return total_x, total_y

def get_site_1(TEST, root=''):

    hourly_site_1 = pd.read_csv(root+'data/laqn_data_hourly_site_1.csv') #'lat', 'lon', 'temperature', 'windspeed', 'windbearing', 'humidity', 'pressure', 'site_id', 'sitecode', 'date', 'sitetype', 'latitude', 'longitude', 'no2', 'o3', 'pm10_raw', 'pm10', 'pm25', 'geom'
    TEST_ID = TEST['id']
    EXP_START_TEST_DATE = TEST['start_test_date']
    EXP_END_TEST_DATE = TEST['end_test_date']

    hourly_site_1_all = setup_df(hourly_site_1)
    hourly_site_1_train = hourly_site_1_all[~ hourly_site_1_all['date'].between(pd.Timestamp(EXP_START_TEST_DATE), pd.Timestamp(EXP_END_TEST_DATE))]
    hourly_site_1_test = hourly_site_1_all[hourly_site_1_all['date'].between(pd.Timestamp(EXP_START_TEST_DATE), pd.Timestamp(EXP_END_TEST_DATE))]

    return hourly_site_1_all, hourly_site_1_train, hourly_site_1_test

def scale_col(df_all, df_train, df_test, col, sphere=True, log=False, wrt_train = False):
    f = lambda x: x
    if log: 
        f = lambda x: np.log(x)

    target = f(df_all[col])
    if wrt_train:
        target = f(df_train[col])

    df_all['scaled_{col}'.format(col=col)] = normalise_wrt(f(df_all[col]), target, sphere_flag=sphere)
    df_train['scaled_{col}'.format(col=col)] = normalise_wrt(f(df_train[col]), target, sphere_flag=sphere)
    df_test['scaled_{col}'.format(col=col)] = normalise_wrt(f(df_test[col]), target, sphere_flag=sphere)
    return df_all, df_train, df_test


def main():
    for TEST in CONFIG['tests']:
        TEST_ID = TEST['id']

        hourly_site_1_all, hourly_site_1_train, hourly_site_1_test = get_site_1(TEST)

        #hourly_site_2 = setup_df(hourly_site_2)





        hourly_site_1_all, hourly_site_1_train, hourly_site_1_test = scale_col(hourly_site_1_all, hourly_site_1_train, hourly_site_1_test, 'epoch')
        hourly_site_1_all, hourly_site_1_train, hourly_site_1_test = scale_col(hourly_site_1_all, hourly_site_1_train, hourly_site_1_test, 'humidity')
        hourly_site_1_all, hourly_site_1_train, hourly_site_1_test = scale_col(hourly_site_1_all, hourly_site_1_train, hourly_site_1_test, 'windspeed')
        hourly_site_1_all, hourly_site_1_train, hourly_site_1_test = scale_col(hourly_site_1_all, hourly_site_1_train, hourly_site_1_test, 'pressure')
        hourly_site_1_all, hourly_site_1_train, hourly_site_1_test = scale_col(hourly_site_1_all, hourly_site_1_train, hourly_site_1_test, 'windbearing')
        hourly_site_1_all, hourly_site_1_train, hourly_site_1_test = scale_col(hourly_site_1_all, hourly_site_1_train, hourly_site_1_test, 'pm10', sphere=True, log=CONFIG['log_transform'], wrt_train=True)
        hourly_site_1_all, hourly_site_1_train, hourly_site_1_test = scale_col(hourly_site_1_all, hourly_site_1_train, hourly_site_1_test, 'no2', sphere=True, log=CONFIG['log_transform'])
        hourly_site_1_all, hourly_site_1_train, hourly_site_1_test = scale_col(hourly_site_1_all, hourly_site_1_train, hourly_site_1_test, 'pm25', sphere=True, log=CONFIG['log_transform'])


        for r in TEST['resolutions']:
            #X_0, Y_0 = get_aggr(hourly_site_1_all, 'pm10', 'DAY', 1)
            X_0, Y_0 = get_aggr(hourly_site_1_all, CONFIG['aggregated_pollutant'], 'hour', r)
            #X_0, Y_0 = get_hourly(hourly_site_1_all, 'pm10')
            X, Y = get_hourly(hourly_site_1_train, CONFIG['target_pollutant'])

            print(r)
            print(X_0)
            print(X)

            X_VIS_R, Y_VIS_R = get_aggr(hourly_site_1_all, CONFIG['aggregated_pollutant'],'hour', r, return_raw=True)

            Y_VIS_R.to_csv('data/data_y_vis_{test_id}_{r}.csv'.format(test_id=TEST_ID,  r=r), header=True, index=False)

            np.save('data/data_x_{test_id}_{r}'.format(test_id=TEST_ID, r=r), [X_0, X])
            np.save('data/data_y_{test_id}_{r}'.format(test_id=TEST_ID, r=r), [Y_0, Y])

        XS, YS = get_hourly(hourly_site_1_test, CONFIG['target_pollutant'])

        X_VIS, Y_VIS = get_hourly(hourly_site_1_all, CONFIG['target_pollutant'])
        VIS_RAW = hourly_site_1_all


        print('X: ', X.shape)
        print('X_0, ', X_0.shape)


        VIS_RAW.to_csv('data/data_vis_raw_{test_id}.csv'.format(test_id=TEST_ID), header=True, index=False)
        
        np.save('data/data_x_vis_{test_id}'.format(test_id=TEST_ID), X_VIS)

        np.save('data/data_xs_{test_id}'.format(test_id=TEST_ID), XS)
        np.save('data/data_ys_{test_id}'.format(test_id=TEST_ID), YS)


if __name__ == '__main__':
    main()

