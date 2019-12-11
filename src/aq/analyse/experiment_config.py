from datetime import datetime
from datetime import timedelta

import sys
sys.path.insert(0,'../../')
sys.path.insert(0,'../../../')

import paper_config
import compare_time_series

def get_config():
    p_config = paper_config.get_config()
    EXP_START_TRAIN_DATE = '2019-02-19'
    EXP_END_TRAIN_DATE = '2019-02-20'

    EXP_START_TEST_DATE = '2019-02-20'
    EXP_END_TEST_DATE = '2019-02-21'
    c =  {
        'data_root': 'data/data_with_features',
        'plot_vis_only': True,
        'plot_vis_x_axis': 'datetime',
        'plot_vis_x_datetime': True,
        'vis_iter': 10,
        'tests': [
            {
                'id': 0,
                'start_test_date': '{start} 11:00:00'.format(start=EXP_START_TRAIN_DATE),
                'end_test_date': '{start} 11:00:00'.format(start=EXP_START_TRAIN_DATE),
            },
            {
                'id': 1,
                'start_test_date': '{d} 20:00:00'.format(d=EXP_START_TRAIN_DATE),
                'end_test_date': '{d} 20:00:00'.format(d=EXP_START_TRAIN_DATE),
            },
            {
                'id': 2,
                'start_test_date': '{d} 21:00:00'.format(d=EXP_START_TEST_DATE),
                'end_test_date': '{d} 21:00:00'.format(d=EXP_START_TEST_DATE),
            },

        ], 
        'target_pollutant': 'pm25',
        'aggregated_pollutant': 'pm10',
        'point_observations': ['pm25'],
        'aggregated_observations': ['pm10'],
        'start_train_date': EXP_START_TRAIN_DATE,
        'end_train_date': EXP_END_TRAIN_DATE,
        'start_test_date': EXP_START_TEST_DATE,
        'end_test_date': EXP_END_TEST_DATE
    }

    return compare_time_series.add_experiment_config_defaults(compare_time_series.add_keys(c, p_config, overwrite=False))

