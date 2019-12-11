from datetime import datetime
from datetime import timedelta

import sys
sys.path.insert(0,'../')
sys.path.insert(0,'../../')

import paper_config
import compare_time_series

def get_config():
    #resolutions = [1, 2, 5, 10, 24]
    p_config = paper_config.get_config()
    #resolutions = [1, 5, 10, 24]
    #resolutions = [5, 10]
    resolutions = [2, 5, 10, 24]
    #resolutions = [24]
    #resolutions = [5]
    #start_test_date = datetime.strptime('2018-06-18', "%Y-%m-%d")
    start_test_date = datetime.strptime('2018-06-18', "%Y-%m-%d")
    end_test_date = datetime.strptime('2018-06-28', "%Y-%m-%d")

    iterators = []

    _id = 0
    cur_date = start_test_date
    while cur_date < end_test_date:
        next_date = cur_date + timedelta(days=2)

        iterators.append({
            'id': _id,
            'start_test_date': cur_date.strftime("%Y-%m-%d"),
            'end_test_date': next_date.strftime("%Y-%m-%d"),
            'resolutions': resolutions,
            'iterator': 'resolutions', 
        })

        cur_date = next_date
        _id = _id + 1

    
    c =   {
        'log_transform': False,
        'scale_transform': False,
        'vis_test': 0,
        'vis_iter': 2,
        'plot_vis_only': True,
        'plot_vis_x_axis': 'date',
        'plot_vis_x_datetime': True,
        'num_x_ticks': 5,
        'tests': iterators, 
        'resolutions': resolutions,
        'target_pollutant': 'pm10',
        'aggregated_pollutant': 'pm25',
        'point_observations': ['pm10'],
        'aggregated_observations': ['pm25'],
        'start_test_date': start_test_date,
        'end_test_date': end_test_date
    }

    return compare_time_series.add_keys(c, p_config, overwrite=False)
    

