from datetime import datetime
from datetime import timedelta

import sys
sys.path.insert(0,'../')

import paper_config
import compare_time_series

def get_config():
    p_config = paper_config.get_config()
    c =  {
        'plot_var': True,
        'label_observed_aggr': None,
        'label_observed': 'Observed',
        'figure': {
            'x_lim': [-1, 20],
            'legend': {
                'loc':  'upper right'
            }
        },
        'plot_var': False
    }

    return compare_time_series.add_keys(c, p_config, overwrite=False)
