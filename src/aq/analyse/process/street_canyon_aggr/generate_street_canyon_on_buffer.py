import sys
sys.path.append('../')

from cov_on_buffer import *

def get_cov_buffer(CONFIG, db):
    cov_buffer = CovOnBuffer(
        config=CONFIG,
        db = db,
        psql_schema = CONFIG['SCHEMA'],
        buffer_root = CONFIG['BUFFER_TABLE']+'_',
        cov_table = 'street_canyons',
        cov_columns = ['max_width', 'routehiera', 'min_width', 'ave_width', 'ratio_avg'],
        overwrite_preprocess_buffers = CONFIG['CLEAN_RUN'],
        run_sql = not CONFIG['DRY_RUN'],
        preprocess_buffer_sizes = CONFIG['BUFFER_SIZES'],
        buffer_sizes = CONFIG['BUFFER_SIZES'],
        generated_columns = [
            ['min_ratio_avg', 'min(ratio_avg)'],
            ['avg_ratio_avg', "avg(ratio_avg)"],
            ['max_ratio_avg', "max(ratio_avg)"],
            ['min_min_width', "min(min_width)"],
            ['avg_min_width', "avg(min_width)"],
            ['max_min_width', "max(min_width)"],
        ]
    );
    return cov_buffer


