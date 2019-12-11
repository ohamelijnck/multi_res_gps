import sys
sys.path.append('../')

from cov_on_buffer import *

def get_cov_buffer(CONFIG, db):
    cov_buffer = CovOnBuffer(
        config=CONFIG,
        db = db,
        psql_schema = CONFIG['SCHEMA'],
        buffer_root = CONFIG['BUFFER_TABLE']+'_',
        cov_table = 'os_highways_links',
        cov_columns = ['routehiera'],
        overwrite_preprocess_buffers = CONFIG['CLEAN_RUN'],
        run_sql = not CONFIG['DRY_RUN'],
        preprocess_buffer_sizes = CONFIG['BUFFER_SIZES'],
        buffer_sizes = CONFIG['BUFFER_SIZES'],
        generated_columns = [
            ['total_road_length', 'sum(ST_Length(buffer.geom))'],
            ['total_a_road_primary_length', "sum(ST_Length(buffer.geom)) filter (where  buffer.routehiera='A Road Primary')"],
            ['total_a_road_length', "sum(ST_Length(buffer.geom)) filter (where buffer.routehiera='A Road' )"],
            ['total_b_road_length', "sum(ST_Length(buffer.geom)) filter (where buffer.routehiera='B Road' or buffer.routehiera='B Road Primary')"],
            ['total_length', "sum(ST_Length(buffer.geom))"],
            ['min_distance_to_road', "min(ST_Distance(buffer.site_geom, buffer.geom))"], #if there are no roads then we want to get the max value
        ] 
    )
    return cov_buffer

