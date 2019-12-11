import sys
sys.path.append('../')

from cov_on_buffer import *

CREATE_BUFFERS_TEMPLATE = """
drop table if exists {schema}.{buffer_table}_{buffer_size_name};

select
   sites.site_id as id,
   sites.geom as site_geom,
   ST_Buffer(sites.geom, {buffer_size}) buffer_geom
into
    {schema}.{buffer_table}_{buffer_size_name}
from
   {schema}.{input_table} as sites;

CREATE INDEX {buffer_table}_{buffer_size_name}_gix ON {schema}.{buffer_table}_{buffer_size_name} USING GIST(buffer_geom);
"""

BUFFER_SIZES_DICT = {
    BUFFER_500: ['500', 0.005], #~500m
    BUFFER_1C:  ['1c', 0.001], #~1000m
    BUFFER_100: ['100', 0.0001] #~100m
}

def create_buffers(CONFIG, db):
    global CREATE_BUFFERS_TEMPLATE
    global BUFFER_SIZES_DICT

    for b in CONFIG['BUFFER_SIZES']:
        b_name, b_size = BUFFER_SIZES_DICT[b]

        sql = CREATE_BUFFERS_TEMPLATE.format(
          schema = CONFIG['SCHEMA'],  
          input_table = CONFIG['SITE_LOCATIONS_TABLE'],  
          buffer_table = CONFIG['BUFFER_TABLE'],  
          buffer_size = b_size,
          buffer_size_name = b_name
        )
        print(sql)

        if not CONFIG['DRY_RUN']:
            db.thread_safe_execute(sql)



