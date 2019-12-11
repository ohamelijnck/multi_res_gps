import sys
import os

from util.sql_util import *
from buffers.create_buffers import *
from locations.create_site_locations import *
import landuse_aggr.ukmap_on_buffer as ukmap_on_buffer
import road_aggr.generate_road_on_buffer_sql as road_aggr_on_buffer
import street_canyon_aggr.generate_street_canyon_on_buffer as street_canyon_on_buffer

from db import *

#===============================================SETUP LOCATION AND BUFFER TABLES===============================================
CONFIG = {
    'DB_NAME': 'postgis_test',
    'SCHEMA': 'orca',
    'INPUT_TABLE': 'nips_locations',
    'TMP_INPUT_TABLE': 'tmp_locations',
    'OUTPUT_TABLE': 'processed_buffers',
    'BUFFER_TABLE': 'location_buffers',
    'SITE_LOCATIONS_TABLE': 'site_locations',
    'BUFFER_SIZES': [BUFFER_1C],
    'ROOT': os.getcwd(),

    'CLEAN_RUN': True,
    'DRY_RUN': False,
}


db = DB(name=CONFIG['DB_NAME'], connect=False)

create_site_locations(CONFIG, db)
create_buffers(CONFIG, db)

#===============================================GENERATE FEATURES===============================================
#objs = [ukmap_on_buffer.cov_buffer, road_aggr_on_buffer.cov_buffer]
objs = [road_aggr_on_buffer, street_canyon_on_buffer, ukmap_on_buffer]
objs = [o.get_cov_buffer(CONFIG, db) for o in objs]


#===============================================MERGE ALL FEATURES TOGETHER===============================================

PROCESS_DATA_TEMPLATE = """
drop table if exists orca.processed_buffers;

select
   location.src_id as id,
   location.src,
   location.datetime,
   location.epoch,
   location.lat,
   location.lon,
   location.val,
   ST_X(location.geom) as x,
   ST_Y(location.geom) as y,
   {columns},
   geom
into
    {schema}.{output_table}
from
    {tables},
    {schema}.{input_table} as location
where
    {where_clause};

copy (select * from orca.processed_buffers) to '{root}/data/processed_data/data_with_features.csv' CSV HEADER DELIMITER ',';
"""


all_where_clause = []
all_tables = []
all_columns = []

for obj in objs:
    schema = obj.psql_schema
    names = obj.get_buffer_size_names()
    column_names = obj.get_column_names()
    buffer_sizes = obj.buffer_sizes

    #get tables names relating to different buffer sizes
    on_buffer_table_names = []
    for name in names:
        on_buffer_table_names.append(obj.get_on_buffer_table_name(name))

    #sql join of the table names
    tables = arr_str_zip(arr_suffix(' as ', arr_prefix('{schema}.'.format(schema=schema), on_buffer_table_names)), on_buffer_table_names)
    all_tables += tables

    #sql where joining on site_id
    where_clause =  arr_suffix('.id ', arr_prefix('location.site_id = ', on_buffer_table_names))
    all_where_clause += where_clause

    #get all feature columns from each of the buffer tables
    for i in range(0, len(on_buffer_table_names)):
        buffer_size_name = names[i]
        on_buffer_table_name = on_buffer_table_names[i] 

        columns = arr_str_zip_with(arr_prefix(on_buffer_table_name+'.', column_names), ' as ', arr_suffix('_'+buffer_size_name, column_names))
        all_columns += columns


sql = PROCESS_DATA_TEMPLATE.format(
    root=CONFIG['ROOT'],
    schema=CONFIG['SCHEMA'],
    input_table=CONFIG['TMP_INPUT_TABLE'],
    output_table=CONFIG['OUTPUT_TABLE'],
    columns = arr_to_sql_select_list(',',all_columns),
    tables = arr_to_sql_select_list(',', all_tables),
    where_clause = arr_to_sql_select_list(' and',  all_where_clause)
)

print(sql)

db = DB(name=CONFIG['DB_NAME'], connect=False)
db.thread_safe_execute(sql)
