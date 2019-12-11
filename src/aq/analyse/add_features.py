import numpy as np
import pandas as pd
import os
from db import DB
#===============================================GET DATA===============================================

def fix_types(df, names, types):
    for i in range(len(names)):
        name = names[i]
        t = types[i]
        df[name] = df[name].astype(t)
    return df

raw_x = pd.read_csv('data/data_x.csv') #id, time, x, y, covs
raw_xs = pd.read_csv('data/data_xs.csv') #id, time, x, y, covs
raw_grid_xs = pd.read_csv('data/data_x_test_grid.csv') #id, time, x, y, covs
raw_fine_grid_xs = pd.read_csv('data/data_x_test_fine_grid.csv') #id, time, x, y, covs
sat_x = pd.read_csv('data/sat_data_x.csv') #id, time, x, y, covs

column_names = ['src', 'id', 'datetime', 'epoch', 'lat', 'lon', 'val']
column_types = [np.int, np.int, np.str, np.int, np.float64, np.float64, np.float64]

raw_x = fix_types(raw_x, column_names, column_types)
raw_xs = fix_types(raw_xs, column_names, column_types)
raw_grid_xs = fix_types(raw_grid_xs, column_names, column_types)
raw_fine_grid_xs = fix_types(raw_fine_grid_xs, column_names, column_types)
sat_x = fix_types(sat_x, column_names, column_types)

total_df = pd.concat([raw_x, raw_xs, raw_grid_xs, raw_fine_grid_xs, sat_x], axis=0)

#===============================================INSERT TO DB===============================================

db = DB(name='postgis_test', connect=True)

SCHEMA = 'orca'
LOCATION_TABLE_NAME = 'nips_locations'


schema = """
drop table if exists {schema}.{table};

create table {schema}.{table} (
    src integer,
    id integer,
    datetime timestamp,
    epoch integer,
    lat double precision,
    lon double precision,
    val double precision
);

""".format(schema=SCHEMA, table=LOCATION_TABLE_NAME)

db.execute(schema)
db.commit()

db.insert_df(total_df[column_names], SCHEMA, LOCATION_TABLE_NAME)

postprocess_sql = """
ALTER TABLE {schema}.{table} RENAME COLUMN id TO src_id;
ALTER TABLE {schema}.{table} ADD COLUMN id SERIAL PRIMARY KEY;
alter table {schema}.{table} add column geom geometry(Point, 4326);
update {schema}.{table} set geom=st_SetSrid(st_MakePoint(lon, lat), 4326);
create index {table}_gix on {schema}.{table} using GIST(geom);
""".format(schema=SCHEMA, table=LOCATION_TABLE_NAME)

db.execute(postprocess_sql)
db.commit()

#===============================================RUN FEATURE PROCESSING===============================================

#===============================================GET FEATURES BACK===============================================
