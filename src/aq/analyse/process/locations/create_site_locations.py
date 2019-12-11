import sys
sys.path.append('../')

from cov_on_buffer import *

CREATE_SITE_LOCATIONS_TEMPLATE = """
drop table if exists {schema}.{site_locations_table};
select
    array_agg(id) as array_input_id,
    row_number() over (order by 1) as site_id,
    array_agg(src) as array_input_src,
    geom
into
    {schema}.{site_locations_table}
from
    {schema}.{input_table}
group by
    geom;
create index {site_locations_table}_gix on {schema}.{site_locations_table} using GIST(geom);


drop table if exists {schema}.{tmp_site_locations};
select
    input.*,
    tmp_locations.site_id
into
    {schema}.{tmp_site_locations}
from
    {schema}.{input_table} as input,
    (
        select
           unnest(unique_sites.array_input_id) as input_id,
           unique_sites.site_id,
           unnest(unique_sites.array_input_src) as src,
           unique_sites.geom
        from
            {schema}.{site_locations_table} as unique_sites
    ) as tmp_locations
where
    input.id = tmp_locations.input_id and 
    input.src = tmp_locations.src;

"""



def create_site_locations(CONFIG, db):
    global CREATE_SITE_LOCATIONS_TEMPLATE

    sql = CREATE_SITE_LOCATIONS_TEMPLATE.format(
      schema = CONFIG['SCHEMA'],  
      input_table = CONFIG['INPUT_TABLE'],  
      site_locations_table = CONFIG['SITE_LOCATIONS_TABLE'],
      tmp_site_locations = CONFIG['TMP_INPUT_TABLE']
    )
    print(sql)

    if not CONFIG['DRY_RUN']:
        db.thread_safe_execute(sql)




