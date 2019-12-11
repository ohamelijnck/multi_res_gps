import sys
sys.path.append('../')

from cov_on_buffer import *

def get_cov_buffer(CONFIG, db):
    cov_buffer = CovOnBuffer(
        config=CONFIG,
        db = db,
        psql_schema = CONFIG['SCHEMA'],
        buffer_root = CONFIG['BUFFER_TABLE']+'_',
        cov_table = 'ukmap_4326',
        cov_columns = ['*'],
        overwrite_preprocess_buffers = CONFIG['CLEAN_RUN'],
        run_sql = not CONFIG['DRY_RUN'],
        preprocess_buffer_sizes = CONFIG['BUFFER_SIZES'],
        buffer_sizes = CONFIG['BUFFER_SIZES'],
        intersected_geom='intersected_geom',
        generated_columns = [
            ['total_museum_area', "sum(ST_Area(buffer.intersected_geom)) filter (where buffer.landuse='Museum')"],
            ['total_hospital_area', "sum(ST_Area(buffer.intersected_geom)) filter (where buffer.landuse='Hospitals')"],
            ['total_grass_area', "sum(ST_Area(buffer.intersected_geom)) filter (where buffer.feature_ty='Vegetated')"],
            ['total_park_area', "sum(ST_Area(buffer.intersected_geom)) filter (where buffer.landuse='Park' or buffer.landuse='Recreational open space')"],
            ['total_water_area', "sum(ST_Area(buffer.intersected_geom)) filter (where buffer.feature_ty='Water')"],
            ['total_flat_area', "sum(ST_Area(buffer.intersected_geom)) filter (where buffer.feature_ty='Vegetated' or buffer.feature_ty='Water')"],
            ['max_building_height', "max(cast(buffer.calcaulate as float))"],
        ]
    );  
    return cov_buffer
