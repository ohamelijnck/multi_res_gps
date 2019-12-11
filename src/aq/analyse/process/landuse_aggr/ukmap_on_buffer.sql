drop table if exists orca.ukmap_on_buffer_500;
drop table if exists orca.ukmap_on_buffer_100;
drop table if exists orca.ukmap_on_buffer_1c;
 drop table if exists orca.ukmap_on_buffer_preprocess;
 drop table if exists orca.ukmap_on_buffer_preprocess_100;
 drop table if exists orca.ukmap_on_buffer_preprocess_1c;

-- Precompute buffer intersection with ukmap

-- select 
--    buffers.id,
--    buffers.src,
--    ukmap.*,
--    ST_Intersection(ST_Force2D(ukmap.geom), buffers.buffer_geom) as intersected_geom,
--    buffers.buffer_geom,
--    buffers.site_geom
-- into 
--    orca.ukmap_on_buffer_preprocess
-- from
--    orca.location_buffers_500 as buffers,
--    orca.ukmap_4326 as ukmap
-- where
--    ST_Intersects(ukmap.geom, buffers.buffer_geom);

-- create index ukmap_buffer_preprocess on orca.ukmap_on_buffer_preprocess  using GIST(buffer_geom);
-- 
-- -- extract features 
-- 
-- 
-- select 
--    buffers.id,
--    buffers.src,
--    ukmap.feature_ty,
--    ukmap.landuse,
--    ukmap.calcaulate,
--    ST_Intersection(ukmap.geom, buffers.buffer_geom) as intersected_geom,
--    buffers.buffer_geom,
--    buffers.site_geom
-- into 
--    orca.ukmap_on_buffer_preprocess_100
-- from
--    orca.location_buffers_100 as buffers,
--    orca.ukmap_on_buffer_preprocess as ukmap
-- where
--    ST_Intersects(ukmap.geom, buffers.buffer_geom) and
--    ukmap.id = buffers.id;
-- 
-- 
-- select 
--    buffers.id,
--    buffers.src,
--    ukmap.feature_ty,
--    ukmap.landuse,
--    ukmap.calcaulate,
--    ST_Intersection(ukmap.geom, buffers.buffer_geom) as intersected_geom,
--    buffers.buffer_geom,
--    buffers.site_geom
-- into 
--    orca.ukmap_on_buffer_preprocess_1c
-- from
--    orca.location_buffers_1c as buffers,
--    orca.ukmap_on_buffer_preprocess as ukmap
-- where
--    ST_Intersects(ukmap.geom, buffers.buffer_geom) and
--    ukmap.id = buffers.id;

select 
   buffer.id,
   buffer.src,
   COALESCE(part.total_museum_area, 0) as total_museum_area,
   COALESCE(part.total_hospital_area, 0) as total_hospital_area,
   COALESCE(part.total_park_area, 0) as total_park_area,
   COALESCE(part.total_grass_area, 0) as total_grass_area,
   COALESCE(part.total_water_area, 0) as total_water_area,
   COALESCE(part.total_flat_area, 0) as total_flat_area,
   COALESCE(part.max_building_height, 0) as max_building_height,
   buffer.buffer_geom
into 
   orca.ukmap_on_buffer_500
from
   (
      select
         buffer.id,
         COALESCE(sum(ST_Area(buffer.intersected_geom)) filter (where buffer.landuse='Museum'), 0) as total_museum_area,
         COALESCE(sum(ST_Area(buffer.intersected_geom)) filter (where buffer.landuse='Hospitals'), 0) as total_hospital_area,
         COALESCE(sum(ST_Area(buffer.intersected_geom)) filter (where buffer.feature_ty='Vegetated'), 0) as total_grass_area,
         COALESCE(sum(ST_Area(buffer.intersected_geom)) filter (where buffer.landuse='Park' or buffer.landuse='Recreational open space'), 0) as total_park_area,
         COALESCE(sum(ST_Area(buffer.intersected_geom)) filter (where buffer.feature_ty='Water'), 0) as total_water_area,
         COALESCE(sum(ST_Area(buffer.intersected_geom)) filter (where buffer.feature_ty='Vegetated' or buffer.feature_ty='Water'), 0) as total_flat_area,
         max(cast(buffer.calcaulate as float)) as max_building_height
      from
         orca.ukmap_on_buffer_preprocess as buffer
      group by
         buffer.id
   ) as part
right join
   orca.location_buffers_500 as buffer
on
   buffer.id = part.id;

select 
   buffer.id,
   buffer.src,
   COALESCE(part.total_museum_area, 0) as total_museum_area,
   COALESCE(part.total_hospital_area, 0) as total_hospital_area,
   COALESCE(part.total_grass_area, 0) as total_grass_area,
   COALESCE(part.total_park_area, 0) as total_park_area,
   COALESCE(part.total_water_area, 0) as total_water_area,
   COALESCE(part.total_flat_area, 0) as total_flat_area,
   COALESCE(part.max_building_height, 0) as max_building_height,
   buffer.buffer_geom
into 
   orca.ukmap_on_buffer_100
from
   (
      select
         buffer.id,
         COALESCE(sum(ST_Area(buffer.intersected_geom)) filter (where buffer.landuse='Museum'), 0) as total_museum_area,
         COALESCE(sum(ST_Area(buffer.intersected_geom)) filter (where buffer.landuse='Hospitals'), 0) as total_hospital_area,
         COALESCE(sum(ST_Area(buffer.intersected_geom)) filter (where buffer.feature_ty='Vegetated'), 0) as total_grass_area,
         COALESCE(sum(ST_Area(buffer.intersected_geom)) filter (where buffer.landuse='Park' or buffer.landuse='Recreational open space'), 0) as total_park_area,
         COALESCE(sum(ST_Area(buffer.intersected_geom)) filter (where buffer.feature_ty='Water'), 0) as total_water_area,
         COALESCE(sum(ST_Area(buffer.intersected_geom)) filter (where buffer.feature_ty='Vegetated' or buffer.feature_ty='Water'), 0) as total_flat_area,
         max(cast(buffer.calcaulate as float)) as max_building_height
      from
         orca.ukmap_on_buffer_preprocess_100 as buffer
      group by
         buffer.id
   ) as part
right join
   orca.location_buffers_100 as buffer
on
   buffer.id = part.id;

select 
   buffer.id,
   buffer.src,
   COALESCE(part.total_museum_area, 0) as total_museum_area,
   COALESCE(part.total_hospital_area, 0) as total_hospital_area,
   COALESCE(part.total_grass_area, 0) as total_grass_area,
   COALESCE(part.total_park_area, 0) as total_park_area,
   COALESCE(part.total_water_area, 0) as total_water_area,
   COALESCE(part.total_flat_area, 0) as total_flat_area,
   COALESCE(part.max_building_height, 0) as max_building_height,
   buffer.buffer_geom
into 
   orca.ukmap_on_buffer_1c
from
   (
      select
         buffer.id,
         COALESCE(sum(ST_Area(buffer.intersected_geom)) filter (where buffer.landuse='Museum'), 0) as total_museum_area,
         COALESCE(sum(ST_Area(buffer.intersected_geom)) filter (where buffer.landuse='Hospitals'), 0) as total_hospital_area,
         COALESCE(sum(ST_Area(buffer.intersected_geom)) filter (where buffer.feature_ty='Vegetated'), 0) as total_grass_area,
         COALESCE(sum(ST_Area(buffer.intersected_geom)) filter (where buffer.landuse='Park' or buffer.landuse='Recreational open space'), 0) as total_park_area,
         COALESCE(sum(ST_Area(buffer.intersected_geom)) filter (where buffer.feature_ty='Water'), 0) as total_water_area,
         COALESCE(sum(ST_Area(buffer.intersected_geom)) filter (where buffer.calcaulate='0'), 0) as total_flat_area,
         max(cast(buffer.calcaulate as float)) as max_building_height
      from
         orca.ukmap_on_buffer_preprocess_1c as buffer
      group by
         buffer.id
   ) as part
right join
   orca.location_buffers_1c as buffer
on
   buffer.id = part.id;

