drop table if exists orca.bridge_roads_on_buffer_100;
drop table if exists orca.bridge_roads_on_buffer_500;
drop table if exists orca.bridge_roads_on_buffer_preprocess_100;
drop table if exists orca.bridge_roads_on_buffer_preprocess_500;
-- 
-- -- Precompute buffer intersection with os_highways
-- 
select 
   buffers.id,
   road_links.routehiera,
   ST_Intersection(road_links.geom, buffers.buffer_geom) as road_geom,
   buffers.buffer_geom,
   buffers.site_geom
into 
   orca.bridge_roads_on_buffer_preprocess_500
from
   orca.location_buffers_500 as buffers,
   orca.os_highways_bridge_roads as road_links
where
   ST_Intersects(road_links.geom, buffers.buffer_geom);

select 
   buffers.id,
   road_links.routehiera,
   ST_Intersection(road_links.road_geom, buffers.buffer_geom) as road_geom,
   buffers.buffer_geom,
   buffers.site_geom
into 
   orca.bridge_roads_on_buffer_preprocess_100
from
   orca.location_buffers_100 as buffers,
   orca.bridge_roads_on_buffer_preprocess_500 as road_links
where
   ST_Intersects(road_links.road_geom, buffers.buffer_geom) and
   road_links.id = buffers.id;



-- extract features 

select 
   buffer.id,
   buffer.src,
   COALESCE(covs.total_road_length, 0) as total_road_length,
   COALESCE(covs.total_a_road_primary_length, 0) as total_a_road_primary_length,
   COALESCE(covs.total_a_road_length, 0) as total_a_road_length,
   COALESCE(covs.total_b_road_length, 0) as total_b_road_length,
   COALESCE(covs.total_length, 0) as total_length,
   COALESCE(covs.min_distance_to_road, 0) as min_distance_to_road,
   buffer.buffer_geom
into 
   orca.bridge_roads_on_buffer_500
from
   (
      select
         buffer.id,
         sum(ST_Length(buffer.road_geom)) as total_road_length,
         sum(ST_Length(buffer.road_geom)) filter (where  buffer.routehiera='A Road Primary') as total_a_road_primary_length,
         sum(ST_Length(buffer.road_geom)) filter (where buffer.routehiera='A Road' ) as total_a_road_length,
         sum(ST_Length(buffer.road_geom)) filter (where buffer.routehiera='B Road' or buffer.routehiera='B Road Primary') as total_b_road_length,
         sum(ST_Length(buffer.road_geom)) as total_length,
         min(ST_Distance(buffer.site_geom, buffer.road_geom)) as min_distance_to_road
      from
         orca.bridge_roads_on_buffer_preprocess_500 as buffer
      group by
         buffer.id
   ) as covs
right join
   orca.location_buffers_500 as buffer
on
   buffer.id = covs.id;


select 
   buffer.id,
   buffer.src,
   COALESCE(covs.total_road_length, 0) as total_road_length,
   COALESCE(covs.total_a_road_primary_length, 0) as total_a_road_primary_length,
   COALESCE(covs.total_a_road_length, 0) as total_a_road_length,
   COALESCE(covs.total_b_road_length, 0) as total_b_road_length,
   COALESCE(covs.total_length, 0) as total_length,
   COALESCE(covs.min_distance_to_road, 0) as min_distance_to_road,
   buffer.buffer_geom
into 
   orca.bridge_roads_on_buffer_100
from
   (
      select
         buffer.id,
         sum(ST_Length(buffer.road_geom)) as total_road_length,
         sum(ST_Length(buffer.road_geom)) filter (where  buffer.routehiera='A Road Primary') as total_a_road_primary_length,
         sum(ST_Length(buffer.road_geom)) filter (where buffer.routehiera='A Road' or buffer.routehiera='A Road Primary') as total_a_road_length,
         sum(ST_Length(buffer.road_geom)) filter (where buffer.routehiera='B Road' or buffer.routehiera='B Road Primary') as total_b_road_length,
         sum(ST_Length(buffer.road_geom)) as total_length,
         min(ST_Distance(buffer.site_geom, buffer.road_geom)) as min_distance_to_road
      from
         orca.bridge_roads_on_buffer_preprocess_100 as buffer
      group by
         buffer.id
   ) as covs
right join
   orca.location_buffers_100 as buffer
on
   buffer.id = covs.id;

