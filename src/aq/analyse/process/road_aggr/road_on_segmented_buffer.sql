drop table if exists orca.os_highways_on_buffer;
-- drop table if exists orca.os_highways_on_buffer_preprocess;

-- Precompute buffer intersection with os_highways

-- select 
--    buffers.id,
--    buffers.part_id,
--    road_links.routehiera,
--    ST_Intersection(road_links.geom, buffers.buffer_geom) as road_geom,
--    buffers.buffer_geom,
--    buffers.site_geom
-- into 
--    orca.os_highways_on_buffer_preprocess
-- from
--    orca.location_buffers as buffers,
--    orca.os_highways_links as road_links
-- where
--    ST_Intersects(road_links.geom, buffers.buffer_geom);

-- extract features 

select 
   buffer.id,
   buffer.site_id,
   buffer.part_id,
   buffer.src,
   COALESCE(part.total_b_road_length,0) as total_b_road_length,
   COALESCE(part.total_length,0) as total_length,
   part.min_distance_to_road,
   buffer.buffer_geom
into 
   orca.os_highways_on_buffer
from
   (
      select
         buffer.id,
         buffer.part_id,
         COALESCE(sum(ST_Length(buffer.road_geom)) filter (where buffer.routehiera='B Road'), 0) as total_b_road_length,
         sum(ST_Length(buffer.road_geom)) as total_length,
         min(ST_Distance(buffer.site_geom, buffer.road_geom)) as min_distance_to_road
      from
         orca.os_highways_on_buffer_preprocess as buffer
      group by
         buffer.id, buffer.part_id
   ) as part
right join
   orca.location_buffers as buffer
on
   buffer.part_id = part.part_id and
   buffer.id = part.id;
