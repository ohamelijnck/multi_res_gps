drop table if exists orca.street_canyon_on_buffer_preprocess_1c;
drop table if exists orca.street_canyon_on_buffer_preprocess_100;

drop table if exists orca.street_canyons_on_buffer_100;
drop table if exists orca.street_canyons_on_buffer_500;
drop table if exists orca.street_canyons_on_buffer_1c;

-- drop table if exists orca.street_canyon_on_buffer_preprocess_500;
-- 
-- -- Precompute buffer intersection with os_highways
-- 
-- select 
--    buffers.id,
--    road_links.routehiera,
--    road_links.max_width,
--    road_links.min_width,
--    road_links.ave_width,
--    road_links.ratio_avg,
--    ST_Intersection(road_links.geom, buffers.buffer_geom) as road_geom,
--    buffers.buffer_geom,
--    buffers.site_geom
-- into 
--    orca.street_canyon_on_buffer_preprocess_500
-- from
--    orca.location_buffers_500 as buffers,
--    orca.street_canyons as road_links
-- where
--    ST_Intersects(road_links.geom, buffers.buffer_geom);


-- Precompute buffer intersection with os_highways

select 
   buffers.id,
   road_links.routehiera,
   road_links.max_width,
   road_links.min_width,
   road_links.ave_width,
   road_links.ratio_avg,
   ST_Intersection(road_links.road_geom, buffers.buffer_geom) as road_geom,
   buffers.buffer_geom,
   buffers.site_geom
into 
   orca.street_canyon_on_buffer_preprocess_100
from
   orca.location_buffers_100 as buffers,
   orca.street_canyon_on_buffer_preprocess_500 as road_links
where
   ST_Intersects(road_links.road_geom, buffers.buffer_geom);

select 
   buffers.id,
   road_links.routehiera,
   road_links.max_width,
   road_links.min_width,
   road_links.ave_width,
   road_links.ratio_avg,
   ST_Intersection(road_links.road_geom, buffers.buffer_geom) as road_geom,
   buffers.buffer_geom,
   buffers.site_geom
into 
   orca.street_canyon_on_buffer_preprocess_1c
from
   orca.location_buffers_1c as buffers,
   orca.street_canyon_on_buffer_preprocess_500 as road_links
where
   ST_Intersects(road_links.road_geom, buffers.buffer_geom);

select 
   buffer.id,
   buffer.src,
   COALESCE(covs.min_ratio_avg, 0) as min_ratio_avg,
   COALESCE(covs.avg_ratio_avg, 0) as avg_ratio_avg,
   COALESCE(covs.max_ratio_avg, 0) as max_ratio_avg,
   COALESCE(covs.min_min_width, 0) as min_min_width,
   COALESCE(covs.avg_min_width, 0) as avg_min_width,
   COALESCE(covs.max_min_width, 0) as max_min_width,

   buffer.buffer_geom
into 
   orca.street_canyons_on_buffer_500
from
   (
      select
         buffer.id,
         min(ratio_avg) as min_ratio_avg,
         avg(ratio_avg) as avg_ratio_avg,
         max(ratio_avg) as max_ratio_avg,

         min(min_width) as min_min_width,
         avg(min_width) as avg_min_width,
         max(min_width) as max_min_width

      from
         orca.street_canyon_on_buffer_preprocess_500 as buffer
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
   COALESCE(covs.min_ratio_avg, 0) as min_ratio_avg,
   COALESCE(covs.avg_ratio_avg, 0) as avg_ratio_avg,
   COALESCE(covs.max_ratio_avg, 0) as max_ratio_avg,
   COALESCE(covs.min_min_width, 0) as min_min_width,
   COALESCE(covs.avg_min_width, 0) as avg_min_width,
   COALESCE(covs.max_min_width, 0) as max_min_width,

   buffer.buffer_geom
into 
   orca.street_canyons_on_buffer_100
from
   (
      select
         buffer.id,
         min(ratio_avg) as min_ratio_avg,
         avg(ratio_avg) as avg_ratio_avg,
         max(ratio_avg) as max_ratio_avg,

         min(min_width) as min_min_width,
         avg(min_width) as avg_min_width,
         max(min_width) as max_min_width

      from
         orca.street_canyon_on_buffer_preprocess_100 as buffer
      group by
         buffer.id
   ) as covs
right join
   orca.location_buffers_100 as buffer
on
   buffer.id = covs.id;

select 
   buffer.id,
   buffer.src,
   COALESCE(covs.min_ratio_avg, 0) as min_ratio_avg,
   COALESCE(covs.avg_ratio_avg, 0) as avg_ratio_avg,
   COALESCE(covs.max_ratio_avg, 0) as max_ratio_avg,
   COALESCE(covs.min_min_width, 0) as min_min_width,
   COALESCE(covs.avg_min_width, 0) as avg_min_width,
   COALESCE(covs.max_min_width, 0) as max_min_width,

   buffer.buffer_geom
into 
   orca.street_canyons_on_buffer_1c
from
   (
      select
         buffer.id,
         min(ratio_avg) as min_ratio_avg,
         avg(ratio_avg) as avg_ratio_avg,
         max(ratio_avg) as max_ratio_avg,

         min(min_width) as min_min_width,
         avg(min_width) as avg_min_width,
         max(min_width) as max_min_width

      from
         orca.street_canyon_on_buffer_preprocess_1c as buffer
      group by
         buffer.id
   ) as covs
right join
   orca.location_buffers_1c as buffer
on
   buffer.id = covs.id;

