drop table if exists orca.ukmap_on_buffer;
-- drop table if exists orca.ukmap_on_buffer_preprocess;

-- Precompute buffer intersection with ukmap
-- select 
--    buffers.id,
--    buffers.site_id,
--    buffers.part_id,
--    buffers.src,
--    ukmap.feature_ty,
--    ST_Intersection(ST_Force2D(ukmap.geom), buffers.buffer_geom) as intersected_geom,
--    buffers.buffer_geom,
--    buffers.site_geom
-- into 
--    orca.ukmap_on_buffer_preprocess
-- from
--    orca.location_buffers as buffers,
--    orca.ukmap_4326 as ukmap
-- where
--    ST_Intersects(ukmap.geom, buffers.buffer_geom);

-- extract features 

select 
   buffer.id,
   buffer.site_id,
   buffer.part_id,
   buffer.src,
   COALESCE(part.total_grass_area,0) as total_grass_area,
   buffer.buffer_geom
into 
   orca.ukmap_on_buffer
from
   (
      select
         buffer.id,
         buffer.part_id,
         COALESCE(sum(ST_Area(buffer.intersected_geom)) filter (where buffer.feature_ty='Vegetated'), 0) as total_grass_area
      from
         orca.ukmap_on_buffer_preprocess as buffer
      group by
         buffer.id, buffer.part_id
   ) as part
right join
   orca.location_buffers as buffer
on
   buffer.part_id = part.part_id and
   buffer.id = part.id;

