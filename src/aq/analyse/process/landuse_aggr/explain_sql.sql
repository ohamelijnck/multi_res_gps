explain select
   buffers.id,

   covs.*,

   ST_Intersection(covs.geom, buffers.buffer_geom) as intersected_geom,
   buffers.buffer_geom,
   buffers.site_geom
into
   orca.ukmap_4326_on_buffer_preprocess_500
from
   orca.location_buffers_500 as buffers,
   orca.ukmap_4326 as covs
where
   ST_Intersects(covs.geom, buffers.buffer_geom);
