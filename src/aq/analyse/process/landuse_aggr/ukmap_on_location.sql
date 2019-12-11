
drop table if exists orca.ukmap_on_location;

select 
   buffer.id,
   buffer.src, 
   ukmap.landuse,
   ukmap.feature_ty,
   ukmap.geom,
   buffer.buffer_geom,
   buffer.site_geom
into
   orca.ukmap_on_location  
from
   orca.ukmap_4326 as ukmap,
   orca.location_buffers_500 as buffer
where
   ST_Contains(ukmap.geom, buffer.site_geom);
