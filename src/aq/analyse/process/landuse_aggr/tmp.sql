drop table if exists orca.ukmap_4326_on_buffer_preprocess_1c;
select
   buffers.id,

   covs.gid,
   covs.objectid,
   covs.geographic,
   covs.geograph_1,
   covs.date_of_fe,
   covs.feature_ty,
   covs.landuse,
   covs.altertativ,
   covs.owner_user,
   covs.building_n,
   covs.primary_nu,
   covs.primary__1,
   covs.secondary_,
   covs.secondar_1,
   covs.number_end,
   covs.number_e_1,
   covs.road_name_,
   covs.road_nam_1,
   covs.locality_n,
   covs.area_name,
   covs.county_reg,
   covs.country,
   covs.postcode,
   covs.address_ra,
   covs.blpu_numbe,
   covs.address_ty,
   covs.cartograph,
   covs.name_of_po,
   covs.descriptio,
   covs.retail_cla,
   covs.retail_des,
   covs.above_reta,
   covs.road_numbe,
   covs.catrograph,
   covs.source_of_,
   covs.height_of_,
   covs.height_o_1,
   covs.calcaulate,
   covs.shape_leng,
   covs.shape_area,
   covs.geom,

   ST_Intersection(covs.geom, buffers.buffer_geom) as intersected_geom,
   buffers.buffer_geom,
   buffers.site_geom
into
   orca.ukmap_4326_on_buffer_preprocess_1c
from
   orca.location_buffers_1c as buffers,
   orca.ukmap_4326 as covs
where
   ST_Intersects(covs.geom, buffers.buffer_geom);


drop table if exists orca.ukmap_4326_on_buffer_preprocess_100;
select
   buffers.id,

   covs.gid,
   covs.objectid,
   covs.geographic,
   covs.geograph_1,
   covs.date_of_fe,
   covs.feature_ty,
   covs.landuse,
   covs.altertativ,
   covs.owner_user,
   covs.building_n,
   covs.primary_nu,
   covs.primary__1,
   covs.secondary_,
   covs.secondar_1,
   covs.number_end,
   covs.number_e_1,
   covs.road_name_,
   covs.road_nam_1,
   covs.locality_n,
   covs.area_name,
   covs.county_reg,
   covs.country,
   covs.postcode,
   covs.address_ra,
   covs.blpu_numbe,
   covs.address_ty,
   covs.cartograph,
   covs.name_of_po,
   covs.descriptio,
   covs.retail_cla,
   covs.retail_des,
   covs.above_reta,
   covs.road_numbe,
   covs.catrograph,
   covs.source_of_,
   covs.height_of_,
   covs.height_o_1,
   covs.calcaulate,
   covs.shape_leng,
   covs.shape_area,
   covs.geom,

   ST_Intersection(covs.geom, buffers.buffer_geom) as intersected_geom,
   buffers.buffer_geom,
   buffers.site_geom
into
   orca.ukmap_4326_on_buffer_preprocess_100
from
   orca.location_buffers_100 as buffers,
   orca.ukmap_4326 as covs
where
   ST_Intersects(covs.geom, buffers.buffer_geom);

