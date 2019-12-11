drop table if exists orca.location_buffers_500;
drop table if exists orca.location_buffers_200;
drop table if exists orca.location_buffers_100;
drop table if exists orca.location_buffers_1c;
drop table if exists orca.locations;

--get locations

create table orca.locations (
   site_id integer,
   src integer,
   geom geometry
);

insert into orca.locations
select
   id,
   0,
   geom
from
   orca.laqn_sites;

insert into orca.locations
select
   id,
   1,
   geom
from
   orca.diffusion_tube_sites;

insert into orca.locations
select
   id,
   2,
   ST_Centroid(geom)
from
   orca.london_grid;

insert into orca.locations
select
   id,
   3,
   ST_Centroid(geom)
from
   orca.london_grid_high_res;

insert into orca.locations
select
   id,
   4,
   geom
from
   orca.aqe_sites;

ALTER TABLE orca.locations ADD COLUMN id SERIAL PRIMARY KEY;

-- create buffers at locations

\set buffer_diam_500 0.005 -- ~500m
\set buffer_diam_100 0.0005 -- ~10m

select
   sites.id,
   sites.src,
   sites.geom as site_geom,
   ST_Buffer(sites.geom, :buffer_diam_500) buffer_geom
into
   orca.location_buffers_500
from
   orca.locations as sites;

select
   sites.id,
   sites.src,
   sites.geom as site_geom,
   ST_Buffer(sites.geom, 0.002) buffer_geom
into
   orca.location_buffers_200
from
   orca.locations as sites;

select
   sites.id,
   sites.src,
   sites.geom as site_geom,
   ST_Buffer(sites.geom, 0.001) buffer_geom
into
   orca.location_buffers_1c
from
   orca.locations as sites;

select
   sites.id,
   sites.src,
   sites.geom as site_geom,
   ST_Buffer(sites.geom, :buffer_diam_100) buffer_geom
into
   orca.location_buffers_100
from
   orca.locations as sites;

CREATE INDEX location_buffers_500_gix ON orca.location_buffers_500 USING GIST(buffer_geom);
CREATE INDEX location_buffers_100_gix ON orca.location_buffers_100 USING GIST(buffer_geom);
CREATE INDEX location_buffers_200_gix ON orca.location_buffers_200 USING GIST(buffer_geom);
create index location_buffers_1c_gix on orca.location_buffers_1c using GIST(buffer_geom);

