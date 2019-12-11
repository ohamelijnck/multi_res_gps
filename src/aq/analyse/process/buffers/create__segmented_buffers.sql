drop table if exists orca.location_buffers;
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
   orca.diffusion_tube_sites_unique;


insert into orca.locations
select
   id,
   2,
   ST_Centroid(geom)
from
   orca.london_grid;

ALTER TABLE orca.locations ADD COLUMN id SERIAL PRIMARY KEY;

-- create buffers at locations

\set buffer_diam 0.005 -- ~500m

drop function if exists add_points(geometry, geometry, int);
create or replace function add_points(
   p1 geometry, p2 geometry, srid int,
   OUT p3 geometry
)
returns geometry as
$$
select ST_SetSRID(ST_MakePoint(ST_X(p1)+ST_X(p2), ST_Y(p1)+ST_Y(p2)), srid) as p3;
$$ LANGUAGE sql IMMUTABLE STRICT;

select
   parts.id as id,
   parts.site_id as site_id,
   parts.src,
   (parts.dumped_geom).path[1]::text as part_id,
   (parts.dumped_geom).geom as buffer_geom,
   parts.geom as site_geom
into
   orca.location_buffers
from
   (
      select 
         sites.id as id,
         sites.site_id as site_id,
         sites.src,
         ST_Dump(
            ST_Split(
               ST_Split(
                  ST_Split(
                     ST_Split(
                        ST_Buffer(sites.geom, :buffer_diam),
                        ST_MakeLine(add_points(sites.geom, ST_MakePoint(:buffer_diam*2, 0.0), 4326), add_points(sites.geom, ST_MakePoint(-:buffer_diam*2, 0.0), 4326)) 
                     ),
                     ST_MakeLine(add_points(sites.geom,ST_MakePoint(0.0, :buffer_diam*2), 4326), add_points(sites.geom, ST_MakePoint(0.0, -:buffer_diam*2), 4326)) 
                  ),
                  ST_MakeLine(add_points(sites.geom,ST_MakePoint(:buffer_diam*2, :buffer_diam*2), 4326), add_points(sites.geom, ST_MakePoint(-:buffer_diam*2, -:buffer_diam*2), 4326)) 
               ),
               ST_MakeLine(add_points(sites.geom,ST_MakePoint(-:buffer_diam*2, :buffer_diam*2), 4326), add_points(sites.geom, ST_MakePoint(:buffer_diam*2, -:buffer_diam*2), 4326)) 
            )
         ) as dumped_geom,
         sites.geom geom
      from
         orca.locations as sites
   ) as parts;

CREATE INDEX location_buffers_gix ON orca.location_buffers USING GIST(buffer_geom);

