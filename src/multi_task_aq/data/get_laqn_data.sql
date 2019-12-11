drop table if exists temp;
drop table if exists temp_1;

-- HV3

select 
   count(*)
from
   orca.laqn_hourly_data_with_geom as data
where
   data.date >= '2018-06-01' and
   data.date < '2018-07-20' and 
   sitecode='RB7';

select 
   st_y(data.geom) as lat,
   st_x(data.geom) as lon,

   weather.temperature,
   weather.windSpeed,
   weather.windBearing,
   weather.humidity,
   weather.pressure,

   data.*
into
   temp
from
   orca.laqn_hourly_data_with_geom as data,
   orca.locations as locations,
   orca.weather_on_site as weather
where 
   data.date >= '2018-06-02' and
   data.date < '2018-07-20' and
   locations.site_id = data.site_id and locations.src = 0 and
   weather.datetime = data.date and
   weather.id= locations.id;


\copy (select * from temp where sitecode='RB7' order by date) to 'laqn_data_hourly_site_1.csv' CSV HEADER DELIMITER ',';
\copy (select * from temp where sitecode='MY1' order by date) to 'laqn_data_hourly_site_2.csv' CSV HEADER DELIMITER ',';

select 
   sitecode,
   lat,
   lon,
   date_part('year', cast(date as date)) as year,
   date_part('month', cast(date as date)) as month,
   date_part('day', cast(date as date)) as day,

   avg(temperature) as temperature,
   avg(windSpeed) as windSpeed,
   avg(windBearing) as windBearing,
   avg(humidity) as humidity,
   avg(pressure) as pressure,

   avg(no2)  as no2,
   avg(pm10) as pm10,
   avg(pm25) as pm25
into 
   temp_1
from 
   temp
group by (
      date_part('year', cast(date as date)),
      date_part('month', cast(date as date)), 
      date_part('day', cast(date as date)), 
      lat,
      lon,
      sitecode
   );


\copy (select * from temp_1 where sitecode='RB7' order by year, month, day) to 'laqn_data_daily_site_1.csv' CSV HEADER DELIMITER ',';
\copy (select * from temp_1 where sitecode='MY1' order by year, month, day) to 'laqn_data_daily_site_2.csv' CSV HEADER DELIMITER ',';

drop table if exists temp;
drop table if exists temp_1;
