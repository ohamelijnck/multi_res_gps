import numpy as np
from db import *
from util.sql_util import *
BUFFER_500 = 0
BUFFER_1C = 1
BUFFER_200 = 2
BUFFER_100 = 3


PREPROCESS_TEMPLATE = """
drop table if exists {schema}.{preprocess_table_name};
select 
   buffers.id,

   {cov_columns},

   ST_Intersection(covs.{cov_geom}, buffers.buffer_geom) as {intersected_geom},
   buffers.buffer_geom,
   buffers.site_geom
into 
   {schema}.{preprocess_table_name}
from
   {schema}.{buffer_table} as buffers,
   {schema}.{covariate_table} as covs
where
   ST_Intersects(covs.{cov_geom}, buffers.buffer_geom);

"""

ON_BUFFER_TEMPLATE = """
drop table if exists {schema}.{on_buffer_table};
select 
   buffer.id,
   {coalesce_columns},
   buffer.buffer_geom
into 
   {schema}.{on_buffer_table}
from
   (
      select
         buffer.id,
         {extract_colums}
      from
         {schema}.{preprocess_table_name} as buffer
      group by
         {group_by}
   ) as covs
right join
   {schema}.{buffer_table} as buffer
on
   buffer.id = covs.id;
"""

class CovOnBuffer(object):
    def __init__(self, config, db, psql_schema ='', buffer_root='', cov_table='', time_column=None, cov_columns=[], run_sql=False, overwrite_preprocess_buffers=False, preprocess_buffer_sizes=[], buffer_sizes=[], generated_columns=[], intersected_geom='geom', cov_geom = 'geom'):
        self.CONFIG = config
        self.db = db
        self.psql_schema = psql_schema
        self.buffer_root = buffer_root
        self.cov_table = cov_table
        self.time_column = time_column
        self.cov_columns = cov_columns
        self.run_sql = run_sql
        self.overwrite_preprocess_buffers = overwrite_preprocess_buffers
        self.preprocess_buffer_sizes = preprocess_buffer_sizes
        self.buffer_sizes = buffer_sizes
        self.generated_columns = np.array(generated_columns)
        self.intersected_geom = intersected_geom
        self.cov_geom = cov_geom
        self.preprocess_use_orginal = True

        self.setup_vars()
        self.setup_time()

        self.total_sql_str = ''
        self.preprocess_table_names = []

        self.setup_cov_columns()

        if self.preprocess_buffer_sizes is not None:
            self.get_max_size_buffer()
            if self.overwrite_preprocess_buffers is True:
                sql = self.generate_preprocess_tables()
                print(sql)
                if self.run_sql:
                    self.db.thread_safe_execute(sql)

            sql = self.generate_on_buffer_table()
            print(sql)
            if self.run_sql:
                self.db.thread_safe_execute(sql)
            
    def setup_time(self):
        if self.time_column:
            self.cov_columns = [self.time_column] + self.cov_columns

    def setup_vars(self):
        self.buffer_size_order = [BUFFER_500, BUFFER_1C, BUFFER_200, BUFFER_100]
        self.buffer_size_names = ['500', '1c', '200', '100']   

    def setup_cov_columns(self):
        if self.cov_columns[0] == '*':
            #select column names from self.cov_table
            sql = "SELECT column_name FROM information_schema.columns WHERE table_schema = '{schema}' AND table_name   = '{table_name}'";
            sql = sql.format(
                schema = self.psql_schema,
                table_name = self.cov_table
            )
            self.db.connect('postgis_test')
            curr = self.db.execute(sql)
            col_names = curr.fetchall()
            self.cov_columns = [row[0] for row in col_names]
            self.db.close()

    def get_column_names(self):
        return self.generated_columns[:, 0]

    def get_buffer_size_names(self):
        a = []
        for size in self.buffer_size_order:
            if size in self.buffer_sizes:
                a.append(self.buffer_size_names[size])
        return a

    def get_max_size_buffer(self):
        for size in self.buffer_size_order:
            if size in self.buffer_sizes:
                self.max_size = size
                return
        self.max_size = 0

    def generate_preprocess_tables(self):
        prev_size = None
        prev_name = None
        #go through biggest to smallest to ensure the tables create in correct order
        total_sql_str = ''
        for size in self.buffer_size_order:
            if size in self.preprocess_buffer_sizes:
                buffer_size_name = self.buffer_size_names[size]

                #we want the max preprocessing buffer size to use the original geom to intersect on,
                #the next sizes only need to work on the buffer that has already been intersected

                if (prev_size is None) or (self.preprocess_use_orginal):
                    cov_table = self.cov_table
                    cov_geom = self.cov_geom
                else:
                    cov_table = prev_name
                    cov_geom = self.intersected_geom

                table_name, sql = self.generate_preprocess_table_template(
                    buffer_size_name=buffer_size_name,
                    cov_table = cov_table,
                    cov_geom = cov_geom
                )

                self.preprocess_table_names.append(table_name)
                total_sql_str += sql

                prev_size = size
                prev_name = table_name
        return total_sql_str


    def get_preprocess_table_name(self, buffer_size_name):
        preprocess_table_name = '{cov_table}_on_buffer_preprocess_{buffer_name}'.format(
            cov_table = self.cov_table,
            buffer_name = buffer_size_name
        )
        return preprocess_table_name

        
    def generate_preprocess_table_template(self, buffer_size_name, cov_table, cov_geom):
        cov_colums = self.to_sql_select_list(self.prefix('covs.',self.cov_columns))

        preprocess_table_name = self.get_preprocess_table_name(buffer_size_name)

        buffer_table = '{buffer_root}{buffer_name}'.format(
            buffer_root = self.buffer_root,
            buffer_name = buffer_size_name
        )

        sql = PREPROCESS_TEMPLATE.format(
            schema = self.psql_schema,
            cov_columns=cov_colums,
            preprocess_table_name = preprocess_table_name,
            buffer_table = buffer_table,
            covariate_table = cov_table,
            intersected_geom = self.intersected_geom,
            cov_geom = cov_geom
        )

        return preprocess_table_name, sql

    def generate_on_buffer_table(self):
        total_sql_str = ''
        for size in self.buffer_size_order:
            if size in self.preprocess_buffer_sizes:
                buffer_size_name = self.buffer_size_names[size]
                sql = self.generate_on_buffer_table_template(buffer_size_name)
                total_sql_str += sql
        return total_sql_str

    def get_on_buffer_table_name(self, buffer_size_name):
        on_buffer_table = '{cov_table}_on_buffer_{buffer_name}'.format(
            cov_table = self.cov_table,
            buffer_name = buffer_size_name
        )
        return on_buffer_table

    def generate_on_buffer_table_template(self, buffer_size_name):
        preprocess_table_name = '{cov_table}_on_buffer_preprocess_{buffer_name}'.format(
            cov_table = self.cov_table,
            buffer_name = buffer_size_name
        )

        buffer_table = '{buffer_root}{buffer_name}'.format(
            buffer_root = self.buffer_root,
            buffer_name = buffer_size_name
        )

        on_buffer_table = self.get_on_buffer_table_name(buffer_size_name=buffer_size_name)

        coalesce_columns = self.suffix(', 0) as ', self.prefix('COALESCE(covs.', self.generated_columns[:, 0]))
        coalesce_columns = self.str_zip(coalesce_columns, self.generated_columns[:, 0])

        if self.time_column:
            coalesce_columns = self.prefix('buffer.',[self.time_column]) + coalesce_columns

        coalesce_columns = self.to_sql_select_list(coalesce_columns)

        extract_colums = self.str_zip(self.suffix(' as ', self.generated_columns[:, 1]), self.generated_columns[:, 0])
        if self.time_column:
            extract_colums = self.prefix('buffer.',[self.time_column]) + extract_colums
        extract_colums = self.to_sql_select_list(extract_colums)

        group_by = ['buffer.id']
        if self.time_column:
            group_by = group_by + self.prefix('\t buffer.',[self.time_column])
        group_by = self.to_sql_select_list(group_by)

        sql = ON_BUFFER_TEMPLATE.format(
            schema = self.psql_schema,
            preprocess_table_name = preprocess_table_name,
            buffer_table = buffer_table,
            on_buffer_table = on_buffer_table, 
            extract_colums = extract_colums,
            coalesce_columns = coalesce_columns,
            group_by = group_by
        )
        return sql



    def prefix(self, pre_str, arr):
        return [pre_str+s for s in arr]

    def suffix(self, post_str, arr):
        return [s+post_str for s in arr]

    def str_zip(self, arr1, arr2):
        a = []
        for i in range(len(arr1)):
            a.append(arr1[i]+arr2[i])
        return a


    def to_sql_select_list(self, arr):
        return ',\n   '.join(arr)


