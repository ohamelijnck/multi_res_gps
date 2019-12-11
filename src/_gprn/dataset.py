import numpy as np
import tensorflow as tf

from .minibatch import MiniBatch

class Dataset(object):
    def __init__(self):
        self.sources = []
        self.num_sources = 0
        self.inducing_added_flag = False

    def setup(self, context):
        self.context = context
        self.setup_mini_batchers()

    def setup_mini_batchers(self):
        self.mini_batchers = []
        for i in range(self.num_sources):
            x = self.get_raw(source=i, var='x')
            y = self.get_raw(source=i, var='y')
            y_meta = self.get_raw(source=i, var='y_meta')

            batch_size = self.sources[i]['batch_size']

            flag = not(batch_size == None)
            m = MiniBatch(x, y, y_meta, flag, batch_size, self.context)
            self.mini_batchers.append(m)

    def get_mini_batcher(self, source):
        return self.mini_batchers[source]

    def add_source_dict(self, d):
        if 'z' in d:
            self.inducing_added_flag = True
            self.inducing_points = d['z']
        else:
            d['z'] = d['x']
            self.inducing_points = d['x']

        self.add_source_obj(d)


    def add_inducing_points(self, z):
        self.inducing_added_flag = True
        self.inducing_points = z

    def get_num_sources(self):
        return len(self.sources)

    def add_source(self, x, y, y_raw, xs, ys, z, batch_size=None):
        self.add_source_obj({'x': x, 'y': y, 'y_raw': y_raw, 'xs': xs, 'ys': ys, 'batch_size': batch_size})

    def add_source_obj(self, o):
        if 'y_meta' not in o:
            o['y_meta'] = o['y']
        self.sources.append(o)
        self.num_sources += 1

    def next_batch(self, epoch, force_all=False):
        feed_dict = {}
        for i in range(self.num_sources):
            x_ph = self.get_placeholder(source=i, var='x')
            y_ph = self.get_placeholder(source=i, var='y')
            y_nan_ph = self.get_placeholder(source=i, var='y_nan')
            y_meta_ph = self.get_placeholder(source=i, var='y_meta')

            x, y, y_meta = self.get_mini_batcher(source=i).next_batch(epoch, force_all)
            y_nans = (1-np.isnan(y).astype(int))

            feed_dict[x_ph] = x
            feed_dict[y_ph] = y
            feed_dict[y_nan_ph] = y_nans
            feed_dict[y_meta_ph] = y_meta

        return feed_dict

    def get_source(self, s):
        return self.sources[s]

    def get_num_sources(self):
        return len(self.sources)

    def get_num_training(self, source):
        return self.sources[source]['x'].shape[0]

    def get_input_dim(self, source):
        return self.sources[source]['x'].shape[1]

    def get_num_outputs(self, source):
        return self.sources[source]['y'].shape[1]

    def get_num_inducing(self, source):
        if self.inducing_added_flag:
            return self.inducing_points.shape[0]
        return self.get_inducing_points_from_source(source).shape[0]

    def get_inducing_points(self):
        return self.inducing_points

    def get_inducing_points_from_source(self, source):
        if self.inducing_added_flag:
            return self.inducing_points
        return self.sources[source]['z']

    def get_batch_size(self, source):
        b =  self.sources[source]['batch_size']
        if b is None:
            return self.get_num_training(source)
        return b

    def get_x_df(self, source):
        return self.sources[source]['x_df']

    def get_y_train_nans(self, source):
        y_train_nans = (1-np.isnan(self.sources[source]['y']).astype(int))
        return y_train_nans

    def get_meta_dim(self, source):
        return self.sources[source]['y_meta'].shape[1]

    def create_placeholders(self):
        self.placeholders = []
        for i in range(self.num_sources):
            x_shp = np.array(self.get_raw(i, 'x').shape)
            x_shp = np.concatenate(([None], x_shp[1:]))
            x_train = tf.placeholder(tf.float32, shape=x_shp, name="train_inputs_{source}".format(source=i))
            y_train = tf.placeholder(tf.float32, shape=[None, self.get_num_outputs(source=i)], name="train_outputs_{source}".format(source=i))
            x_test = tf.placeholder(tf.float32, shape=x_shp, name="test_inputs_{source}".format(source=i))
            y_train_nans = tf.placeholder(tf.float32, shape=[None, self.get_num_outputs(source=i)], name="train_outputs_nans_{source}".format(source=i))

            y_train_meta = tf.placeholder(tf.float32, shape=[None, self.get_meta_dim(source=i)], name="train_meta_{source}".format(source=i))

            self.placeholders.append({'x': x_train, 'y': y_train, 'xs': x_test, 'y_nan': y_train_nans, 'y_meta': y_train_meta})

    def get_placeholder(self, source, var):
        return self.placeholders[source][var]

    def get_raw(self, source, var):
        return self.sources[source][var]

