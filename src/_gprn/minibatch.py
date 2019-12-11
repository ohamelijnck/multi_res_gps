import tensorflow as tf
import numpy as np
import math
from . import util

class MiniBatch(object):
    def __init__(self, x, y, y_meta, batch_flag, batch_size, context):
        self.context = context
        self.flag = batch_flag
        self.seed = self.context.seed
        self.batch_size = batch_size
        self.with_replace = self.context.batch_with_replace
        self.x = x
        self.y = y
        self.y_meta = y_meta

    def next_batch(self, epoch, force_all=False):
        if force_all or (self.flag is False): 
            return self.x, self.y, self.y_meta
        return self._batch(epoch)

    def _batch(self, epoch):
        np.random.seed(self.seed+epoch)
        idx = np.random.choice(list(range(self.x.shape[0])), self.batch_size, replace=self.with_replace)

        x = self.x[idx]
        y = self.y[idx]
        y_meta = self.y_meta[idx]
        return x, y, y_meta





 



