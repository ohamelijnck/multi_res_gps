import numpy as np
import tensorflow as tf

from . import Model

class Composite(Model):
    def __init__(self, context):
        super(Composite, self).__init__()
        self.context = context

    def setup_composite_variables(self):
        pass
