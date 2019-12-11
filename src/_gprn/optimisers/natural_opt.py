from . import optimizer

class NaturalGradientOptimizer(optimizer.Optimizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def minimize(self, model, var_list=None, session=None, feed_dict=None,  anchor=True, **kwargs):
        pass

