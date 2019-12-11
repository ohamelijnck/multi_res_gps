from ..kernels import SE
class Context(object):
    def __init__(self):
        self.default_values()

    def default_values(self):
        self.batch_with_replace = None
        self.num_latent = 1
        self.num_outputs = 1
        self.num_weights = 1
        self.num_components = 1
        self.use_inducing_flag = False
        self.kern_f = [None]
        self.kern_w = [[None]]
        self.use_diag_covar = False 
        self.jitter = 1e-6
        self.whiten = False
        self.log_transform = False
        self.train_inducing_points_flag = False
        self.split_optimise = False

        self.multi_res = False

        self.sigma_y_train_flag = False
        self.sigma_y_init = [-2.0]
        self.sigma_f_init = None

        self.save_parameters = False
        self.plot_posterior = False
        self.constant_w=False
        


    def load_context(self):
        pass

    def save_context(self):
        pass

