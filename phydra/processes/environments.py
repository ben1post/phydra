import numpy as np
import xsimlab as xs


@xs.process
class BaseEnvironment:
    """
    This physical environment provides a base dimension (0D), that is inherited by other components,
    so that all components can be group at grid points of a larger grid

    can be extended to higher dimensions via another grid process, defining 'grid_dims'
    """
    dim_label = xs.variable(default='Env')

    Env = xs.index(dims='Env')

    grid_dims = xs.variable(intent='inout')

    # Input
    dims = xs.variable(intent='out')
    Env_dim = xs.variable(default=1)

    def initialize(self):
        self.Env = np.array([1])

        self.dims = self.grid_dims + (self.Env_dim, 1)