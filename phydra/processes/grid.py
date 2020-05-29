import numpy as np
import xsimlab as xs

from ..processes.environments import BaseEnvironment

@xs.process
class BaseGrid:
    pass



@xs.process
class GridXY(BaseGrid):
    """
    This process supplies the Grid dimensions to the Physical Environment
    """
    # Dimension labels and indices
    x_label = xs.variable(default='x')
    y_label = xs.variable(default='y')

    x = xs.index(dims='x')
    y = xs.index(dims='y')

    # Input
    x_dim = xs.variable(intent='in', description='length of dimension, x direction')
    y_dim = xs.variable(intent='in', description='length of dimension, y direction')

    dx = xs.variable(intent='out', description='grid distance in regular grid, x direction')
    dy = xs.variable(intent='out', description='grid distance in regular grid, y direction')

    grid_dims = xs.foreign(BaseEnvironment, 'grid_dims', intent='out')

    def initialize(self):
        self.dx, self.dy = 10 / np.array([self.x_dim, self.x_dim], dtype='float64')

        self.x = np.arange(self.x_dim)
        self.y = np.arange(self.y_dim)

        self.grid_dims = (self.x_dim, self.y_dim)
