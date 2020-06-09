import numpy as np
import xsimlab as xs

from .main import Grid0D


@xs.process
class ChemostatForcing:
    """Here we initialise the Nutrient Input Forcing (also spatially defined)"""
    Model_dims = xs.foreign(Grid0D, 'dims')

    N_0 = xs.variable(dims=('x', 'y', 'Env'), intent='out', static=True)

    def initialize(self):
        # initialize empty array
        self._N_0 = np.tile(np.array(0., dtype='float64'), self.Model_dims)

        # calculate the center area of grid
        halfway_x = int(self._N_0.shape[0] / 2)
        halfway_y = int(self._N_0.shape[1] / 2)
        dy_dx = int(sum([self._N_0.shape[0], self._N_0.shape[1]]) / 20)

        # add nutrient input at some cells (concentration 5)
        self._N_0[halfway_x - dy_dx:halfway_x + dy_dx, halfway_y - dy_dx:halfway_y + dy_dx, :] = np.array(5,
                                                                                                          dtype='float64')

        self.N_0 = self._N_0


