import numpy as np
import xsimlab as xs

from ..processes.grid import GridXY

from ..processes.environments import BaseEnvironment

from ..processes.components import Nutrient, Phytoplankton


@xs.process
class GridExchange:
    """
    This process collects pairwise interaction between all adjacent gridpoints
    advection equation adapted from https://scipython.com/book/chapter-7-matplotlib/examples/the-two-dimensional-diffusion-equation/
    """
    Model_dims = xs.foreign(BaseEnvironment, 'dims')

    dx = xs.foreign(GridXY, 'dx')
    dy = xs.foreign(GridXY, 'dy')

    N = xs.foreign(Nutrient, 'state')
    P = xs.foreign(Phytoplankton, 'state')

    N_advected = xs.variable(dims=('x', 'y', 'Env', 'N'), intent='out', groups='N_flux')
    P_advected = xs.variable(dims=('x', 'y', 'Env', 'P'), intent='out', groups='P_flux')

    exchange_rate = xs.variable(intent='in')

    def advection(self, state, dt):
        # Propagate with forward-difference in time, central-difference in space
        advect = self.exchange_rate * dt * (
                (state[2:, 1:-1] - 2 * state[1:-1, 1:-1] + state[:-2, 1:-1]) / self.dx2
                + (state[1:-1, 2:] - 2 * state[1:-1, 1:-1] + state[1:-1, :-2]) / self.dy2)
        return advect

    def initialize(self):
        self.N_advected = np.zeros_like(self.N)
        self.P_advected = np.zeros_like(self.P)

        self.dx2 = self.dx ** 2
        self.dy2 = self.dy ** 2

    @xs.runtime(args="step_delta")
    def run_step(self, dt):
        # indexing below defines that the boundaries are note affected by advection (i.e. highly simplified boundary condition -> nutrient source placed in the center)
        self.N_advected[1:-1, 1:-1] = self.advection(self.N, dt)
        self.P_advected[1:-1, 1:-1] = self.advection(self.P, dt)