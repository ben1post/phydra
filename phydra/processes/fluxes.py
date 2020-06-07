import numpy as np
import xsimlab as xs

from ..processes.environments import BaseEnvironment

from ..processes.components import Nutrient, Phytoplankton

import attr


# NOTE: like so! add a CellVariable base class, that each component is initialised as
# this keeps adding to common Grid ! automatically initialises component at each grid point

# what do I need for Term/Fluxes
# generally need to define interactions between CellVariables
# needs to take multiple CellVariables as input, compute a function, and output flux

# for now: solve needs to supply to run_step


#@attr.s
#class Term:
#    var = attr.ib(default=attr.Factory(dict))#

#    _in = attr.ib(default=attr.Factory(dict))

#    @property
#    def solve(self):
#        """This is called to create system of ode"""
#        raise NotImplementedError


@xs.process
class Flux:
    var = xs.any_object("This contains the actual mathematical formulation described by this term")

    _in = xs.any_object("This handles input (and output) to var")

    def initialize(self):
        self.var = 0

    def run_step(self):
        self._delta = self.solve(self.var)


#xs.Model({'Flux':Flux})


@xs.process
class NutrientUptake:
    """This is an example for a MultiComp interacting with a SingularComp"""
    Model_dims = xs.foreign(BaseEnvironment, 'dims')

    N = xs.foreign(Nutrient, 'state')
    P = xs.foreign(Phytoplankton, 'state')

    N_uptake = xs.variable(dims=('x', 'y', 'Env', 'N'), intent='out', groups='N_flux')
    P_growth = xs.variable(dims=('x', 'y', 'Env', 'P'), intent='out', groups='P_flux')

    P_halfsat = xs.foreign(Phytoplankton, 'halfsat')

    NutLim = xs.variable(intent='out')

    @property
    def NutrientLimitation(self):
        lim = self.N / (self.P_halfsat + self.N)
        # print(lim.shape, np.zeros_like(self.N).shape)
        return lim

    def initialize(self):
        self.N_uptake = np.zeros_like(self.N)
        self.P_growth = np.zeros_like(self.P)

    def run_step(self):
        # calculate Nutrient limitation:
        self.NutLim = np.array(self.NutrientLimitation, dtype='float64')

        self.P_growth = self.NutLim * self.P

        # since there is only a single N, that dimension is summed up via "axis = -1"
        self.N_uptake = - np.sum(self.P_growth, axis=-1, keepdims=True)  # negative flux


@xs.process
class PhytoplanktonMortality:
    """Quadratic mortality """
    Model_dims = xs.foreign(BaseEnvironment, 'dims')

    P = xs.foreign(Phytoplankton, 'state')

    P_mortality = xs.variable(dims=('x', 'y', 'Env', 'P'), intent='out', groups='P_flux')

    P_mortality_rate = xs.foreign(Phytoplankton, 'mortality_rate')

    def initialize(self):
        self.P_mortality = np.zeros_like(self.P)

    def run_step(self):
        self.P_mortality = - np.array(self.P_mortality_rate * self.P ** 2, dtype='float64')

