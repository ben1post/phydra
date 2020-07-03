import numpy as np
import xsimlab as xs

from .gekkocontext import GekkoContext

@xs.process
class Time(GekkoContext):
    days = xs.variable(dims='time', description='time in days')
    # for indexing xarray IO objects
    time = xs.index(dims='time', description='time in days')

    def initialize(self):
        print('Initializing Model Time')
        self.time = self.days

        # provide time steps that model is explicitly solved at:
        self.m.time = self.time

        # add variable keeping track of time within model:
        self.m.timevar = self.m.Var(0, lb=0)
        self.m.Equation(self.m.timevar.dt() == 1)


###############################################################
@xs.process
class Grid0D:
    """Base class, higher dim grids need to inherit and modify"""
    shape = xs.variable(default=1)
    length = xs.variable(default=1)

@xs.process
class Boundary0D:
    """Base class, higher dim grids need to inherit and modify"""
    ibc = xs.variable(default=None, description='Initial Boundary Conditions')
