import numpy as np
import xsimlab as xs

@xs.process
class Time:
    days = xs.variable(dims='time', description='time in days')

    # for indexing xarray IO objects
    time = xs.index(dims='time', description='time in days')

    def initialize(self):
        print('Initializing Model Time')
        self.time = self.days

@xs.process
class Grid0D:
    """Base class, higher dim grids need to inherit and modify"""
    shape = xs.variable(default=1)
    length = xs.variable(default=1)


@xs.process
class Boundary0D:
    """Base class, higher dim grids need to inherit and modify"""
    ibc = xs.variable(default=None, description='Initial Boundary Conditions')
