import numpy as np
import xsimlab as xs

from .gekkocontext import GekkoContext
from .components import Time
from .main import Grid0D


@xs.process
class Forcing:
    """Here we initialise the Nutrient Input Forcing (also spatially defined)"""
    m = xs.foreign(GekkoContext, 'm')
    gk_context = xs.foreign(GekkoContext, 'context')

    mld = xs.variable(dims=('time'), intent='out', static=True)
    par = xs.variable(dims=('time'), intent='out', static=True)
    sst = xs.variable(dims=('time'), intent='out', static=True)

    time = xs.foreign(Time, 'days')

    def initialize(self):
        self.mld = np.cos(self.time / 365 * np.pi * 2) * 100 + 200
        self.par = np.sin(self.time / 365 * np.pi) * 50 + 0
        self.sst = np.sin(self.time / 365 * np.pi) * 10 + 10
        print(self.mld)