import numpy as np
import xsimlab as xs
from gekko import GEKKO
from collections import defaultdict

from .main import (Grid0D, Boundary0D, Time)
from ..utility.modelcontext import (ContextList, GekkoMath, SVDimsDict)

@xs.process
class GekkoContext:
    """This process takes care of proper initialization,
    update and clean-up of gekko PDE/ODE internal
    state.
    """
    time = xs.foreign(Time, 'days')

    shape = xs.foreign(Grid0D, 'shape')
    length = xs.foreign(Grid0D, 'length')
    ibc = xs.foreign(Boundary0D, 'ibc')

    m = xs.any_object(description='Gekko model instance')
    context = xs.any_object(description='defaultdict - Gekko model context _ for debugging and checking')

    SVs = xs.any_object(description='defaultdict - Stores all state variables')
    SVshapes = xs.any_object(description='defaultdict - Stores all state variables dimensions')
    Fluxes = xs.any_object(description='defaultdict - Stores all gekko m.Intermediates corresponding to a specific SV')

    def initialize(self):
        print('Initializing Gekko Context')
        self.m = GEKKO()  # specific gekko model instance

        self.context = ContextList()  # simple defaultdict list store containing additional info
        self.SVs = GekkoMath()  # stores gekko m.SVs by label
        self.SVshapes = SVDimsDict()  # stores dims as np.arrays for iteration over multiple dimensions
        self.Fluxes = SVDimsDict()  # stores m.Intermediates with corresponding label, needs to be appended to

        self.m.time = self.time

        self.context["shape"] = self.shape


@xs.process
class GekkoSolve:
    """This process solves the model, different solvers could be defined
    this one uses IMODE 7, i.e. sequential solve based on supplied timesteps
    disp=False suppresses output of gekko backend
    """
    m = xs.foreign(GekkoContext, 'm')
    gk_context = xs.foreign(GekkoContext, 'context')

    #gk_SVs = xs.foreign(GekkoContext, 'SVs')
    #gk_SVshapes = xs.foreign(GekkoContext, 'SVshapes')
    #gk_Fluxes = xs.foreign(GekkoContext, 'Fluxes')

    gridshape = xs.foreign(GekkoContext, 'shape')

    def run_step(self):
        print('SolveInit')

        print(self.gk_context)

        self.m.options.IMODE = 7
        self.m.solve(disp=False)

        print('ModelSolve done')