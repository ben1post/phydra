import numpy as np
import xsimlab as xs
from gekko import GEKKO

from time import process_time
from .main import (Grid0D, Boundary0D)
from ..utility.modelcontext import (ContextDict, GekkoMath, SVDimsDict, FluxesDict, SVDimFluxes, ParameterDict)

@xs.process
class GekkoContext:
    """This process takes care of proper initialization,
    update and clean-up of gekko PDE/ODE internal
    state.
    """
    shape = xs.foreign(Grid0D, 'shape')
    length = xs.foreign(Grid0D, 'length')
    ibc = xs.foreign(Boundary0D, 'ibc')

    m = xs.any_object(description='Gekko model instance')
    context = xs.any_object(description='defaultdict - Gekko model context _ for debugging and checking')

    SVs = xs.any_object(description='defaultdict - Stores all state variables')
    SVshapes = xs.any_object(description='defaultdict - Stores all state variables dimensions')
    Parameters = xs.any_object(description='defaultdict - Stores all parameters in state variable dimesnions')
    Fluxes = xs.any_object(description='defaultdict - Stores all gekko m.Intermediates corresponding to a specific SV')
    Flux_Intermediates = xs.any_object(description='defaultdict - Stores all gekko m.Intermediates corresponding to a '
                                                   'specific ForcingFlux')

    def initialize(self):
        print('Initializing Gekko Context')
        self.m = GEKKO(remote=False)  # specific gekko model instance

        self.context = ContextDict()  # simple defaultdict list store containing additional info
        self.SVs = GekkoMath()  # stores gekko m.SVs by label
        self.SVshapes = SVDimsDict()  # stores dims as np.arrays for iteration over multiple dimensions
        self.Parameters = ParameterDict()
        self.Fluxes = SVDimFluxes()  # stores m.Intermediates with corresponding label, needs to be appended to
        self.Flux_Intermediates = FluxesDict()  # used to retrieve flux output and store

        self.context["shape"] = ('env', self.shape)


@xs.process
class InheritGekkoContext:
    """ This class is a base class that allows all subclasses to access the common GekkoContext"""
    m = xs.foreign(GekkoContext, 'm')
    gk_context = xs.foreign(GekkoContext, 'context')
    gk_SVs = xs.foreign(GekkoContext, 'SVs')
    gk_SVshapes = xs.foreign(GekkoContext, 'SVshapes')
    gk_Parameters = xs.foreign(GekkoContext, 'Parameters')
    gk_Fluxes = xs.foreign(GekkoContext, 'Fluxes')
    gk_Flux_Int = xs.foreign(GekkoContext, 'Flux_Intermediates')
    gridshape = xs.foreign(GekkoContext, 'shape')


@xs.process
class GekkoSolve:
    """This process solves the model, different solvers could be defined
    this one uses IMODE 7, i.e. sequential solve based on supplied timesteps
    disp=False suppresses output of gekko backend
    """
    m = xs.foreign(GekkoContext, 'm')
    gk_context = xs.foreign(GekkoContext, 'context')

    def run_step(self):
        print('SolveInit')

        self.m.options.IMODE = 7

        solve_start = process_time()
        self.m.solve(disp=False)  # disp=True) # to print gekko output
        solve_end = process_time()

        print(f"ModelSolve done in {round(solve_end-solve_start,2)} seconds")