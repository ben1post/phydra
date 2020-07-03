import numpy as np
import xsimlab as xs
from gekko import GEKKO

from time import process_time
from .main import (Grid0D, Boundary0D)
from ..utility.modelcontext import (ContextDict, GekkoMath, SVDimsDict, FluxesDict, SVDimFluxes, ParameterDict)


@xs.process
class GekkoCore:
    """"""
    m = xs.any_object(description='Gekko model instance')

    def initialize(self):
        print('Initializing Gekko Context')
        self.m = GEKKO(remote=False, name='phydra')  # specific gekko model instance

@xs.process
class GekkoContext:
    """ Inherited by all other model processes to access GekkoCore"""
    m = xs.foreign(GekkoCore, 'm')


@xs.process
class GekkoSolve(GekkoContext):
    """
    SOLVER INTERFACE - for now no input args
    """

    def run_step(self):
        print('SolveInit')
        self.m.options.REDUCE = 3
        self.m.options.NODES = 3
        self.m.options.IMODE = 7

        self.m.options.MAX_MEMORY = 6

        solve_start = process_time()
        self.m.solve(disp=False)  # disp=True) # to print gekko output
        solve_end = process_time()

        print(f"ModelSolve done in {round(solve_end-solve_start,2)} seconds")





#############################################################################################
@xs.process
class OldGekkoContext:
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

#TODO: I have THE idea for a slim functional core using Gekko, so instead of
# tasking storing all my weird model parameter dicts, instead, keep all of that inside a
# wrapped class of GEKKO, so that each model instance corresponds to one m = GEKKO()
# and I can add (if necessary, on top of the lists already present) all comps as attributes of self.m
# e.g. self.m.LightHarvesting = self.m.Intermediate(xxx)
# now if I additionally want to integrate Gekko variables directly with xarray simlab variables
# the first test I need to run is: Can I store a GK_variable within xarray structure from the start,
# so that it automatically references the final values after m.solve() ?
# this will need some trick, of assigning m.SV/m.Intermediate .VALUE before solve (if possible)...

# Once that works, I could try and make it so that instead of having to initialize an xarray variable,
# and then assigning it the respective Gekko var to pass on, that this happens in the same step!
# so every xsimlab variable (or any_object) or whatever, corresponds directly to the Gekko var it references

# And don't forget to allow solve mode 4, run_step, that functions fully with xarray simlab usability!
# might be slow, but great for smaller models..

#TODO Actually been thinking some more
# and what I realised is that, if I really want it to be flexible, I need to create a class that wraps each variable,
# and then this is passed to different "Solvers"-Processes, and that solver process only accesses the
# correct sub-attribute, like for example, when it wants the function, it collects x.function
# or if it wants the gekko attribs, it calls x.gekko_attribs... hm?

# of o can get solve mode 4 to run, with the point above, of not having to unpack the vals, that would be powerful

# also this is how it would be possible, to include some kind of visualisation or
# print of model structure (perhaps even with latex/math?)

# SO, i can make connections between variables..
# this might be a good way to include the multiple env concept
# like: I can instead of storing the entire model at self.m,
# store it in sub-modules (attributes!), like
# self.m.env1 = self.Slab
# .. hm.. might do this later

# m.options.DIAGLEVEL = 4 outputs LATEX FILE OF MODDEL!!!!!




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
        #self.m.options.REDUCE = 3
        #self.m.options.solver = 2
        self.m.options.REDUCE = 3
        self.m.options.NODES = 3
        self.m.options.IMODE = 7

        self.m.options.MAX_MEMORY = 6

        solve_start = process_time()
        self.m.solve(disp=False)  # disp=True) # to print gekko output
        solve_end = process_time()

        print(f"ModelSolve done in {round(solve_end-solve_start,2)} seconds")