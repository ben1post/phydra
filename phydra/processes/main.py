# This will be the working ground for the next instance of the phydra core code, that will be used to power the first publication

from gekko import GEKKO
import numpy as np
import xsimlab as xs

from collections import defaultdict

# from .grid import Grid0D
# from .boundary import Boundary0D

class ContextDict:
    """context dict getter/setter that collects processes & parameters
        and creates full gekko model from collected arguments
        """
    def __init__(self):
        self.context = defaultdict()
        self.name = 'baseclass'

    def __getitem__(self, key):
        return self.context[key]

    def __setitem__(self, key, newvalue):
        self.context[key] = newvalue

    def __repr__(self):
        return f"{self.name} stores: {self.context.items()}"

class ContextList(ContextDict):
    """ This stores all model context in lists for setup and debugging
    """
    def __init__(self):
        self.context = defaultdict(list)
        self.name = 'Model context dict'

    def __setitem__(self, key, newvalue):
        self.context[key].append(newvalue)

class GekkoMath(ContextDict):
    """ This stores gekko m.intermediates
    """
    def __init__(self):
        self.context = defaultdict(object)
        self.name = 'Gekko math dict'

class SVDimsDict(ContextDict):
    """ This stores a corresponding numpy array of same dimensions as state variable m.Array
    """
    def __init__(self):
        self.context = defaultdict(np.array)
        self.name = 'SVDims dict'

class FluxesDict(ContextDict):
    """ This stores a corresponding numpy array of same dimensions as state variable m.Array
    """
    def __init__(self):
        self.context = defaultdict(np.array(dtype='object'))
        self.name = 'SVDims dict'



@xs.process
class Time:
    days = xs.variable(dims='time', description='time in days')

    # for indexing xarray IO objects
    time = xs.index(dims='time', description='time in days')

    def initialize(self):
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
        self.m = GEKKO()  # specific gekko model instance
        self.context = ContextList()  # simple defaultdict list store containing additional info
        self.SVs = GekkoMath()  # stores gekko m.SVs by label
        self.SVshapes = SVDimsDict()  # stores dims as np.arrays for iteration over multiple dimensions
        self.Fluxes = SVDimsDict()  # stores m.Intermediates with corresponding label, needs to be appended to

        self.m.time = self.time

        self.context["shape"] = self.shape

        # freeze context defaultdict after full initialization:
        #self.context.default_factory = None

    def run_step(self):
        print('step')


#TODO:
# So I want every instance of a component to supply it's own index
# this should be possible somehow!
# perhaps I need to simply wrap the Component process? !


@xs.process
class Component:
    m = xs.foreign(GekkoContext, 'm')
    gk_context = xs.foreign(GekkoContext, 'context')

    gk_SVs = xs.foreign(GekkoContext, 'SVs')
    gk_SVshapes = xs.foreign(GekkoContext, 'SVshapes')
    gk_Fluxes = xs.foreign(GekkoContext, 'Fluxes')

    gridshape = xs.foreign(GekkoContext, 'shape')

    #label = xs.variable(intent='in')
    init = xs.variable(intent='in')

    def initialize(self):

        # add label to gk_context - components list:
        self.gk_context['components'] = (self.label, self.dim)

        # define np.array of full dimensions for this component:
        self.FullDims = np.zeros((self.gridshape, self.dim))

        # add to SVDims dict:
        self.gk_SVshapes[self.label] = self.FullDims
        self.gk_Fluxes[self.label] = np.array(self.FullDims, dtype='object')

        print(self.gk_SVshapes[self.label])

        # define m.SV array in full model dimensions, add to SV dict:
        self.gk_SVs[self.label] = self.m.Array(self.m.SV, (self.FullDims.shape))
        print(self.gk_SVs)

        # initialize SV m.Array through FullDims multi_index
        it = np.nditer(self.FullDims, flags=['multi_index'])
        while not it.finished:
            self.gk_SVs[self.label][it.multi_index].value = self.init
            it.iternext()


@xs.process
class Flux:
    # THIS kind of flux needs to check if dims are the same
    # provide different kind of fluxes, that check dims and work accordingly!
    m = xs.foreign(GekkoContext, 'm')
    gk_context = xs.foreign(GekkoContext, 'context')

    gk_SVs = xs.foreign(GekkoContext, 'SVs')
    gk_SVshapes = xs.foreign(GekkoContext, 'SVshapes')
    gk_Fluxes = xs.foreign(GekkoContext, 'Fluxes')

    growthrate = xs.variable(intent='in')
    c1_label = xs.variable(intent='in')
    c2_label = xs.variable(intent='in')

    def initialize(self):
        # get SVs
        C1 = self.gk_SVs[self.c1_label]
        print(C1)
        C2 = self.gk_SVs[self.c2_label]
        print(C2)

        print(self.gk_Fluxes[self.c1_label])
        print(self.gk_Fluxes[self.c2_label])

        # define flux for all dimensions of SV
        it1 = np.nditer(self.gk_SVshapes[self.c1_label], flags=['multi_index', 'refs_ok'])
        it2 = np.nditer(self.gk_SVshapes[self.c2_label], flags=['multi_index', 'refs_ok'])
        while not it1.finished:
            while not it2.finished:
                self.gk_Fluxes[self.c1_label][it1.multi_index] = self.m.Intermediate(C1[it1.multi_index] * self.growthrate)
                self.gk_Fluxes[self.c2_label][it2.multi_index] = self.m.Intermediate(C2[it2.multi_index] * self.growthrate)
                it1.iternext()
                it2.iternext()

        it1 = np.nditer(self.gk_SVshapes[self.c1_label], flags=['multi_index', 'refs_ok'])
        it2 = np.nditer(self.gk_SVshapes[self.c2_label], flags=['multi_index', 'refs_ok'])
        while not it1.finished:
            while not it2.finished:
                self.m.Equation(
                    self.gk_SVs[self.c1_label][it1.multi_index].dt() == \
                    self.gk_Fluxes[self.c1_label][it1.multi_index])
                self.m.Equation(
                    self.gk_SVs[self.c2_label][it2.multi_index].dt() == \
                    self.gk_Fluxes[self.c2_label][it2.multi_index])
                it1.iternext()
                it2.iternext()

        print(self.gk_Fluxes[self.c1_label])
        print(self.gk_Fluxes[self.c2_label])


@xs.process
class GekkoSolve:
    m = xs.foreign(GekkoContext, 'm')
    gk_context = xs.foreign(GekkoContext, 'context')

    gk_SVs = xs.foreign(GekkoContext, 'SVs')
    gk_SVshapes = xs.foreign(GekkoContext, 'SVshapes')
    gk_Fluxes = xs.foreign(GekkoContext, 'Fluxes')

    gridshape = xs.foreign(GekkoContext, 'shape')

    # get all dims as xs.group, then use labels to create index?
    components = xs.group('component')
    time = xs.foreign(Time,'time')


    output1 = xs.variable(intent='out', dims=[(),('c1','time'),('c1','env','time')])
    output2 = xs.variable(intent='out', dims=('c2','time'))

    def initialize(self):
        print('SolveInit')
        #dimshape = ((len(self.time),) + self.gk_SVshapes['c1'].shape)
        #print(dimshape)
        #self.output1 = np.zeros(dimshape, dtype='float64')
        #self.output2 = np.zeros_like((self.gk_SVshapes['c2'].shape, len(self.time)), dtype='float64')

        print('finalize')
        print(self.gk_context['components'])

        self.m.options.IMODE = 7
        self.m.solve(disp=False)

        print('XXX')

        out = []
        _it = np.nditer(self.gk_SVshapes['c1'], flags=['multi_index', 'refs_ok'])
        while not _it.finished:
            out.append([val for val in self.gk_SVs['c1'][_it.multi_index].value])
            _it.iternext()

        self.output1 = np.array(out, dtype='float64')

        print('filled_output')
        print(self.output1)

    def finalize_x(self):
        # clean data here, not sure how yet.
        # after full initialization build model:
        print('finalize')
        print(self.gk_context['components'])

        self.m.options.IMODE = 7
        self.m.solve(disp=False)


        print('XXX')

        out = []
        _it = np.nditer(self.gk_SVshapes['c1'], flags=['multi_index', 'refs_ok'])
        while not _it.finished:
            out.append([val for val in self.gk_SVs['c1'][_it.multi_index].value])
            _it.iternext()

        self.output1 = np.array(out, dtype='float64')

        print('filled_output')
        print(self.output1)
        #i = 0
        #for label, dim in self.gk_context['components']:
        #    out = np.empty_like(self.gk_SVshapes[label])
        #    print(out)
        #    _it = np.nditer(self.gk_SVshapes[label], flags=['multi_index', 'refs_ok'])
        #    while not _it.finished:
        #        print('iterator:')
        #        out[_it.index] = [val for val in self.gk_SVs[label][_it.multi_index].value]
        #        _it.iternext()
        #    print(self.outputs[i])
        #    self.outputs[i] = np.array(self.out)
        #    i += 1
        #    print('HEEE')
        #    print(label, dim)


        print(self.gk_SVs)
        print('hey')
        print(self.gk_SVs['D'])