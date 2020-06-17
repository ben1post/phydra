import numpy as np
from collections import defaultdict

class Context:
    """context dict getter/setter that allows sharing properties between model processes
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


class ContextDict(Context):
    """ This stores model context in a dict of dicts for setup and debugging
    """
    def __init__(self):
        """overwrite ContextDict to default list"""
        self.context = defaultdict(dict)
        self.name = 'Model context dict'

    def __setitem__(self, key, args):
        label, value = args
        self.context[key].update({label: value})


class GekkoMath(Context):
    """ This stores gekko m.intermediates
    """
    def __init__(self):
        self.context = defaultdict(object)
        self.name = 'Gekko math dict'


class SVDimsDict(Context):
    """ This stores a corresponding numpy array of same dimensions as state variable m.Array
    """
    def __init__(self):
        self.context = defaultdict(np.array)
        self.name = 'SVDims dict'


class FluxesDict(Context):
    """ This stores a corresponding numpy array of same dimensions as state variable m.Array
    """
    def __init__(self):
        self.context = defaultdict(list)
        self.name = 'SVFluxes dict'

    def __setitem__(self, key, newvalue):
        self.context[key].append(newvalue)


class SVDimFluxes:
    """ This is a more complex defaultdict, that handles dynamically assigned
    n-dimensional numpy arrays, that contain lists, to which Fluxes (i.e. gekko m.Intermediates)
    can be appended to build the model
    """
    def __init__(self):
        self.context = defaultdict(object)
        self.name = 'SV Dimensional Fluxes'
        self.shape = 'Not Initialized'

    def __getitem__(self, key):
        return self.context[key]

    def __repr__(self):
        return f"{self.name} stores: {self.context.items()}"

    def setup_dims(self, key, fulldims):
        self.shape = np.zeros(fulldims.shape)
        self.context[key] = np.full(fulldims.shape, object)

        it = np.nditer(self.context[key], flags=['multi_index', 'refs_ok'])
        while not it.finished:
            self.context[key][it.multi_index] = list()
            it.iternext()

    def apply_across_dims(self, key, newvalue, index=None):
        if index != None:
            self.context[key][index].append(newvalue)
        else:
            if np.array(newvalue).size == 1:
                it = np.nditer(self.shape, flags=['multi_index', 'refs_ok'])
                while not it.finished:
                    self.context[key][it.multi_index].append(newvalue)
                    it.iternext()
            else:
                try:
                    x = newvalue.size
                except AttributeError:
                    raise TypeError(
                        f"When supplying dimensional fluxes, make sure to have them in np.array format, not {type(newvalue)}")

                if np.array(newvalue).size == self.shape.size:
                    it = np.nditer(self.shape, flags=['multi_index', 'refs_ok'])
                    while not it.finished:
                        self.context[key][it.multi_index].append(newvalue[it.multi_index])
                        it.iternext()
                else:
                    raise BaseException(f"dimensions of value do not match SV dims \n \
                          needs to be scalar or a numpy array of shape {self.shape.shape}")

    def apply_exchange_flux(self, input, output, flux, indexC=None, indexR=None):
        if indexC != None and indexR != None:
            self.context[input][indexR].append(-flux)
            self.context[output][indexC].append(flux)