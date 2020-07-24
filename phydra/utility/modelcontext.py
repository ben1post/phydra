import numpy as np
from collections import defaultdict

class Context:
    """fluxes dict getter/setter that allows sharing properties between model processes
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
    """ This stores model fluxes in a dict of dicts for setup and debugging
    """
    def __init__(self):
        """overwrite ContextDict to default list"""
        self.context = defaultdict(dict)
        self.name = 'Model fluxes dict'

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


class ParameterDict:
    """ This stores parameters initialized for specific parameters in a dict,
    within a dict of the key for specific component
    """
    def __init__(self):
        """overwrite ContextDict to default list"""
        self.parameters = defaultdict(dict)
        self.shapes = defaultdict(dict)
        self.name = 'SV Parameter Dict'

    def __getitem__(self, key):
        return self.parameters[key]

    def __repr__(self):
        return f"{self.name} stores: {self.parameters.items()}"

    def setup_dims(self, key_label, par_label, dims):
        print('setup_dims',dims, key_label, par_label)
        self.parameters[key_label].update({par_label: np.full(dims, object)})
        self.shapes[key_label].update({par_label: np.zeros(dims)})

    def init_param_across_dims(self, comp, param, newvalue, index=None):
        if index != None:
            self.parameters[comp][param][index] = newvalue
        else:
            print('PARAMETER NOT INITIALIZED WITH INDEX')
            if np.array(newvalue).size == 1:
                it = np.nditer(self.shapes[comp][param], flags=['multi_index', 'refs_ok'])
                while not it.finished:
                    self.parameters[comp][param][it.multi_index] = newvalue
                    it.iternext()
            else:
                try:
                    x = newvalue.size
                except AttributeError:
                    raise TypeError(
                        f"When supplying dimensional parameters, make sure to have them in np.array format, not {type(newvalue)}")

                if np.array(newvalue).size == self.shapes[comp][param].size:
                    it = np.nditer(self.shapes[comp][param], flags=['multi_index', 'refs_ok'])
                    while not it.finished:
                        self.parameters[comp][param][it.multi_index] = newvalue[it.multi_index]
                        it.iternext()
                else:
                    raise BaseException(f"dimensions of supplied parameter do not match SV dims \n \
                          needs to be scalar or a numpy array of shape {self.shapes[comp][param].shape}")

    def init_param_range(self, comp, param, minvalue, maxvalue, spacing='linear'):
        size = self.shapes[comp][param].size
        shape = self.shapes[comp][param].shape
        if spacing == 'linear':
            parameter_range = np.linspace(minvalue, maxvalue, size)
        elif spacing == 'log':
            parameter_range = np.logspace(np.log10(minvalue), np.log10(maxvalue), size)
        else:
            raise ValueError("spacing argument needs to be 'linear' or 'log'")

        parameter = np.full(shape, parameter_range)
        print(parameter)

        it = np.nditer(self.shapes[comp][param], flags=['multi_index', 'refs_ok'])
        while not it.finished:
            self.parameters[comp][param][it.multi_index] = parameter[it.multi_index]
            it.iternext()



class SVDimFluxes:
    """ This is a more complex defaultdict, that handles dynamically assigned
    n-dimensional numpy arrays, that contain lists, to which Fluxes (i.e. gekko m.Intermediates)
    can be appended to build the model
    """
    def __init__(self):
        self.fluxes = defaultdict(object)
        self.name = 'SV Dimensional Fluxes'
        self.shape = 'Not Initialized'

    def __getitem__(self, key):
        return self.fluxes[key]

    def __repr__(self):
        return f"{self.name} stores: {self.fluxes.items()}"

    def setup_dims(self, key, fulldims):
        self.shape = np.zeros(fulldims.shape)
        self.fluxes[key] = np.full(fulldims.shape, object)

        it = np.nditer(self.fluxes[key], flags=['multi_index', 'refs_ok'])
        while not it.finished:
            self.fluxes[key][it.multi_index] = list()
            it.iternext()

    def apply_flux(self, key, newvalue, index=None):
        if index != None:
            self.fluxes[key][index].append(newvalue)
        else:
            if np.array(newvalue).size == 1:
                it = np.nditer(self.shape, flags=['multi_index', 'refs_ok'])
                while not it.finished:
                    self.fluxes[key][it.multi_index].append(newvalue)
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
                        self.fluxes[key][it.multi_index].append(newvalue[it.multi_index])
                        it.iternext()
                else:
                    raise BaseException(f"dimensions of value do not match SV dims \n \
                          needs to be scalar or a numpy array of shape {self.shape.shape}")

    def apply_exchange_flux(self, ressource, consumer, flux, indexR=None, indexC=None):
        if indexC != None and indexR != None:
            self.fluxes[ressource][indexR].append(-flux)
            self.fluxes[consumer][indexC].append(flux)
        else:
            raise AttributeError('index not passed to apply_exchange_flux function')


class OldCodeSave:
    pass
    """# Necessary Input:
    init = xs.variable(intent='in')
    dim = xs.variable(intent='in', groups='comp_dim')

    size_min = xs.variable(intent='in')
    size_max = xs.variable(intent='in')

    def initialize_parametersetup(self):
        self.gk_Parameters.setup_dims(self.comp_label, 'Size', self.gk_SVshapes[self.comp_label].shape)
        self.gk_Parameters.init_param_range(self.comp_label, 'Size', self.size_min,
                                            self.size_max, spacing='log')

        print(f"Size Range Initialized for {self.comp_label} with sizes {self.gk_Parameters[self.comp_label]['Size']}")
    """