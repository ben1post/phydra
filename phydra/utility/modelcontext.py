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

class SVFluxesDict(Context):
    """ This stores a corresponding numpy array of same dimensions as state variable m.Array
    """
    def __init__(self):
        self.context = defaultdict(list)
        self.name = 'SVFluxes dict'

    def __setitem__(self, key, newvalue):
        self.context[key].append(newvalue)
        #print('DICTITEMS', key, newvalue, self.context)
