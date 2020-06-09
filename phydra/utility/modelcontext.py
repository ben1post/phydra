import numpy as np
from collections import defaultdict

class ContextDict:
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


class ContextList(ContextDict):
    """ This stores model context in lists for setup and debugging
    """
    def __init__(self):
        """overwrite ContextDict to default list"""
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