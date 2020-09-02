from ._version import get_versions

__version__ = get_versions()['version']
del get_versions

from . import processes
from . import models

from .utility.xsimlabwrappers import setup, create, update_setup

from .core.flux import flux, sv, param, fx
