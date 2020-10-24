from ._version import get_versions

__version__ = get_versions()['version']
del get_versions

from . import processes
from . import models

from phydra.core.xsimlabwrappers import setup, create, update_setup

from .core.flux_decorator import sv, param, fx, flux, multiflux
