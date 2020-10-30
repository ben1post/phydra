from ._version import get_versions

__version__ = get_versions()['version']
del get_versions

from . import components
from . import models

from .backend.variable import variable, parameter, forcing

from .backend.component import comp

from phydra.backend.xsimlabwrappers import setup, create, update_setup
