import xso

from .variables import SV
from .fluxes import LinearForcingInput, MonodGrowth, LinearDecay_ListInput
from .forcings import ConstantForcing, SinusoidalForcing

NPChemostat = xso.create({
    # State variables
    'Nutrient': SV,
    'Phytoplankton': SV,

    # Flows:
    'Inflow': LinearForcingInput,
    'Outflow': LinearDecay_ListInput,

    # Growth
    'Growth': MonodGrowth,

    # Forcings
    'N0': ConstantForcing
})

NPChemostat_sinu = NPChemostat.update_processes({'N0': SinusoidalForcing})
