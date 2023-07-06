import xso

from .variables import Nutrient, Phytoplankton
from .fluxes import LinearInflow, MonodGrowth, LinearOutflow_ListInput
from .forcings import ConstantExternalNutrient, SinusoidalExternalNutrient

NPChemostat = xso.create({
    # State variables
    'Nutrient': Nutrient,
    'Phytoplankton': Phytoplankton,

    # Flows:
    'Inflow': LinearInflow,
    'Outflow': LinearOutflow_ListInput,

    # Growth
    'Growth': MonodGrowth,

    # Forcings
    'N0': ConstantExternalNutrient
})

NPChemostat_sinu = NPChemostat.update_processes({'N0': SinusoidalExternalNutrient})
