import xso

from .variables import StateVariable
from .fluxes import LinearInflow, MonodGrowth, LinearOutflow_ListInput
from .forcings import ConstantExternalNutrient, SinusoidalExternalNutrient

NPChemostat = xso.create({
    # State variables
    'Nutrient': StateVariable,
    'Phytoplankton': StateVariable,

    # Flows:
    'Inflow': LinearInflow,
    'Outflow': LinearOutflow_ListInput,

    # Growth
    'Growth': MonodGrowth,

    # Forcings
    'N0': ConstantExternalNutrient
})

NPChemostat_sinu = NPChemostat.update_processes({'N0': SinusoidalExternalNutrient})
