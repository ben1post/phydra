import xso

from .variables import SV
from .forcings import IrradianceFromLat, StationForcingFromFile
from .fluxes.basic import LinearExchange, QuadraticExchange, QuadraticDecay

from .fluxes.mixing import (Mixing_K, SlabUpwelling_KfromGroup,
                            SlabMixing_KfromGroup, SlabSinking)

from .fluxes.growth import (EMPOWER_Growth_ML, EMPOWER_Monod_ML, EMPOWER_Eppley_ML,
                            EMPOWER_Smith_Anderson3Layer_ML,
                            EMPOWER_Smith_LambertBeer_ML)

from .fluxes.grazing import (HollingTypeIII_ResourcesListInput_Consumption2Group,
                             GrossGrowthEfficiency)

NPZDSlabOcean = xso.create({
    # State variables
    'Nutrient': SV,
    'Phytoplankton': SV,
    'Zooplankton': SV,
    'Detritus': SV,

    # Mixing:
    'K': Mixing_K,
    'Upwelling': SlabUpwelling_KfromGroup,
    'Mixing': SlabMixing_KfromGroup,
    'Sinking': SlabSinking,

    # Growth
    'Growth': EMPOWER_Growth_ML,
    'Nut_lim': EMPOWER_Monod_ML,
    'Light_lim': EMPOWER_Smith_LambertBeer_ML,
    'Temp_lim': EMPOWER_Eppley_ML,

    # Grazing
    'Grazing': HollingTypeIII_ResourcesListInput_Consumption2Group,
    'GGE': GrossGrowthEfficiency,

    # Mortality
    'PhytoLinMortality': LinearExchange,
    'PhytoQuadMortality': QuadraticExchange,
    'ZooLinMortality': LinearExchange,
    'HigherOrderPred': QuadraticDecay,
    'DetRemineralisation': LinearExchange,

    # Forcings
    'Irradiance': IrradianceFromLat,
    'Forcings': StationForcingFromFile,
})

NPZDSlabOcean_3layer = NPZDSlabOcean.update_processes({'Light_lim': EMPOWER_Smith_Anderson3Layer_ML})
