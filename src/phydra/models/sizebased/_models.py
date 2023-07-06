import xso

from .variables import SV, SVArray, SVArraySize

from .forcings import ConstantForcing

from .fluxes.basic import LinearForcingInput, LinearDecay_Dims, QuadraticDecay_Dim_Sum
from .fluxes.growth import MonodGrowth_mu_ConsumerDim
from .fluxes.grazing import SizebasedGrazingKernel_Dims, GrossGrowthEfficiency_MatrixGrazing

NPxZxSizeBased = xso.create({
    # State variables
    'Nutrient': SV,
    'Phytoplankton': SVArray,  #
    'Zooplankton': SVArray,  #

    # Flows:
    'Inflow': LinearForcingInput,

    # Growth
    'Growth': MonodGrowth_mu_ConsumerDim,

    # Grazing
    'Grazing': SizebasedGrazingKernel_Dims,
    'GGE': GrossGrowthEfficiency_MatrixGrazing,

    # Mortality
    'PhytoMortality': LinearDecay_Dims,
    'ZooMortality': QuadraticDecay_Dim_Sum,

    # Forcings
    'N0': ConstantForcing,
})
