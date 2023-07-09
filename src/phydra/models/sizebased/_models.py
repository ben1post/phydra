import xso

from .variables import Nutrient, PhytoSizeSpectrum, ZooSizeSpectrum

from .forcings import ConstantExternalNutrient

from .fluxes.basic import LinearForcingInput, LinearPhytoMortality, QuadraticZooMortality
from .fluxes.growth import MonodGrowth_SizeBased
from .fluxes.grazing import SizebasedGrazingMatrix, GrossGrowthEfficiency_MatrixGrazing

NPxZxSizeBased = xso.create({
    # State variables
    'Nutrient': Nutrient,
    'Phytoplankton': PhytoSizeSpectrum,
    'Zooplankton': ZooSizeSpectrum,

    # Flows:
    'Inflow': LinearForcingInput,

    # Growth
    'Growth': MonodGrowth_SizeBased,

    # Grazing
    'Grazing': SizebasedGrazingMatrix,
    'GGE': GrossGrowthEfficiency_MatrixGrazing,

    # Mortality
    'PhytoMortality': LinearPhytoMortality,
    'ZooMortality': QuadraticZooMortality,

    # Forcings
    'N0': ConstantExternalNutrient,
})
