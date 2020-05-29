import xsimlab as xs

from ..processes.grid import BaseGrid, GridXY
from ..processes.environments import BaseEnvironment
from ..processes.components import Nutrient, Phytoplankton
from ..processes.fluxes import PhytoplanktonMortality, NutrientUptake

from ..processes.forcing import ChemostatForcing
from ..processes.forcingfluxes import Mixing
from ..processes.gridfluxes import GridExchange
from ..processes.init import ChemostatGridXYSetup


# ``Bootstrap model`` has the minimal set of processes required to
# simulate a

Bootstrap_model = xs.Model({
    'Grid': BaseGrid
})


# ``NPZD Slab model`` has the minimal set of processes required to
# simulate a

ZeroD_NPZD_Slab_model = xs.Model({
    'Grid': BaseGrid
})


# ``ZeroD NPxZx Chemostat model`` has the minimal set of processes required to
# simulate a

ZeroD_NPxZx_Chemostat_model = xs.Model({
    'Grid': BaseGrid
})


# ``ZeroD NPxZx Slab model`` has the minimal set of processes required to
# simulate a

ZeroD_NPxZx_Slab_model = xs.Model({
    'Grid': BaseGrid
})


# ``GridXY Chemostat model`` has a minimal set of processes for a
# 2-dimensional model, with Nutrients and Phytoplankton as components


GridXY_NP_Chemostat_model = xs.Model({
    'Grid': GridXY,

    'Env': BaseEnvironment,

    'N': Nutrient, 'P': Phytoplankton,

    'NP_uptake': NutrientUptake, 'P_Mortality': PhytoplanktonMortality,

    'FX': ChemostatForcing, 'Mix': Mixing,

    'GX': GridExchange,

    'MS': ChemostatGridXYSetup
})