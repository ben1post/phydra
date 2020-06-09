import xsimlab as xs

import phydra
from phydra.processes.main import Time, Grid0D,Boundary0D
from phydra.processes.fluxes import Flux
from phydra.processes.components import Component
from phydra.processes.gekkocontext import GekkoContext, GekkoSolve

from phydra.utility.xsimlabwrappers import phydra_setup, createMultiComp


# ``Bootstrap model`` has the minimal set of processes required to
# simulate a

Bootstrap_model = xs.Model({
    'Grid': Grid0D
})


# ``NPZD Slab model`` has the minimal set of processes required to
# simulate a

ZeroD_NPZD_Slab_model = xs.Model({
    'Grid': Grid0D
})


# ``ZeroD NPxZx Chemostat model`` has the minimal set of processes required to
# simulate a

ZeroD_NPxZx_Chemostat_model = xs.Model({
    'Grid': Grid0D
})


# ``ZeroD NPxZx Slab model`` has the minimal set of processes required to
# simulate a

ZeroD_NPxZx_Slab_model = xs.Model({
    'Grid': Grid0D
})


# ``GridXY Chemostat model`` has a minimal set of processes for a
# 2-dimensional model, with Nutrients and Phytoplankton as components


GridXY_NP_Chemostat_model = xs.Model({
    'Grid': Grid0D,

    'Env': Grid0D,

    'N': Grid0D, 'P': Grid0D,

    'NP_uptake': Grid0D, 'P_Mortality': Grid0D,

    'FX': Grid0D, 'Mix': Grid0D,

    'GX': Grid0D,

    'MS': Grid0D
})