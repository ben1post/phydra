import xsimlab as xs

import phydra
from phydra.processes.main import GekkoCore, GekkoSequentialSolve



# ``Bootstrap model`` has the minimal set of processes required to
# simulate a

Bootstrap_model = xs.Model({
    'Grid': GekkoCore
})


# ``NPZD Slab model`` has the minimal set of processes required to
# simulate a

ZeroD_NPZD_Slab_model = xs.Model({
    'Grid': GekkoCore
})


# ``ZeroD NPxZx Chemostat model`` has the minimal set of processes required to
# simulate a

ZeroD_NPxZx_Chemostat_model = xs.Model({
    'Grid': GekkoCore
})


# ``ZeroD NPxZx Slab model`` has the minimal set of processes required to
# simulate a

ZeroD_NPxZx_Slab_model = xs.Model({
    'Grid': GekkoCore
})


# ``GridXY Chemostat model`` has a minimal set of processes for a
# 2-dimensional model, with Nutrients and Phytoplankton as components


GridXY_NP_Chemostat_model = xs.Model({
    'Grid': GekkoCore,

    'Env': GekkoCore,

    'N': GekkoCore, 'P': GekkoCore,

    'NP_uptake': GekkoCore, 'P_Mortality': GekkoCore,

    'FX': GekkoCore, 'Mix': GekkoCore,

    'GX': GekkoCore,

    'MS': GekkoCore
})