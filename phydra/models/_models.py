import xsimlab as xs

import phydra
from phydra.processes.main import ModelCore



# ``Bootstrap model`` has the minimal set of processes required to
# simulate a

Bootstrap_model = xs.Model({
    'Grid': ModelCore
})


# ``NPZD Slab model`` has the minimal set of processes required to
# simulate a

ZeroD_NPZD_Slab_model = xs.Model({
    'Grid': ModelCore
})


# ``ZeroD NPxZx Chemostat model`` has the minimal set of processes required to
# simulate a

ZeroD_NPxZx_Chemostat_model = xs.Model({
    'Grid': ModelCore
})


# ``ZeroD NPxZx Slab model`` has the minimal set of processes required to
# simulate a

ZeroD_NPxZx_Slab_model = xs.Model({
    'Grid': ModelCore
})


# ``GridXY Chemostat model`` has a minimal set of processes for a
# 2-dimensional model, with Nutrients and Phytoplankton as components


GridXY_NP_Chemostat_model = xs.Model({
    'Grid': ModelCore,

    'Env': ModelCore,

    'N': ModelCore, 'P': ModelCore,

    'NP_uptake': ModelCore, 'P_Mortality': ModelCore,

    'FX': ModelCore, 'Mix': ModelCore,

    'GX': ModelCore,

    'MS': ModelCore
})