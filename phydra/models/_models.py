import xsimlab as xs

import phydra
from phydra.components.main import Backend



# ``Bootstrap model`` has the minimal set of components required to
# simulate a

Bootstrap_model = xs.Model({
    'Grid': Backend
})


# ``NPZD Slab model`` has the minimal set of components required to
# simulate a

ZeroD_NPZD_Slab_model = xs.Model({
    'Grid': Backend
})


# ``ZeroD NPxZx Chemostat model`` has the minimal set of components required to
# simulate a

ZeroD_NPxZx_Chemostat_model = xs.Model({
    'Grid': Backend
})


# ``ZeroD NPxZx Slab model`` has the minimal set of components required to
# simulate a

ZeroD_NPxZx_Slab_model = xs.Model({
    'Grid': Backend
})


# ``GridXY Chemostat model`` has a minimal set of components for a
# 2-dimensional model, with Nutrients and Phytoplankton as components


GridXY_NP_Chemostat_model = xs.Model({
    'Grid': Backend,

    'Env': Backend,

    'N': Backend, 'P': Backend,

    'NP_uptake': Backend, 'P_Mortality': Backend,

    'FX': Backend, 'Mix': Backend,

    'GX': Backend,

    'MS': Backend
})