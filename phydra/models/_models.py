import xsimlab as xs

from ..processes.grid import GridXY
#from ..processes


# ``slab NPZD model`` has the minimal set of processes required to
# simulate a

NPZDslab_model = xs.Model({
    'grid': GridXY
})
