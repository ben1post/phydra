import xso


@xso.component
class MonodGrowth_SizeBased:
    """Component that calculates the growth flux of phytoplankton on a singular nutrient,
    based on Michealis-menten kinetics."""

    resource = xso.variable(foreign=True, flux='uptake', negative=True)
    consumer = xso.variable(foreign=True, dims='phyto', flux='uptake', negative=False)

    halfsat = xso.parameter(dims='phyto', description='half-saturation constants')
    mu_max = xso.parameter(dims='phyto', description='maximum growth rates')

    @xso.flux(dims='phyto')
    def uptake(self, resource, consumer, halfsat, mu_max):
        return mu_max * resource / (resource + halfsat) * consumer
