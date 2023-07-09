import xso


@xso.component
class LinearForcingInput:
    """Non-dimensional linear forcing input flux."""
    var = xso.variable(foreign=True, flux='input', negative=False, description='variable affected by flux')
    forcing = xso.forcing(foreign=True, description='forcing affecting flux')
    rate = xso.parameter(description='linear rate of change')

    @xso.flux
    def input(self, var, forcing, rate):
        """ """
        return forcing * rate


@xso.component
class LinearPhytoMortality:
    """Linear Phytplankton Mortality Flux."""
    var = xso.variable(dims='phyto', foreign=True, flux='decay', negative=True, description='variable affected by flux')
    rate = xso.parameter(dims='phyto', description='linear rate of mortality')

    @xso.flux(dims='phyto')
    def decay(self, var, rate):
        """Linear decay function."""
        return var * rate

@xso.component
class QuadraticZooMortality:
    """Quadratic Zooplankton Mortality Flux."""
    var = xso.variable(foreign=True, dims='zoo', flux='decay', negative=True, description='variable affected by flux')
    rate = xso.parameter(description='quadratic rate of mortality')

    @xso.flux(dims='zoo')
    def decay(self, var, rate):
        """Flux function describing the quadratic mortality of zooplankton
        according to Banas et al. (2011).

        Utilizes the XSO Math module, available at self.m within fluxes,
        to allow for flexible implementation of math functions according
        to solver backend."""
        return var * self.m.sum(var) * rate