import xso


@xso.component
class LinearForcingInput:
    var = xso.variable(foreign=True, flux='input', negative=False, description='variable affected by flux')
    forcing = xso.forcing(foreign=True, description='forcing affecting flux')
    rate = xso.parameter(description='linear rate of change')

    @xso.flux
    def input(self, var, forcing, rate):
        """ """
        return forcing * rate