import phydra


@phydra.comp
class LinearForcingInput:
    var = phydra.variable(foreign=True, flux='input', negative=False, description='variable affected by flux')
    forcing = phydra.forcing(foreign=True, description='forcing affecting flux')
    rate = phydra.parameter(description='linear rate of change')

    @phydra.flux
    def input(self, var, forcing, rate):
        """ """
        return forcing * rate