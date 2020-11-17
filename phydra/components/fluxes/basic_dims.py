import phydra


@phydra.comp
class LinearMortality_Dim:
    var = phydra.variable(foreign=True, dims='var', flux='death', negative=True,
                          description='variable affected by flux')
    rate = phydra.parameter(dims='var', description='linear rate of change')

    @phydra.flux
    def death(self, var, rate):
        """ """
        return var * rate

