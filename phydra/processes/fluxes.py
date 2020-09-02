import phydra


@phydra.flux
class LinearOutputFlux:
    sv = phydra.sv(flow='output', description='state variable affected by flux')
    rate = phydra.param(description='flowing rate')

    def flux(sv, rate):
        return sv * rate


@phydra.flux
class LinearInputFlux:
    sv = phydra.sv(flow='input', description='state variable affected by flux')
    rate = phydra.param(description='flowing rate')

    def flux(sv, rate):
        return sv * rate


@phydra.flux
class ForcingLinearInputFlux:
    sv = phydra.sv(flow='input', description='state variable affected by forcing flux')
    fx = phydra.fx(description='forcing linearly affecting rate')
    rate = phydra.param(description='flowing rate')

    def flux(sv, fx, rate):
        return fx * rate
