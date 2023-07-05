import xso


@xso.component
class ConstantForcing:
    """XSO component to define a constant forcing in the model."""

    forcing = xso.forcing(foreign=True, description='forcing affecting flux')
    rate = xso.parameter(description='constant rate of change')

    @xso.flux
    def input(self, forcing, rate):
        """Flux function for constant input flux

        Parameters
        ----------
        forcing : xso.forcing
            forcing affecting flux
        rate : xso.parameter
            constant rate of change

        Returns
        -------
        xso.flux
            constant input flux
        """
        return forcing * rate