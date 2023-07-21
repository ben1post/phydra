import xso


@xso.component
class SV:
    """XSO component to define a state variable in the model."""
    var = xso.variable(description='basic state variable', attrs={'units': 'µM N'})
