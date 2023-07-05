import xso


@xso.component
class SV:
    """XSO component to define a state variable in the model."""

    var = xso.variable(dims=[(), 'var'], description='basic state variable')


@xso.component
class SVArray:
    """XS0 component to define an array of state variables in the model."""

    var = xso.variable(dims='var', description='basic state variable')


@xso.component
class SVArraySize:
    """XSO component to define an array of state variables in the model.
    Additionally, there is a parameter defined, that stores an array of cell sizes."""

    var = xso.variable(dims='var', description='basic state variable')
    sizes = xso.parameter(dims='sizes', description='store of size array')
