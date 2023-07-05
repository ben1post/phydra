import xso


@xso.component
class SV:
    """represents a state variable in the model"""

    var = xso.variable(description='basic state variable')


@xso.component
class SVArray:
    """represents a state variable in the model"""

    var = xso.variable(dims='var', description='basic state variable')


@xso.component
class SVArraySize:
    """represents a state variable in the model"""

    var = xso.variable(dims='var', description='basic state variable')
    sizes = xso.parameter(dims='sizes', description='store of size array')
