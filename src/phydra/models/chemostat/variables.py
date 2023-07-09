import xso


@xso.component
class StateVariable:
    """XSO component to define a state variable in the model."""
    value = xso.variable(description='concentration of state variable',
                         attrs={'units': 'mmol N m-3'})

