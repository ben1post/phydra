import xso


@xso.component
class Nutrient:
    """XSO component to define a state variable in the model."""
    value = xso.variable(description='nutrient concentration',
                         attrs={'units': 'mmol N m-3'})


@xso.component
class Phytoplankton:
    """XSO component to define a state variable in the model."""
    value = xso.variable(description='phytoplankton concentration',
                         attrs={'units': 'mmol N m-3'})
