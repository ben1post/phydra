import xso


@xso.component
class Nutrient:
    """XSO component to define a state variable in the model."""
    value = xso.variable(description='nutrient concentration',
                         attrs={'units': 'mmol N m-3'})


@xso.component
class PhytoSizeSpectrum:
    """XSO component to define an array of state variables in the model.
    Additionally, there is a parameter defined, that stores an array of cell sizes."""

    biomass = xso.variable(dims='phyto', description='phytoplankton biomass',
                           attrs={'units': 'mmol N m-3', 'long_name': 'Phytoplankton biomass concentration',
                                  'standard_name': 'Phytoplankton'})
    phyto = xso.index(dims='phyto', description='size spectrum of phytoplankton',
                     attrs={'units': 'µm ESD', 'long_name': 'Phytoplankton size classes',
                                   'standard_name': 'P size classes'})


@xso.component
class ZooSizeSpectrum:
    """XSO component to define an array of state variables in the model.
    Additionally, there is a parameter defined, that stores an array of cell sizes."""

    biomass = xso.variable(dims='zoo', description='zooplankton biomass',
                           attrs={'units': 'mmol N m-3', 'long_name': 'Zooplankton biomass concentration',
                                  'standard_name': 'Zooplankton'})
    zoo = xso.index(dims='zoo', description='size spectrum of zooplankton',
                          attrs={'units': 'µm ESD', 'long_name': 'Zooplankton size classes',
                                 'standard_name': 'Z size classes'})
