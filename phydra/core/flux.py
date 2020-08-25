from .parts import Parameter

from ..processes.main import ThirdInit

import xsimlab as xs

import attr
from attr._make import _CountingAttr

from collections import defaultdict

from enum import Enum

class FluxVarType(Enum):
    STATEVARIABLE = "statevariable"
    PARAMETER = "parameter"
    FORCING = "forcing"


class FluxVarFlow(Enum):
    INPUT = "input"
    OUTPUT = "output"


class FluxVarIntent(Enum):
    IN = "in"
    OUT = "out"


def flux_sv(
    flow='input',
    intent='in',
    default=attr.NOTHING,
    validator=None,
    converter=None,
    description='',
    attrs=None,
):
    metadata = {
        "var_type": FluxVarType.STATEVARIABLE,
        "intent": FluxVarIntent(intent),
        "flow": FluxVarFlow(flow),
        "attrs": attrs or {},
        "description": description,
    }

    if FluxVarIntent(intent) == FluxVarIntent.OUT:
        _init = False
        _repr = False
    else:
        _init = True
        _repr = True

    return attr.attrib(
        metadata=metadata,
        default=default,
        validator=validator,
        converter=converter,
        init=_init,
        repr=_repr,
        kw_only=True,
    )


def flux_param(
    intent='in',
    default=attr.NOTHING,
    validator=None,
    converter=None,
    description='',
    attrs=None,
):
    metadata = {
        "var_type": FluxVarType.PARAMETER,
        "intent": FluxVarIntent(intent),
        "flow": None,
        "attrs": attrs or {},
        "description": description,
    }

    return attr.attrib(
        metadata=metadata,
        default=default,
        validator=validator,
        converter=converter,
        init=True,
        repr=True,
        kw_only=True,
    )


def _convert_2_xsimlabvar(var):
    var_description = var.metadata.get('description')
    return xs.variable(intent='in', description=var_description)


def flux_decorator(cls):
    """ flux decorator
    that converts simplified flux class into fully functional
    xarray simlab process
    """
    new_cls_dict = defaultdict()
    flux_dict = defaultdict(list)

    # convert state variables, forcing and parameters to xs.variables in new process
    for var_name, var in cls.__dict__.items():
        if isinstance(var, _CountingAttr):
            var_type = var.metadata.get('var_type')
            if var_type is not None:
                new_cls_dict[var_name] = _convert_2_xsimlabvar(var)
                flux_dict[var_type].append({'var_name': var_name,
                                            'metadata': var.metadata})

    new_cls = type(cls.__name__, (ThirdInit,), new_cls_dict)

    # convert flux function into functional xarray-simlab flux
    def flux(self, **kwargs):
        """linear loss flux of state variable"""
        state = kwargs.get('state')
        parameters = kwargs.get('parameters')
        forcings = kwargs.get('forcings')

        # print(state, '\n',parameters)

        input_args = {}
        for name in self.states:
            input_args[name['name']] = state[name['value']]
        for name in self.pars:
            input_args[name] = parameters[self.label + '_' + name]

        return cls.flux(**input_args)

    def negative_flux(self, **kwargs):
        """simple wrapper function to return negative flux to output flow"""
        out = flux(self, **kwargs)
        return - out

    setattr(new_cls, 'flux', flux)
    setattr(new_cls, 'negative_flux', negative_flux)

    def initialize_flux(self):
        self.label = self.__xsimlab_name__
        self.group = 3  # handles initialisation stages
        print(f"initializing flux {self.label}")

        self.states = []
        self.pars = []
        self.forcings = []
        for key, varlist in flux_dict.items():
            for var in varlist:
                var_value = getattr(self, var['var_name'])
                # parameters var_value is a float, statevariable var_value is string!
                if key is FluxVarType.PARAMETER:
                    self.pars.append(var['var_name'])
                    self.m.Parameters[self.label + '_' + var['var_name']] = \
                        Parameter(name=self.label + '_' + var['var_name'], value=var_value)
                elif key is FluxVarType.STATEVARIABLE:
                    self.states.append({'value': var_value, 'name': var['var_name']})
                    if var['metadata']['flow'] is FluxVarFlow.OUTPUT:
                        self.m.Fluxes[var_value].append(self.negative_flux)
                    elif var['metadata']['flow'] is FluxVarFlow.INPUT:
                        self.m.Fluxes[var_value].append(self.flux)
        # TODO: add handling of forcing!

    setattr(new_cls, 'initialize', initialize_flux)

    return xs.process(new_cls)