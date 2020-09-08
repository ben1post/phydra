from .parts import Parameter

from ..processes.main import ThirdInit

import xsimlab as xs

import attr
from attr._make import _CountingAttr

from collections import defaultdict

from enum import Enum

import numpy as np

import functools


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


def sv(
        flow='input',
        intent='in',
        dims=[()],
        sub_label=None,
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
        "dims": dims,
        "sub_label": sub_label,
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


def fx(
        intent='in',
        dims=[()],
        sub_label=None,
        default=attr.NOTHING,
        validator=None,
        converter=None,
        description='',
        attrs=None,
):
    metadata = {
        "var_type": FluxVarType.FORCING,
        "intent": FluxVarIntent(intent),
        "dims": dims,
        "sub_label": sub_label,
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


def param(
        intent='in',
        dims=[()],
        sub_label=None,
        default=attr.NOTHING,
        validator=None,
        converter=None,
        description='',
        attrs=None,
):
    metadata = {
        "var_type": FluxVarType.PARAMETER,
        "intent": FluxVarIntent(intent),
        "dims": dims,
        "sub_label": sub_label,
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
    var_dims = var.metadata.get('dims')
    return xs.variable(intent='in', dims=var_dims, description=var_description)


def flux(cls):
    """ flux decorator
    that converts simplified flux class into fully functional
    xarray simlab process
    """
    new_cls_dict = defaultdict()
    flux_dict = defaultdict(list)

    # convert state variables, forcing and parameters to xs.variables in new process
    for var_name, var in cls.__dict__.items():
        if isinstance(var, _CountingAttr):

            var_dims = var.metadata.get('dims')

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
        for name in self.forcings:
            input_args[name['name']] = forcings[name['value']]
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
                elif key is FluxVarType.FORCING:
                    self.forcings.append({'value': var_value, 'name': var['var_name']})

    setattr(new_cls, 'initialize', initialize_flux)

    return xs.process(new_cls)


def multiflux(cls):
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
            var_dims = var.metadata.get('dims')

            if var_type is not None:
                new_cls_dict[var_name] = _convert_2_xsimlabvar(var)
                flux_dict[var_type].append({'var_name': var_name,
                                            'metadata': var.metadata,
                                            'dims': var_dims})

    new_cls = type(cls.__name__, (ThirdInit,), new_cls_dict)

    # convert flux function into functional xarray-simlab flux

    # so the flux below should actually return the bespoke flux, but how?

    def flux(self, **kwargs):
        """linear loss flux of state variable"""
        state = kwargs.get('state')
        parameters = kwargs.get('parameters')
        forcings = kwargs.get('forcings')

        # print("self.states", self.states)
        # print("self.forcings", self.forcings)
        # print("self.pars", self.pars)

        input_args = {}

        for var in self.states:
            if isinstance(var['value'], np.ndarray):
                input_args[var['keyword']] = np.array([state[value] for value in var['value']])
            else:
                input_args[var['keyword']] = state[var['value']]
        for var in self.forcings:
            input_args[var['keyword']] = forcings[var['value']]
        for var in self.pars:
            input_args[var] = parameters[self.label + '_' + var]

        return cls.flux(**input_args)

    # TODO: IDEA: maybe i can simply use a wrapper like negative flux
    #   BUT, still the question remains , how to use labels in order to make this completely safe?

    def partial_flux(self, **kwargs):
        out = flux(self, **kwargs)
        return out

    def negative_flux(self, **kwargs):
        """simple wrapper function to return negative flux to output flow"""
        out = flux(self, **kwargs)
        return -out

    setattr(new_cls, 'flux', flux)
    setattr(new_cls, 'negative_flux', negative_flux)
    setattr(new_cls, 'partial_flux', partial_flux)

    def initialize_flux(self):
        self.label = self.__xsimlab_name__
        self.group = 3  # handles initialisation stages
        print(f"initializing flux {self.label}")
        print(" ")
        print("flux_dict: ")

        print(flux_dict)
        print(" ")

        self.states = []
        self.pars = []
        self.forcings = []

        self.multiflux_routing = {}

        for key, varlist in flux_dict.items():
            for var in varlist:
                var_value = getattr(self, var['var_name'])

                if key is FluxVarType.PARAMETER:
                    self.pars.append(var['var_name'])
                    self.m.Parameters[self.label + '_' + var['var_name']] = \
                        Parameter(name=self.label + '_' + var['var_name'], value=var_value)

                elif key is FluxVarType.STATEVARIABLE:
                    self.states.append({'value': var_value, 'keyword': var['var_name']})
                    if var['metadata']['flow'] is FluxVarFlow.OUTPUT:
                        self.multiflux_routing.update({val: 'OUTPUT' for val in var_value})
                    elif var['metadata']['flow'] is FluxVarFlow.INPUT:
                        self.multiflux_routing.update({val: 'INPUT' for val in var_value})

                elif key is FluxVarType.FORCING:
                    self.forcings.append({'value': var_value, 'keyword': var['var_name']})

        # add all necessary info to do multiflux routing in backend.model:
        self.m.MultiFluxes[self.label] = {'flux': self.negative_flux,
                                          'routing': self.multiflux_routing}

        # TODO: add handling of forcing!

    setattr(new_cls, 'initialize', initialize_flux)

    return xs.process(new_cls)
