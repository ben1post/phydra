import xsimlab as xs

import attr
from attr import fields_dict

from .variable import FluxVarFlow, FluxVarType, FluxVarIntent
from ..components.main import FirstInit, SecondInit, ThirdInit

from collections import OrderedDict, defaultdict

import numpy as np


# TODO: Simplify!
#   - spell out the basic components of decorated flux class:
#       1. variables (that are referenced via string labels)
#       2. parameters (that are usually initialized with the flux) +
#           additionally there should be the possibility to reference pars (e.g. cell size, etc.)
#       3. forcings (that are referenced via string labels)
#       4. the actual flux function, that takes all attributes as arguments
#   - this should work for the simplest case, but also for ARRAYS & MultiFluxes!
#   - write flux base class that automatically stores value for its flux function (model diagnostic)

def variables_dict(process_cls):
    """Get all phydra variables declared in a component.
    Exclude attr.Attribute objects that are not xsimlab-specific.
    """
    return OrderedDict(
        (k, v) for k, v in fields_dict(process_cls).items() if "var_type" in v.metadata
    )


def _convert_2_xsimlabvar(var, label=""):
    """ """
    var_description = var.metadata.get('description')
    var_dims = var.metadata.get('dims')

    return xs.variable(intent='in', dims=var_dims, description=label + var_description)


def _create_xsimlab_var_dict(cls_vars):
    """ """
    xs_var_dict = defaultdict()

    for key, var in cls_vars.items():
        if var.metadata.get('var_type') is FluxVarType.VARIABLE:
            if var.metadata.get('foreign') is True:
                xs_var_dict[key] = _convert_2_xsimlabvar(var)
            elif var.metadata.get('foreign') is False:
                xs_var_dict[key + '_label'] = _convert_2_xsimlabvar(var, 'label /')
                xs_var_dict[key + '_init'] = _convert_2_xsimlabvar(var, 'initial value /')
                xs_var_dict[key + '_value'] = xs.variable(intent='out', dims='time',
                                                          description='output of variable value')

        if var.metadata.get('var_type') is FluxVarType.PARAMETER:
            xs_var_dict[key] = _convert_2_xsimlabvar(var)

    return xs_var_dict


def _create_process_dict(cls, var_dict):
    process_dict = defaultdict(dict)
    print("creating process dict")
    for key, var in var_dict.items():
        if var.metadata.get('var_type') is FluxVarType.VARIABLE:
            _flux = var.metadata.get('flux')
            if _flux is not None:
                process_dict[key] = {'flux': getattr(cls, _flux),
                                     'negative': var.metadata.get('negative'),
                                     'foreign': var.metadata.get('foreign')}

        print("\n")
        print(key, var)

    print("DONE//////")
    print(process_dict)
    return process_dict


def _create_new_cls(cls, cls_dict, init_stage):
    """ """
    if init_stage == 1:
        new_cls = type(cls.__name__, (FirstInit,), cls_dict)
    elif init_stage == 2:
        new_cls = type(cls.__name__, (SecondInit,), cls_dict)
    elif init_stage == 3:
        new_cls = type(cls.__name__, (ThirdInit,), cls_dict)
    else:
        raise Exception("Wrong init_stage supplied, needs to be 1, 2 or 3")
    return new_cls


def _initialize_process_vars(cls, vars_dict):
    """ """
    print("registering process vars")
    for key, var in vars_dict.items():
        if var.metadata.get('var_type') is FluxVarType.VARIABLE:
            if var.metadata.get('foreign') is False:
                _init = getattr(cls, key + '_init')
                _label = getattr(cls, key + '_label')
                setattr(cls, key + '_value', cls.m.add_variable(label=_label, initial_value=_init))


def _initialize_fluxes(cls, process_dict):
    """ """
    print("XYXYXY registering fluxes XYXYXYXYX")

    print(process_dict)

    for var, fluxdict in process_dict.items():
        if fluxdict['foreign'] is True:
            var_value = getattr(cls, var)
        elif fluxdict['foreign'] is False:
            var_value = getattr(cls, var+'_label')

        print(var, var_value, fluxdict)
        cls.m.Model.fluxes[var_value].append(fluxdict['flux'])

        # cls.m.add_flux()

        # TODO:
        #   what info do i need here?
        #       - the flux function + the variable it is assigned to
        #   what do i need to do here?
        #       - add the flux to the model backend
        #       - make sure the flux func is properly wrapped (with dict of pars, vars, etc) */use kwargs/to pass unused
        #       - ...


def parametrized(dec):
    """ Simple decorator that allows adding parameters to another decorator
    Source: https://stackoverflow.com/a/26151604/11826333
    """

    def layer(*args, **kwargs):
        def repl(f):
            return dec(f, *args, **kwargs)

        return repl

    return layer


@parametrized
def comp(cls, init_stage):
    """ component decorator
    that converts simple base class using phydra.backend.variables into fully functional xarray simlab process
    """

    attr_cls = attr.attrs(cls, repr=False)
    vars_dict = variables_dict(attr_cls)
    process_dict = _create_process_dict(cls, vars_dict)

    new_cls_dict = _create_xsimlab_var_dict(vars_dict)
    new_cls = _create_new_cls(cls, new_cls_dict, init_stage)

    # TODO:
    #  1. Step = clean unpacking of all defined variables + flux function
    #   NOW NEED TO DO:
    #       - assign flux function from "flux" attribute
    #       - allow passing through flux function (add to model)
    #          - "negative" functionality
    #       - for flux, add value of derivative

    # TODO:
    #  2. Step = initialize new class to modify and add new functions to --> xs.process

    # TODO:
    #  3. Step = write new initialize function for xs.process

    def initialize(self):
        super(new_cls, self).initialize()
        print(f"running initialize func of {self.label}")

        # add non-foreign variables to model:
        _initialize_process_vars(self, vars_dict)
        _initialize_fluxes(self, process_dict)

        # per flux, give input args as dict
        # or perhaps, they can all have the same inputs?
        # but that shouldn't be necessary...

    #
    #
    #     self.vars = []
    #     self.pars = []
    #     self.forcs = []
    #
    #     for var, meta in flux_dict.items():
    #         var_value = getattr(self, var)
    #
    #         # TODO: here store everything needed to initialise flux in backend as well
    #         #   so: for flux func I need= name + value
    #         #   and: for adding flux in backend I need= label, FLUX (+state vars (+routing)) ,(new) pars
    #
    #         if meta['var_type'] is FluxVarType.PARAMETER:
    #             self.pars.append(var_value)
    #         elif meta['var_type'] is FluxVarType.VARIABLE:
    #             self.vars.append({'value': var_value, 'name': var})
    #         elif meta['var_type'] is FluxVarType.FORCING:
    #             self.forcs.append({'value': var_value, 'name': var})
    #
    #         # if var_type is FluxVarType.PARAMETER:
    #         #     self.Model.parameters[label + '_' + var_name] = var_value
    #         # elif var_type is FluxVarType.VARIABLE:
    #         #     if var_flow is FluxVarFlow.OUTPUT:
    #         #         self.Model.fluxes[var_value].append(flux.negative_flux)
    #         #     elif var_flow is FluxVarFlow.INPUT:
    #         #         self.Model.fluxes[var_value].append(flux.flux)
    #
    #     self.m.add_flux(self.label, flux_dict)

    # TODO:
    #  4. Step = provide wrapper that simplifies adding flux function(s)
    #   idea: perhaps allow manually providing function name, instead of having it fixed to "flux"?

    # convert flux function into functional xarray-simlab flux
    def flux(self, **kwargs):
        """ """
        state = kwargs.get('state')
        parameters = kwargs.get('parameters')
        forcings = kwargs.get('forcings')

        input_args = {}

        for name, value in self.vars:
            input_args[name] = state[value]
        for name, value in self.forcs.items():
            input_args[name] = forcings[value]
        for name in self.pars:
            input_args[name] = parameters[self.label + '_' + name]

        return cls.comp(**input_args)

    def negative_flux(self, **kwargs):
        """simple wrapper function to return negative flux to output flow"""
        out = flux(self, **kwargs)
        return - out

    # setattr(new_cls, 'flux', flux)
    # setattr(new_cls, 'negative_flux', negative_flux)
    setattr(new_cls, 'initialize', initialize)

    return xs.process(new_cls)


####################################################################################################################
####################################################################################################################
####################################################################################################################
####################################################################################################################
####################################################################################################################
####################################################################################################################
####################################################################################################################

def OLD_flux(cls):
    """ flux decorator
    that converts simplified flux class into fully functional
    xarray simlab process
    """

    new_cls_dict = defaultdict()
    flux_dict = defaultdict()
    cls_name = cls.__name__

    # convert state variables, forcing and parameters to xs.variables in new process
    for var_name, var in cls.__dict__.items():
        if isinstance(var, _CountingAttr):
            new_cls_dict[var_name] = _convert_2_xsimlabvar(var)

            flux_dict[var_name] = var.metadata

    new_cls = type(cls_name, (ThirdInit,), new_cls_dict)

    # convert flux function into functional xarray-simlab flux
    def flux(self, **kwargs):
        """ """
        state = kwargs.get('state')
        parameters = kwargs.get('parameters')
        forcings = kwargs.get('forcings')

        input_args = {}

        for name, value in self.vars:
            input_args[name] = state[value]
        for name, value in self.forcs.items():
            input_args[name] = forcings[value]
        for name in self.pars:
            input_args[name] = parameters[self.label + '_' + name]

        return cls.comp(**input_args)

    def negative_flux(self, **kwargs):
        """simple wrapper function to return negative flux to output flow"""
        out = flux(self, **kwargs)
        return - out

    def initialize_flux(self):
        self.label = self.__xsimlab_name__
        self.group = 3  # handles initialisation stages
        print(f"initializing flux {self.label}")

        print(flux_dict)

        self.vars = []
        self.pars = []
        self.forcs = []

        for var, meta in flux_dict.items():
            var_value = getattr(self, var)

            # TODO: here store everything needed to initialise flux in backend as well
            #   so: for flux func I need= name + value
            #   and: for adding flux in backend I need= label, FLUX (+state vars (+routing)) ,(new) pars

            if meta['var_type'] is FluxVarType.PARAMETER:
                self.pars.append(var_value)
            elif meta['var_type'] is FluxVarType.VARIABLE:
                self.vars.append({'value': var_value, 'name': var})
            elif meta['var_type'] is FluxVarType.FORCING:
                self.forcs.append({'value': var_value, 'name': var})

            if var_type is FluxVarType.PARAMETER:
                self.Model.parameters[label + '_' + var_name] = var_value
            elif var_type is FluxVarType.VARIABLE:
                if var_flow is FluxVarFlow.OUTPUT:
                    self.Model.fluxes[var_value].append(flux.negative_flux)
                elif var_flow is FluxVarFlow.INPUT:
                    self.Model.fluxes[var_value].append(flux.flux)

        self.m.add_flux(self.label, flux_dict)

        # IDEA: so here, instead of converting stuff around all the time, only keep one flux_dict,
        # that stores everything necessary, and is passed around

    setattr(new_cls, 'flux', flux)
    setattr(new_cls, 'negative_flux', negative_flux)
    setattr(new_cls, 'initialize', initialize_flux)

    return xs.process(new_cls)


def Old_flux(cls):
    """ flux decorator
    that converts simplified flux class into fully functional
    xarray simlab process
    """
    new_cls_dict = defaultdict()
    flux_dict = defaultdict(list)

    # convert state variables, forcing and parameters to xs.variables in new process
    for var_name, var in cls.__dict__.items():
        if isinstance(var, _CountingAttr):

            # var_dims = var.metadata.get('dims')

            var_type = var.metadata.get('var_type')
            if var_type is not None:
                new_cls_dict[var_name] = _convert_2_xsimlabvar(var)
                # HERE, all of them are basic xarray simlab variables input!
                # no linkages, foreign, or groups.. hm.. maybe another way better?

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
        print(self.vars, '\n', self.forcs, '\n', self.pars)
        for name, value in self.vars:
            print(name, value)
            input_args[name] = state[value]
        for name, value in self.forcs.items():
            input_args[name] = forcings[value]
        for name in self.pars:
            input_args[name] = parameters[self.label + '_' + name]

        return cls.comp(**input_args)

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

        self.vars = []
        self.pars = []
        self.forcs = []
        for key, varlist in flux_dict.items():
            for var in varlist:
                var_value = getattr(self, var['var_name'])
                # parameters var_value is a float, statevariable var_value is string!
                if key is FluxVarType.PARAMETER:
                    self.pars.append(var['var_name'])
                    self.m.Model.parameters[self.label + '_' + var['var_name']] = var_value
                elif key is FluxVarType.VARIABLE:
                    self.vars.append({'value': var_value, 'name': var['var_name']})
                    if var['metadata']['flow'] is FluxVarFlow.OUTPUT:
                        self.m.Model.fluxes[var_value].append(self.negative_flux)
                    elif var['metadata']['flow'] is FluxVarFlow.INPUT:
                        self.m.Model.fluxes[var_value].append(self.comp)
                elif key is FluxVarType.FORCING:
                    self.forcs.append({'value': var_value, 'name': var['var_name']})

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

        # print("self.vars", self.vars)
        # print("self.forcings", self.forcings)
        # print("self.parameters", self.parameters)

        input_args = {}

        for var in self.vars:
            if isinstance(var['value'], np.ndarray):
                input_args[var['keyword']] = np.array([state[value] for value in var['value']])
            else:
                input_args[var['keyword']] = state[var['value']]
        for var in self.forcings:
            input_args[var['keyword']] = forcings[var['value']]
        for var in self.parameters:
            input_args[var] = parameters[self.label + '_' + var]

        return cls.comp(**input_args)

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

        self.vars = []
        self.pars = []
        self.forcings = []

        self.multiflux_routing = defaultdict()

        for key, varlist in flux_dict.items():
            for var in varlist:
                var_value = getattr(self, var['var_name'])

                if key is FluxVarType.PARAMETER:
                    self.pars.append(var['var_name'])
                    self.m.Parameters[self.label + '_' + var['var_name']] = \
                        Parameter(name=self.label + '_' + var['var_name'], value=var_value)

                elif key is FluxVarType.VARIABLE:
                    self.vars.append({'value': var_value, 'keyword': var['var_name']})
                    # TODO: add this as a list for different keys in defaultdict

                    if var['metadata']['partial_out'] is not None:
                        partial_out_func = getattr(cls, var['metadata']['partial_out'])
                    else:
                        partial_out_func = None

                    if var['metadata']['flow'] is FluxVarFlow.OUTPUT:
                        self.multiflux_routing['OUTPUT'] = {'labels': [val for val in var_value],
                                                            'partial_out': partial_out_func}
                    elif var['metadata']['flow'] is FluxVarFlow.INPUT:
                        self.multiflux_routing['INPUT'] = {'labels': [val for val in var_value],
                                                           'partial_out': partial_out_func}

                elif key is FluxVarType.FORCING:
                    self.forcings.append({'value': var_value, 'keyword': var['var_name']})

        print("Routing Setup: ", self.multiflux_routing)
        # add all necessary info to do multiflux routing in backend.model:
        self.m.MultiFluxes[self.label] = {'flux': self.comp,
                                          'routing': self.multiflux_routing}

        # TODO: add handling of forcing!

    setattr(new_cls, 'initialize', initialize_flux)

    return xs.process(new_cls)
