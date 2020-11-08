import xsimlab as xs

import attr
from attr import fields_dict

from collections import OrderedDict, defaultdict
from functools import wraps
import inspect
import numpy as np

from .variable import FluxVarType
from ..components.main import FirstInit, SecondInit, ThirdInit


def _create_variables_dict(process_cls):
    """Get all phydra variables declared in a component.
    Exclude attr.Attribute objects that are not xsimlab-specific.
    """
    return OrderedDict(
        (k, v) for k, v in fields_dict(process_cls).items() if "var_type" in v.metadata
    )


def _convert_2_xsimlabvar(var, intent='in', var_dims=None, value_store=False, description_label=""):
    """ """
    var_description = var.metadata.get('description')

    if var_dims is None:
        var_dims = var.metadata.get('dims')

    if value_store:
        if not var_dims:
            var_dims = 'time'
        else:
            var_dims = (var_dims, 'time')

    return xs.variable(intent=intent, dims=var_dims, description=description_label + var_description)


def _create_xsimlab_var_dict(cls_vars):
    """ """
    xs_var_dict = defaultdict()

    for key, var in cls_vars.items():
        if var.metadata.get('var_type') is FluxVarType.VARIABLE:
            if var.metadata.get('foreign') is True:
                xs_var_dict[key] = _convert_2_xsimlabvar(var=var, var_dims=(), description_label='label reference / ')
            elif var.metadata.get('foreign') is False:
                xs_var_dict[key + '_label'] = _convert_2_xsimlabvar(var=var, var_dims=(), description_label='label / ')
                xs_var_dict[key + '_init'] = _convert_2_xsimlabvar(var=var, description_label='initial value / ')
                xs_var_dict[key + '_value'] = _convert_2_xsimlabvar(var=var, intent='out',
                                                                    value_store=True,
                                                                    description_label='output of variable value / ')
            flux_label = var.metadata.get('flux')
            if flux_label is not None:
                if flux_label + '_value' not in xs_var_dict:
                    xs_var_dict[flux_label + '_value'] = _convert_2_xsimlabvar(var=var, intent='out',
                                                                               value_store=True,
                                                                               description_label='output of flux '
                                                                                                 'value / ')

        if var.metadata.get('var_type') is FluxVarType.PARAMETER:
            xs_var_dict[key] = _convert_2_xsimlabvar(var=var)

        if var.metadata.get('var_type') is FluxVarType.FORCING:
            if var.metadata.get('foreign') is True:
                xs_var_dict[key] = _convert_2_xsimlabvar(var=var, description_label='label reference / ')
            elif var.metadata.get('foreign') is False:
                xs_var_dict[key + '_label'] = _convert_2_xsimlabvar(var=var, description_label='label / ')
                xs_var_dict[key + '_value'] = _convert_2_xsimlabvar(var=var, intent='out',
                                                                    value_store=True,
                                                                    description_label='output of forcing value / ')

    return xs_var_dict


def _create_fluxes_dict(cls, var_dict):
    """ """
    fluxes_dict = defaultdict()
    flux_var_dict = defaultdict(dict)

    for key, var in var_dict.items():
        if var.metadata.get('var_type') is FluxVarType.VARIABLE:
            _flux = var.metadata.get('flux')
            if _flux is not None:
                if _flux not in flux_var_dict:
                    fluxes_dict[_flux] = getattr(cls, _flux)

                flux_var_dict[key] = {'flux': _flux,
                                      'negative': var.metadata.get('negative'),
                                      'foreign': var.metadata.get('foreign')}

    flux_var_dict['_fluxes'] = fluxes_dict

    return flux_var_dict


def _create_forcing_dict(cls, var_dict):
    """ """
    forcings_dict = defaultdict()

    for key, var in var_dict.items():
        if var.metadata.get('var_type') is FluxVarType.FORCING:
            _file_input_func = var.metadata.get('file_input_func')

            if _file_input_func is not None:
                forcings_dict[key] = getattr(cls, _file_input_func)

    return forcings_dict


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
    process_label = cls.label
    for key, var in vars_dict.items():
        if var.metadata.get('var_type') is FluxVarType.VARIABLE:
            if var.metadata.get('foreign') is False:
                _init = getattr(cls, key + '_init')
                _label = getattr(cls, key + '_label')
                setattr(cls, key + '_value', cls.m.add_variable(label=_label, initial_value=_init))

        elif var.metadata.get('var_type') is FluxVarType.PARAMETER:
            if var.metadata.get('foreign') is False:
                _par_value = getattr(cls, key)
                cls.m.add_parameter(label=process_label + '_' + key, value=_par_value)


def _initialize_forcings(cls, forcing_dict):
    """ """
    for var, forc_input_func in forcing_dict.items():
        forc_label = getattr(cls, var + '_label')

        argspec = inspect.getfullargspec(forc_input_func)
        input_args = defaultdict()

        for arg in argspec.args:
            input_args[arg] = getattr(cls, arg)

        forc_func = forc_input_func(**input_args)

        setattr(cls, var + '_value',
                cls.m.add_forcing(label=forc_label, forcing_func=forc_func))


def _initialize_fluxes(cls, process_dict):
    """ """
    for flx_label, flux in process_dict['_fluxes'].items():
        setattr(cls, flux.__name__ + '_value',
                cls.m.register_flux(process_label=cls.label, flux=cls.flux_decorator(flux)))

    for var, flx_dict in process_dict.items():
        if var != '_fluxes':
            if flx_dict['foreign'] is True:
                var_label = getattr(cls, var)
            elif flx_dict['foreign'] is False:
                var_label = getattr(cls, var + '_label')

            cls.m.add_flux(process_label=cls.label,
                           var_label=var_label,
                           flux_label=flx_dict['flux'],
                           negative=flx_dict['negative'])


def _create_flux_inputargs_dict(cls, vars_dict):
    """ """
    input_arg_dict = defaultdict(list)

    for key, var in vars_dict.items():
        if var.metadata.get('var_type') is FluxVarType.VARIABLE:
            if var.metadata.get('foreign') is False:
                var_label = getattr(cls, key + '_label')
            elif var.metadata.get('foreign') is True:
                var_label = getattr(cls, key)
            input_arg_dict['vars'].append({'var': key, 'label': var_label})
        elif var.metadata.get('var_type') is FluxVarType.PARAMETER:
            # TODO: so it doesn't work with foreign parameters here yet!
            input_arg_dict['pars'].append({'var': key, 'label': cls.label + '_' + key})
        elif var.metadata.get('var_type') is FluxVarType.FORCING:
            if var.metadata.get('foreign') is False:
                forc_label = getattr(cls, key + '_label')
            elif var.metadata.get('foreign') is True:
                forc_label = getattr(cls, key)
            input_arg_dict['forcs'].append({'var': key, 'label': forc_label})

    return input_arg_dict


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
    vars_dict = _create_variables_dict(attr_cls)
    fluxes_dict = _create_fluxes_dict(cls, vars_dict)
    forcing_dict = _create_forcing_dict(cls, vars_dict)

    new_cls = _create_new_cls(cls, _create_xsimlab_var_dict(vars_dict), init_stage)

    def flux_decorator(self, func):
        """ flux function decorator to unpack arguments """

        @wraps(func)
        def unpack_args(**kwargs):
            state = kwargs.get('state')
            parameters = kwargs.get('parameters')
            forcings = kwargs.get('forcings')

            input_args = {}

            for v_dict in self.flux_input_args['vars']:
                input_args[v_dict['var']] = state[v_dict['label']]
            for p_dict in self.flux_input_args['pars']:
                input_args[p_dict['var']] = parameters[p_dict['label']]
            for f_dict in self.flux_input_args['forcs']:
                input_args[f_dict['var']] = forcings[f_dict['label']]

            # added option to force vectorisation for model arrays/lists containing objects (i.e. gekko components):
            try:
                vectorized = kwargs.pop('vectorized')
            except:
                vectorized = False

            if vectorized:
                return np.vectorize(func,  excluded=['forcing'])(self, **input_args)
            else:
                return func(self, **input_args)

        return unpack_args

    def initialize(self):
        """ """
        super(new_cls, self).initialize()
        print(f"Initializing component {self.label}")

        _initialize_process_vars(self, vars_dict)

        self.flux_input_args = _create_flux_inputargs_dict(self, vars_dict)

        _initialize_forcings(self, forcing_dict)

        _initialize_fluxes(self, fluxes_dict)

    setattr(new_cls, 'flux_decorator', flux_decorator)
    setattr(new_cls, 'initialize', initialize)

    process_cls = xs.process(new_cls)

    # TODO: clean this up below:
    try:
        process_cls.flux = getattr(cls, 'flux')
    except AttributeError:
        # print("process cls contains no flux func")
        pass

    return process_cls
