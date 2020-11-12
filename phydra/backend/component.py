import xsimlab as xs

import attr
from attr import fields_dict

from collections import OrderedDict, defaultdict
from functools import wraps
import inspect
import numpy as np

from .variable import PhydraVarType
from ..components.main import FirstInit, SecondInit, ThirdInit, FourthInit


def _create_variables_dict(process_cls):
    """Get all phydra variables declared in a component.
    Exclude attr.Attribute objects that are not xsimlab-specific.
    """
    return OrderedDict(
        (k, v) for k, v in fields_dict(process_cls).items() if "var_type" in v.metadata
    )


def _convert_2_xsimlabvar(var, intent='in',
                          var_dims=None, value_store=False, groups=None,
                          description_label=''):
    """ """

    var_description = var.metadata.get('description')
    if var_description:
        description_label = description_label + var_description

    if var_dims is None:
        var_dims = var.metadata.get('dims')

    if value_store:
        if not var_dims:
            var_dims = 'time'
        else:
            var_dims = (var_dims, 'time')

    var_attrs = var.metadata.get('attrs')

    return xs.variable(intent=intent, dims=var_dims, groups=groups, description=description_label, attrs=var_attrs)


def _make_phydra_variable(label, variable):
    """ """
    xs_var_dict = defaultdict()
    if variable.metadata.get('foreign') is True:
        xs_var_dict[label] = _convert_2_xsimlabvar(var=variable, var_dims=(),
                                                   description_label='label reference / ')
    elif variable.metadata.get('foreign') is False:
        xs_var_dict[label + '_label'] = _convert_2_xsimlabvar(var=variable, var_dims=(),
                                                              description_label='label / ')
        xs_var_dict[label + '_init'] = _convert_2_xsimlabvar(var=variable, description_label='initial value / ')
        xs_var_dict[label + '_value'] = _convert_2_xsimlabvar(var=variable, intent='out',
                                                              value_store=True,
                                                              description_label='output of variable value / ')
    return xs_var_dict


def _make_phydra_parameter(label, variable):
    """ """
    xs_var_dict = defaultdict()
    xs_var_dict[label] = _convert_2_xsimlabvar(var=variable)
    return xs_var_dict


def _make_phydra_forcing(label, variable):
    """ """
    xs_var_dict = defaultdict()
    if variable.metadata.get('foreign') is True:
        xs_var_dict[label] = _convert_2_xsimlabvar(var=variable, description_label='label reference / ')
    elif variable.metadata.get('foreign') is False:
        xs_var_dict[label + '_label'] = _convert_2_xsimlabvar(var=variable, description_label='label / ')
        xs_var_dict[label + '_value'] = _convert_2_xsimlabvar(var=variable, intent='out',
                                                              value_store=True,
                                                              description_label='output of forcing value / ')
    return xs_var_dict


def _make_phydra_flux(label, variable):
    """ """
    xs_var_dict = defaultdict()
    xs_var_dict[label + '_value'] = _convert_2_xsimlabvar(var=variable, intent='out',
                                                          value_store=True,
                                                          description_label='output of flux value / ')
    group = variable.metadata.get('group')
    group_to_arg = variable.metadata.get('group_to_arg')

    if group and group_to_arg:
        raise Exception("A flux can be either added to group or call a group to an attribute, not both.")

    if group:
        xs_var_dict[label + '_label'] = _convert_2_xsimlabvar(var=variable, intent='out', groups=group,
                                                              description_label='label reference with group / ')
    if group_to_arg:
        xs_var_dict[group_to_arg] = xs.group(group_to_arg)

    return xs_var_dict


_make_xsimlab_vars = {
    PhydraVarType.VARIABLE: _make_phydra_variable,
    PhydraVarType.FORCING: _make_phydra_forcing,
    PhydraVarType.PARAMETER: _make_phydra_parameter,
    PhydraVarType.FLUX: _make_phydra_flux,
}


def _create_xsimlab_var_dict(cls_vars):
    """ """
    xs_var_dict = defaultdict()

    for key, var in cls_vars.items():
        var_type = var.metadata.get('var_type')
        var_dict = _make_xsimlab_vars[var_type](key, var)
        for xs_key, xs_var in var_dict.items():
            xs_var_dict[xs_key] = xs_var

    return xs_var_dict


def _create_forcing_dict(cls, var_dict):
    """ """
    forcings_dict = defaultdict()

    for key, var in var_dict.items():
        if var.metadata.get('var_type') is PhydraVarType.FORCING:
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
    elif init_stage == 4:
        new_cls = type(cls.__name__, (FourthInit,), cls_dict)
    else:
        raise Exception("Wrong init_stage supplied, needs to be 1, 2, 3 or 4")
    return new_cls


def _initialize_process_vars(cls, vars_dict):
    """ """
    process_label = cls.label
    for key, var in vars_dict.items():
        var_type = var.metadata.get('var_type')
        if var_type is PhydraVarType.VARIABLE:
            foreign = var.metadata.get('foreign')
            if foreign is True:
                _label = getattr(cls, key)
            elif foreign is False:
                _init = getattr(cls, key + '_init')
                _label = getattr(cls, key + '_label')
                setattr(cls, key + '_value', cls.m.add_variable(label=_label, initial_value=_init))
            flux_label = var.metadata.get('flux')
            if flux_label:
                cls.m.add_flux(process_label=cls.label, var_label=_label, flux_label=flux_label,
                               negative=var.metadata.get('negative'))
        elif var_type is PhydraVarType.PARAMETER:
            if var.metadata.get('foreign') is False:
                _par_value = getattr(cls, key)
                cls.m.add_parameter(label=process_label + '_' + key, value=_par_value)
            else:
                raise Exception("Currently Phydra does not support foreign=True for parameters -> TODO 4 v1")


def _create_flux_inputargs_dict(cls, vars_dict):
    """ """
    input_arg_dict = defaultdict(list)

    for key, var in vars_dict.items():
        var_type = var.metadata.get('var_type')
        if var_type is PhydraVarType.VARIABLE:
            if var.metadata.get('foreign') is False:
                var_label = getattr(cls, key + '_label')
            elif var.metadata.get('foreign') is True:
                var_label = getattr(cls, key)
            input_arg_dict['vars'].append({'var': key, 'label': var_label})
        elif var_type is PhydraVarType.PARAMETER:
            # TODO: so it doesn't work with foreign parameters here yet!
            input_arg_dict['pars'].append({'var': key, 'label': cls.label + '_' + key})
        elif var_type is PhydraVarType.FORCING:
            if var.metadata.get('foreign') is False:
                forc_label = getattr(cls, key + '_label')
            elif var.metadata.get('foreign') is True:
                forc_label = getattr(cls, key)
            input_arg_dict['forcs'].append({'var': key, 'label': forc_label})
        elif var_type is PhydraVarType.FLUX:
            group_to_arg = var.metadata.get('group_to_arg')
            if group_to_arg:
                print("ADDING INPUT ARG HERE:", group_to_arg)
                group = list(getattr(cls, group_to_arg))
                print([i for i in group])
                input_arg_dict['vars'].append({'var': group_to_arg, 'label': group})

    print("returning input arg dict", input_arg_dict)

    return input_arg_dict


def _initialize_fluxes(cls, vars_dict):
    """ """
    for key, var in vars_dict.items():
        var_type = var.metadata.get('var_type')
        if var_type is PhydraVarType.FLUX:
            flux_func = var.metadata.get('flux_func')
            label = cls.label + '_' + flux_func.__name__

            if var.metadata.get('group'):
                setattr(cls, flux_func.__name__ + '_label', label)

            setattr(cls, key + '_value',
                    cls.m.register_flux(label=label, flux=cls.flux_decorator(flux_func)))


def _initialize_forcings(cls, forcing_dict):
    """ """
    for var, forc_input_func in forcing_dict.items():
        forc_label = getattr(cls, var + '_label')

        argspec = inspect.getfullargspec(forc_input_func)

        input_args = defaultdict()
        for arg in argspec.args:
            if arg != "self":
                input_args[arg] = getattr(cls, arg)

        forc_func = forc_input_func(cls, **input_args)
        setattr(cls, var + '_value',
                cls.m.add_forcing(label=forc_label, forcing_func=forc_func))


# TODO: currently only function calls phydra.comp() of decorator work,
#  i.e. phydra.comp returns parameterized type obj, not process

def comp(cls=None, *, init_stage=3):
    """ component decorator
    that converts simple base class using phydra.backend.variables into fully functional xarray simlab process
    """

    def create_component(cls):

        attr_cls = attr.attrs(cls, repr=False)
        vars_dict = _create_variables_dict(attr_cls)
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
                args_vectorize_exclude = []

                for v_dict in self.flux_input_args['vars']:
                    if isinstance(v_dict['label'], list):
                        input_args[v_dict['var']] = [state[label] for label in v_dict['label']]
                        args_vectorize_exclude.append(v_dict['var'])
                    else:
                        input_args[v_dict['var']] = state[v_dict['label']]

                for p_dict in self.flux_input_args['pars']:
                    input_args[p_dict['var']] = parameters[p_dict['label']]

                for f_dict in self.flux_input_args['forcs']:
                    input_args[f_dict['var']] = forcings[f_dict['label']]
                    args_vectorize_exclude.append(f_dict['var'])

                # added option to force vectorisation for model arrays/lists
                #   containing objects (i.e. gekko components), excluding the forcings
                try:
                    vectorized = kwargs.pop('vectorized')
                except:
                    vectorized = False

                if vectorized:
                    return np.vectorize(func, excluded=args_vectorize_exclude)(self, **input_args)
                else:
                    return func(self, **input_args)

            return unpack_args

        def initialize(self):
            """ """
            super(new_cls, self).initialize()
            print(f"Initializing component {self.label}")

            _initialize_process_vars(self, vars_dict)

            self.flux_input_args = _create_flux_inputargs_dict(self, vars_dict)
            print(self.flux_input_args)

            _initialize_fluxes(self, vars_dict)

            _initialize_forcings(self, forcing_dict)

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

    if cls:
        return create_component(cls)

    return create_component
