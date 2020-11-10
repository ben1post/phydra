import attr

from enum import Enum

from functools import wraps


class PhydraVarType(Enum):
    VARIABLE = "variable"
    PARAMETER = "parameter"
    FORCING = "forcing"
    FLUX = "flux"


def variable(foreign=False,
             flux=None, negative=False, dims=(),
             description='', attrs=None):

    metadata = {
        "var_type": PhydraVarType.VARIABLE,
        "foreign": foreign,
        "negative": negative,
        "flux": flux,
        "dims": dims,
        "attrs": attrs or {},
        "description": description,
    }

    return attr.attrib(metadata=metadata)


def forcing(foreign=False,
            file_input_func=None, dims=(), description='', attrs=None):

    metadata = {
        "var_type": PhydraVarType.FORCING,
        "foreign": foreign,
        "file_input_func": file_input_func,
        "dims": dims,
        "attrs": attrs or {},
        "description": description,
    }

    return attr.attrib(metadata=metadata)


def parameter(foreign=False, dims=(), description='', attrs=None):

    metadata = {
        "var_type": PhydraVarType.PARAMETER,
        "foreign": foreign,
        "dims": dims,
        "attrs": attrs or {},
        "description": description,
    }

    return attr.attrib(metadata=metadata)


def flux(flux_func=None, *, group_input_arg=None, dims=(), description='', attrs=None):
    """ decorator arg setup allows to be applied to function with and without args """

    def create_attrib(function):

        metadata = {
            "var_type": PhydraVarType.FLUX,
            "flux_func": function,
            "group_input_arg": group_input_arg,
            "dims": dims,
            "attrs": attrs or {},
            "description": description,
        }
        return attr.attrib(metadata=metadata)

    if flux_func:
        return create_attrib(flux_func)

    return create_attrib
