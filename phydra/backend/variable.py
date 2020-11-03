import attr

from enum import Enum

import xsimlab as xs


class FluxVarType(Enum):
    VARIABLE = "variable"
    PARAMETER = "parameter"
    FORCING = "forcing"


def variable(foreign=False,
        flux=None, negative=False, dims=[()], partial_out=None, sub_label=None, description='', attrs=None):

    metadata = {
        "var_type": FluxVarType.VARIABLE,
        "foreign": foreign,
        "negative": negative,
        "flux": flux,
        "dims": dims,
        "partial_out": partial_out,
        "sub_label": sub_label,
        "attrs": attrs or {},
        "description": description,
    }

    return attr.attrib(metadata=metadata)


def forcing(foreign=False,
            file_input_func=None, dims=[()], sub_label=None, description='', attrs=None):

    metadata = {
        "var_type": FluxVarType.FORCING,
        "foreign": foreign,
        "file_input_func": file_input_func,
        "dims": dims,
        "sub_label": sub_label,
        "attrs": attrs or {},
        "description": description,
    }

    return attr.attrib(metadata=metadata)


def parameter(foreign=False, dims=[()], sub_label=None, description='', attrs=None):

    metadata = {
        "var_type": FluxVarType.PARAMETER,
        "foreign": foreign,
        "dims": dims,
        "sub_label": sub_label,
        "attrs": attrs or {},
        "description": description,
    }

    return attr.attrib(metadata=metadata)

