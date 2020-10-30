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
            intent='in', dims=[()], sub_label=None, description='', attrs=None):

    metadata = {
        "var_type": FluxVarType.FORCING,
        "foreign": foreign,
        "intent": intent,
        "dims": dims,
        "sub_label": sub_label,
        "attrs": attrs or {},
        "description": description,
    }

    return attr.attrib(metadata=metadata)


def parameter(foreign=False,
              intent='in', dims=[()], sub_label=None, description='', attrs=None):

    metadata = {
        "var_type": FluxVarType.PARAMETER,
        "foreign": foreign,
        "intent": intent,
        "dims": dims,
        "sub_label": sub_label,
        "attrs": attrs or {},
        "description": description,
    }

    return attr.attrib(metadata=metadata)

