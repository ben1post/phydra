import attr
import numpy as np

@attr.s
class StateVariable:
    name = attr.ib()
    initial_value = attr.ib(default=0.)
    value = attr.ib(default=None)
    lb = attr.ib(default=0)


@attr.s
class Parameter:
    # usually constant
    name = attr.ib()
    value = attr.ib()


@attr.s
class Forcing:
    # usually changes over Time
    name = attr.ib()
    value = attr.ib()


@attr.s
class Flux:
    name = attr.ib()
    args = attr.ib()
    equation = attr.ib()
