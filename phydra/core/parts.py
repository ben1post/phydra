import attr


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
    # TODO: support variable forcings (i.e. set as array)! currently this produces error
    name = attr.ib()
    value = attr.ib()


@attr.s
class Flux:
    name = attr.ib()
    args = attr.ib()
    equation = attr.ib()
