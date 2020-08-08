from gekko import GEKKO

from .parts import StateVariable, Forcing, Flux, Parameter

# Utilizing a visitor pattern
# to allow seamless switching between solver methods

# code copied and modified from Joren Van Severen:
# https://stackoverflow.com/questions/25891637/visitor-pattern-in-python

def _qualname(obj):
    """Get the fully-qualified name of an object (including module)."""
    return obj.__module__ + '.' + obj.__qualname__


def _declaring_class(obj):
    """Get the name of the class that declared an object."""
    name = _qualname(obj)
    print(name)
    return name[:name.rfind('.')]


# Stores the actual visitor methods
_methods = {}


# Delegating visitor implementation
def _convertor_impl(self, arg):
    """Actual visitor method implementation."""
    method = _methods[(_qualname(type(self)), type(arg))]
    return method(self, arg)


# The actual @visitor decorator
def convertor(arg_type):
    """Decorator that creates a visitor method."""

    # @wraps(arg_type)
    def decorator(fn):
        declaring_class = _declaring_class(fn)
        _methods[(declaring_class, arg_type)] = fn

        # Replace all decorated methods with _visitor_impl
        return _convertor_impl

    return decorator



class OdeintConverter:

    @convertor(StateVariable)
    def convert(self, obj):
        return obj

    @convertor(Parameter)
    def convert(self, obj):
        return obj.name, obj.value

    @convertor(Forcing)
    def convert(self, obj):
        # this should return function of t
        return obj.value, obj.name

    @convertor(Flux)
    def convert(self, obj):
        return obj.equation, obj.name



class GekkoContext:
    def __init__(self):
        self.gekko = GEKKO()


class GekkoConverter(GekkoContext):

    @convertor(StateVariable)
    def convert(self, obj):
        return self.gekko.SV(obj.initial_value, name=obj.name, lb=obj.lb)

    @convertor(Parameter)
    def convert(self, obj):
        return self.gekko.Param(obj.value, name=obj.name)

    @convertor(Forcing)
    def convert(self, obj):
        # this should return m.Param, discretized
        return self.gekko.Param(obj.value, name=obj.name)

    @convertor(Flux)
    def convert(self, obj):
        return self.gekko.Intermediate(obj.equation, name=obj.name)
