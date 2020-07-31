import numpy as np
from gekko import GEKKO

from collections import defaultdict
from scipy.integrate import odeint

# to measure process time
import time as tm
import attr

@attr.s
class StateVariable:
    name = attr.ib()
    initial_value = attr.ib()
    lb = attr.ib(default=0)


@attr.s
class Parameter:
    # usually constant
    name = attr.ib()
    value = attr.ib()


@attr.s
class Forcing:
    # usually changes over time
    name = attr.ib()
    value = attr.ib()


@attr.s
class Flux:
    name = attr.ib()
    args = attr.ib()
    equation = attr.ib()


class PhydraCore:
    def __init__(self):
        self.time = None
        self.SVs = defaultdict()
        self.Parameters = defaultdict()
        self.Forcings = defaultdict()
        self.Fluxes = defaultdict(list)

    def __repr__(self):
        return f"Model contains: \n SVs:{self.SVs} \n Params:{self.Parameters}\n Forcings:{self.Forcings}\n Fluxes:{self.Fluxes}"

    def odeint_solve(self):
        C = OdeintConverter()
        # here create function def model(y,t)

        y_labels = [label for label in self.SVs.keys()]
        y_values = [C.convert(SV) for SV in self.SVs.values()]

        print(self.Parameters)

        parameters = {Param.name: Param.value for Param in self.Parameters.values()}
        print(parameters)

        print(y_labels, y_values)

        fluxes = {label: flux
                  for label, flux in zip(y_labels, self.Fluxes.values())}
        print(fluxes)

        def ode(c, t):
            state = {label: val for label, val in zip(y_labels, c)}
            return [sum(flux(state, parameters) for flux in fluxes[label]) for label in y_labels]

        if self.time is None:
            raise Exception('time needs to be supplied before solve')

        c_out = odeint(ode, y_values, self.time)
        # print(c_out)
        c_rows = (row for row in c_out.T)
        # print(c_rows)
        c_dict = {y_label: vals for y_label, vals in zip(y_labels, c_rows)}
        return c_dict

    def gekko_solve(self):

        C = GekkoConverter()
        # here create function def model(y,t)

        y_labels = [label for label in self.SVs.keys()]
        y_values = [C.convert(SV) for SV in self.SVs.values()]

        print(self.Parameters)

        parameters = {Param.name: Param.value for Param in self.Parameters.values()}
        print(parameters)

        print(y_labels, y_values)

        fluxes = {label: flux
                  for label, flux in zip(y_labels, self.Fluxes.values())}
        print(fluxes)

        state = {label: val for label, val in zip(y_labels, y_values)}

        C.gekko.Equations(
            [SV.dt() == C.gekko.sum([flux(state, parameters) for flux in fluxes[SV.name]]) for SV in y_values]
        )

        if self.time is None:
            raise Exception('time needs to be supplied before solve')

        C.gekko.time = self.time

        C.gekko.options.IMODE = 7  # sequential dynamic solver

        print(C.gekko.__dict__)

        solve_start = tm.time()
        C.gekko.solve(disp=False)  # use option disp=True to print gekko output
        solve_end = tm.time()

        print(f"Model was solved in {round(solve_end - solve_start, 2)} seconds")

        return state


