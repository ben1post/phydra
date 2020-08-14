from collections import defaultdict
from scipy.integrate import odeint
import numpy as np

# to measure process time
import time as tm

from .converters import BaseConverter, GekkoConverter

class ModelBackend:
    def __init__(self, solver_type):
        self.solver = solver_type
        self.time = None
        self.model = None
        self.SVs = defaultdict()
        self.Parameters = defaultdict()
        self.Forcings = defaultdict()
        self.Fluxes = defaultdict(list)

        self.sv_labels = None
        self.sv_values = None

        if self.solver == "odeint":
            self.core = BaseConverter()
        elif self.solver == "gekko":
            self.core = GekkoConverter()
        elif self.solver == "stepwise":
            self.core = BaseConverter()
        else:
            raise Exception("Please provide solver type to core, can be 'gekko', 'odeint' or 'stepwise")

    def __repr__(self):
        return f"Model contains: \n SVs:{self.SVs} \n Params:{self.Parameters}\n Forcings:{self.Forcings}\n Fluxes:{self.Fluxes}"

    def setup_SV(self, label, SV):
        """
        function that takes the state variable setup as input
        and returns the storage values
        and adds state variable to SVs defaultdict in the background
        """
        # the following step registers the values with gekko, but simply returns SV for odeint
        self.SVs[label] = self.core.convert(SV)

        if self.solver == "gekko":
            # return view on gekko intermediate
            return self.SVs[label].value
        elif self.solver == "odeint":
            # return view on empty numpy array of zeroes, that is filled after solve
            self.SVs[label].value = np.zeros(np.shape(self.time))
            return self.SVs[label].value
        elif self.solver == "stepwise":
            # return list of values to be appended to
            self.SVs[label].value = [self.SVs[label].initial_value]
            return self.SVs[label].value

    def cleanup(self):
        if self.core is not None:
            if self.solver == "gekko":
                self.core.gekko.cleanup()
        else:
            pass

    def assemble(self):
        """Assembles model for all solver types"""
        self.sv_labels = [label for label in self.SVs.keys()]
        self.sv_values = [SV for SV in self.SVs.values()]

        self.parameters = {Param.name: Param.value for Param in self.Parameters.values()}

        def ode(c, t):
            state = {label: val for label, val in zip(self.sv_labels, c)}
            return [sum(flux(state, self.parameters) for flux in self.Fluxes[label]) for label in self.sv_labels]

        self.model = ode

    def solve(self, time_step):
        if self.solver == "gekko":
            self.core.gekko.options.REDUCE = 3  # handles reduction of larger models, have not benchmarked it yet
            self.core.gekko.options.NODES = 3  # improves solution accuracy
            self.core.gekko.options.IMODE = 5  # 7  # sequential dynamic solver
            self.gekko_solve()  # use option disp=True to print gekko output

        elif self.solver == "odeint":
            self.odeint_solve()

        elif self.solver == "stepwise":
            self.step_solve(time_step)

        else:
            raise Exception("Please provide solver type to core, can be 'gekko', 'odeint' or 'stepwise")

    def step_solve(self, time_step):
        """XXX"""
        sv_state = [SV.value[-1] for SV in self.SVs.values()]

        c_out = self.model(sv_state, self.time)

        c_dict = {sv_label: vals for sv_label, vals in zip(self.sv_labels, c_out)}

        for sv_val in self.sv_values:
            # model calculates derivative, so needs to be computed to value with previous value
            state = sv_val.value[-1] + c_dict[sv_val.name] * time_step
            sv_val.value.append(state)

    def odeint_solve(self):
        """XXX"""
        y_init = [SV.initial_value for SV in self.SVs.values()]

        if self.time is None:
            raise Exception('time needs to be supplied before solve')

        print("start solve now")
        solve_start = tm.time()
        c_out = odeint(self.model, y_init, self.time)
        solve_end = tm.time()
        print(f"Model was solved in {round(solve_end - solve_start, 5)} seconds")

        # print(c_out)
        c_rows = (row for row in c_out.T)
        # print(c_rows)
        c_dict = {y_label: vals for y_label, vals in zip(self.sv_labels, c_rows)}

        for y_val in self.sv_values:
            print('here unpacking values', y_val.name)
            y_val.value[:] = c_dict[y_val.name]

    def gekko_solve(self, disp=False):
        """XXX"""
        fluxes = {label: flux
                  for label, flux in zip(self.sv_labels, self.Fluxes.values())}

        state = {label: val for label, val in zip(self.sv_labels, self.sv_values)}

        self.core.gekko.Equations(
            [SV.dt() == sum([flux(state, self.parameters) for flux in fluxes[SV.name]]) for SV in self.sv_values]
        )

        if self.time is None:
            raise Exception('time needs to be supplied before solve')

        self.core.gekko.time = self.time

        solve_start = tm.time()
        self.core.gekko.solve(disp=disp)  # use option disp=True to print gekko output
        solve_end = tm.time()

        print(f"Model was solved in {round(solve_end - solve_start, 2)} seconds")
