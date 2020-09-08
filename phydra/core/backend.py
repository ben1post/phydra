from collections import defaultdict
from scipy.integrate import odeint
import numpy as np

# to measure process Time
import time as tm

from .converters import BaseConverter, GekkoConverter


class ModelBackend:
    def __init__(self, solver_type):
        self.Solver = solver_type
        self.Time = None

        self.SVs = defaultdict()
        self.Parameters = defaultdict()
        self.Forcings = defaultdict()
        self.Fluxes = defaultdict(list)
        self.MultiFluxes = defaultdict()

        self.sv_labels = None
        self.sv_values = None

        if self.Solver == "odeint":
            self.core = BaseConverter()
        elif self.Solver == "stepwise":
            self.core = BaseConverter()
        elif self.Solver == "gekko":
            self.core = GekkoConverter()
        else:
            raise Exception("Please provide Solver type to core, can be 'gekko', 'odeint' or 'stepwise")

    def __repr__(self):
        return (f"Model contains: \n"
                f"SVs:{self.SVs} \n"
                f"Params:{self.Parameters} \n"
                f"Forcings:{self.Forcings} \n"
                f"Fluxes:{self.Fluxes}")

    def setup_SV(self, label, SV):
        """
        function that takes the state variable setup as input
        and returns the storage values
        and adds state variable to SVs defaultdict in the background
        """
        # the following step registers the values with gekko, but simply returns SV for odeint
        self.SVs[label] = self.core.convert(SV)

        if self.Solver == "gekko":
            # return view on gekko intermediate
            return self.SVs[label].value
        elif self.Solver == "odeint":
            # return view on empty numpy array of zeroes, that is filled after solve
            self.SVs[label].value = np.zeros(np.shape(self.Time))
            return self.SVs[label].value
        elif self.Solver == "stepwise":
            # return list of values to be appended to
            self.SVs[label].value = [self.SVs[label].initial_value]
            return self.SVs[label].value

    def cleanup(self):
        if self.core is not None:
            if self.Solver == "gekko":
                self.core.gekko.cleanup()
        else:
            pass

    def model(self, current_state, time):
        state = {label: val for label, val in zip(self.sv_labels, current_state)}

        multifluxes = defaultdict(list)

        for label, multiflx in self.MultiFluxes.items():
            flx = multiflx['flux'](state=state, parameters=self.parameters, forcings=self.forcings)

            if len(multiflx['routing'].keys()) != len(flx):
                raise Exception("Damn! better check your MultiFlux routing, "
                                "dimensions of func output do not match.")

            for sv_label, sub_flx in zip(multiflx['routing'].keys(), flx):
                multifluxes[sv_label].append(sub_flx)

        fluxes_out = []
        for label in self.sv_labels:
            sv_fluxes = []
            for sub_flux in multifluxes[label]:
                sv_fluxes.append(sub_flux)
            for flux in self.Fluxes[label]:
                sv_fluxes.append(flux(state=state, parameters=self.parameters, forcings=self.forcings))

            fluxes_out.append(sum(sv_fluxes))

        return fluxes_out

    def assemble(self):
        """Assembles model for all Solver types"""
        self.sv_labels = [label for label in self.SVs.keys()]
        self.sv_values = [SV for SV in self.SVs.values()]

        self.parameters = {Param.name: Param.value for Param in self.Parameters.values()}
        self.forcings = {Param.name: Param.value for Param in self.Forcings.values()}

    def solve(self, time_step):
        if self.Solver == "gekko":
            self.core.gekko.options.REDUCE = 3  # handles reduction of larger models, have not benchmarked it yet
            self.core.gekko.options.NODES = 3  # improves solution accuracy
            self.core.gekko.options.IMODE = 5  # 7  # sequential dynamic Solver

            self.gekko_solve()  # use option disp=True to print gekko output

        elif self.Solver == "odeint":
            self.odeint_solve()

        elif self.Solver == "stepwise":
            self.step_solve(time_step)

        else:
            raise Exception("Please provide Solver type to core, can be 'gekko', 'odeint' or 'stepwise")

    def step_solve(self, time_step):
        """XXX"""
        sv_state = [SV.value[-1] for SV in self.SVs.values()]
        state_out = self.model(sv_state, self.Time)
        state_dict = {sv_label: vals for sv_label, vals in zip(self.sv_labels, state_out)}

        for sv_val in self.sv_values:
            # model calculates derivative, so needs to be computed to value with previous value
            state = sv_val.value[-1] + state_dict[sv_val.name] * time_step
            sv_val.value.append(state)

    def odeint_solve(self):
        """XXX"""
        y_init = [SV.initial_value for SV in self.SVs.values()]

        if self.Time is None:
            raise Exception('Time needs to be supplied before solve')

        print("start solve now")
        solve_start = tm.time()
        state_out = odeint(self.model, y_init, self.Time)
        solve_end = tm.time()
        print(f"Model was solved in {round(solve_end - solve_start, 5)} seconds")

        state_rows = (row for row in state_out.T)
        state_dict = {y_label: vals for y_label, vals in zip(self.sv_labels, state_rows)}

        for sv_val in self.sv_values:
            print('here unpacking values', sv_val.name)
            sv_val.value[:] = state_dict[sv_val.name]

    def gekko_solve(self, disp=False):
        """XXX"""
        sv_states = [sv for sv in self.sv_values]
        state_out = self.model(sv_states, self.Time)
        state_dict = {sv_label: eq for sv_label, eq in zip(self.sv_labels, state_out)}

        equations = []

        for SV, label in zip(self.sv_values, self.sv_labels):
            print(SV, state_dict[label])
            try:
                state_dict[label]
            except KeyError:
                # if not, define derivative as 0
                equations.append(SV.dt() == 0)
            else:
                equations.append(SV.dt() == state_dict[label])

        # create Equations
        self.core.gekko.Equations(equations)


        if self.Time is None:
            raise Exception('Time needs to be supplied before solve')

        self.core.gekko.time = self.Time

        print(self.core.gekko.__dict__)
        print([val.value for val in self.core.gekko.__dict__['_equations']])

        solve_start = tm.time()
        self.core.gekko.solve(disp=disp)  # use option disp=True to print gekko output
        solve_end = tm.time()

        print(f"Model was solved in {round(solve_end - solve_start, 2)} seconds")
