import xso
import pandas
import numpy as np
import scipy.interpolate as intrp


@xso.component
class IrradianceFromLat:
    """Component that calculates daily irradiance from latitude of station."""

    I0 = xso.forcing(setup_func='calculate_I0', description='calculated irradiance for latitude',
                     attrs={'unit': 'W m^-2'})

    station = xso.parameter(description="name of station, options: 'india', 'biotrans', 'kerfix', 'papa'")

    def calculate_I0(self, station):
        """Function adapted from EMPOWER model (Anderson et al. 2015)."""

        def day_length_calc(jday, latradians):
            """Function to calculate day length for location"""
            declin = 23.45 * np.sin(2 * np.pi * (284 + jday) * 0.00274) * np.pi / 180  # solar declination angle
            daylnow = 2 * np.arccos(-1 * np.tan(latradians) * np.tan(declin)) * 12 / np.pi  # day length
            return daylnow

        def noon_PAR_calc(jday, latradians, clouds, e0):
            """Function to calculate noon PAR for location"""
            albedo = 0.04  # albedo
            solarconst = 1368.0  # solar constant, w m-2
            parrac = 0.43  # PAR fraction
            declin = 23.45 * np.sin(2 * np.pi * (284 + jday) * 0.00274) * np.pi / 180  # solar declination angle
            coszen = np.sin(latradians) * np.sin(declin) + np.cos(latradians) * np.cos(declin)  # cosine of zenith angle
            zen = np.arccos(coszen) * 180 / np.pi  # zenith angle, degrees
            Rvector = 1 / np.sqrt(1 + 0.033 * np.cos(2 * np.pi * jday * 0.00274))  # Earth's radius vector
            Iclear = solarconst * coszen ** 2 / (Rvector ** 2) / (
                    1.2 * coszen + e0 * (1.0 + coszen) * 0.001 + 0.0455)  # irradiance at ocean surface, clear sky
            cfac = (1 - 0.62 * clouds * 0.125 + 0.0019 * (90 - zen))  # cloud factor (atmospheric transmission)
            Inoon = Iclear * cfac * (1 - albedo)  # noon irradiance: total solar
            noonparnow = parrac * Inoon
            return noonparnow

        if station == 'india':
            latitude = 60.0  # latitude, degrees
            clouds = 6.0  # cloud fraction, oktas
            e0 = 12.0  # atmospheric vapour pressure
        elif station == 'biotrans':
            latitude = 47.0
            clouds = 6.0
            e0 = 12.0
        elif station == 'kerfix':
            latitude = -50.67
            clouds = 6.0
            e0 = 12.0
        elif station == 'papa':
            latitude = 50.0
            clouds = 6.0
            e0 = 12.0
        else:
            raise ValueError("station label not found, options: 'india', 'biotrans', 'kerfix', 'papa'")

        latradians = latitude * np.pi / 180.

        def return_PAR_forcing(time):
            """Forcing function to return daily PAR for location"""
            day_length = day_length_calc(time, latradians)
            noonpar = noon_PAR_calc(time, latradians, clouds, e0)
            return noonpar * day_length * np.sin(2 / np.pi)  # sinusoidal integration
            # return noonpar * day_length / 2  # trapezoidal integration

        return return_PAR_forcing


@xso.component
class StationForcingFromFile:
    """Component that reads forcing data for EMPOWER stations from file."""

    MLD = xso.forcing(setup_func='create_MLD_forcing', description='Empower MLD Forcing')
    MLDderiv = xso.forcing(setup_func='create_MLD_deriv_forcing', description='Empower MLDderiv Forcing')
    SST = xso.forcing(setup_func='create_SST_forcing', description='Empower SST Forcing')
    N0 = xso.forcing(setup_func='create_N0_forcing', description='Empower N0 Forcing')

    station = xso.parameter(description="name of station, options: 'india', 'biotrans', 'kerfix', 'papa'")

    def read_intrp_forcing(self, station, data, k, smooth, deriv):
        """Method to read forcing data from file and interpolate to daily values."""

        stations_dict = {'india': {'MLD': 'MLD_India', 'SST': 'SST_India'},
                         'biotrans': {'MLD': 'MLD_Biotrans', 'SST': 'SST_Biotrans'},
                         'kerfix': {'MLD': 'MLD_Kerfix', 'SST': 'SST_Kerfix'},
                         'papa': {'MLD': 'MLD_Papa', 'SST': 'SST_Papa'}}

        all_forcings = pandas.read_csv("stations_forcing.txt", sep=r'\s*,\s*',
                                       header=0, encoding='ascii', engine='python')

        station_data = all_forcings[stations_dict[station][data]].values[:-1]

        dayspermonth = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        dpm = dayspermonth
        dpm_cumsum = np.cumsum(dpm) - np.array(dpm) / 2

        time = np.concatenate([[0], dpm_cumsum, [365]], axis=None)

        boundary_int = [(station_data[0] + station_data[-1]) / 2]
        dat = np.concatenate([boundary_int, station_data, boundary_int], axis=None)

        spl = intrp.splrep(time, dat, per=True, k=k, s=smooth)

        def forcing(time):
            """Forcing function to return interpolated daily forcing for location"""
            return intrp.splev(np.mod(time, 365), spl, der=deriv)

        return forcing

    def create_MLD_forcing(self, station):
        return self.read_intrp_forcing(station, 'MLD', deriv=0, k=1, smooth=1)

    def create_MLD_deriv_forcing(self, station):
        return self.read_intrp_forcing(station, 'MLD', deriv=1, k=1, smooth=1)

    def create_SST_forcing(self, station):
        return self.read_intrp_forcing(station, 'SST', deriv=0, k=1, smooth=1)

    def create_N0_forcing(self, station):
        MLD_func = self.read_intrp_forcing(station, 'MLD', deriv=0, k=1, smooth=1)

        if station == 'india':
            aN = 0.0074  # coeff. for N0 as fn depth
            bN = 10.85  # coeff. for N0 as fn depth
        elif station == 'biotrans':
            aN = 0.0174
            bN = 4.0
        elif station == 'kerfix':
            aN = 0
            bN = 26.1
        elif station == 'papa':
            aN = 0.0
            bN = 14.6
        else:
            raise ValueError("station label not found, options: 'india', 'biotrans', 'kerfix', 'papa'")

        def N0_forcing(time):
            return aN * MLD_func(time) + bN

        return N0_forcing
