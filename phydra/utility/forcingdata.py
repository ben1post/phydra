import os
import numpy as np
import pandas as pd

from scipy.io import netcdf

# get phydra-data from context for now
from pathlib import Path
# TODO: Get this data in smaller files on server to query as described in:
# https://zonca.dev/2019/08/large-files-python-packages.html


class ClimatologyForcing:
    """
    initializes and reads forcing from a certain location in the WOA 2009 data, contained in ncdf files

    """
    def __init__(self, lat, lon, rangebb, varname):
        self.Lat = lat
        self.Lon = lon
        self.RangeBB = rangebb
        self.varname = varname
        self.fordir = str(Path(__file__).resolve().parents[3]) + '/phydra-data/'
        self.outForcing = self.spatialave()

    def spatialave(self):
        """
        Method to extract spatially averaged environmental forcing.

        Returns
        -------
        The spatial average of the respective environmental forcing per month.
        """

        if self.varname == 'mld':
            ncfile = netcdf.netcdf_file(self.fordir +'MLDclimatology_DeBoyerMontegut/mld_mindtr02_l3_nc3.nc', 'r')
            # print(ncfile.dimensions)
            nclat = ncfile.variables['lat'].data.copy()
            nclon = ncfile.variables['lon'].data.copy()
            ncdat = ncfile.variables['mld_mindtr02_rmoutliers_smth_okrg'].data.copy()
            ncfile.close()
            longrid, latgrid = np.meshgrid(nclon, nclat)
            selectarea = np.logical_and(longrid <= self.Lon + self.RangeBB, longrid >= self.Lon - self.RangeBB) * \
                         np.logical_and(latgrid <= self.Lat + self.RangeBB, latgrid >= self.Lat - self.RangeBB)
            outforcing = list(np.nanmean(ncdat[:, selectarea], axis=1))
            return outforcing

        elif self.varname == 'par':
            ncfile = netcdf.netcdf_file(self.fordir + 'MODISaqua/PARclimatology_MODISaqua_L3.nc', 'r')
            nclat = ncfile.variables['lat'].data.copy()
            nclon = ncfile.variables['lon'].data.copy()
            ncdat = ncfile.variables['par'].data.copy()
            ncfile.close()
            longrid, latgrid = np.meshgrid(nclon, nclat)
            selectarea = np.logical_and(longrid <= self.Lon + self.RangeBB, longrid >= self.Lon - self.RangeBB) * \
                         np.logical_and(latgrid <= self.Lat + self.RangeBB, latgrid >= self.Lat - self.RangeBB)
            outforcing = list(np.nanmean(ncdat[:, selectarea], axis=1))
            return outforcing

        elif self.varname == 'n0x':
            ncfile = netcdf.netcdf_file(self.fordir + 'WOA2018/NitrateAboveMLD_WOA.nc', 'r')
            nclat = ncfile.variables['lat'].data.copy()
            nclon = ncfile.variables['lon'].data.copy()
            ncdat = ncfile.variables['n0'].data.copy()
            ncfile.close()
            longrid, latgrid = np.meshgrid(nclon, nclat)
            selectarea = np.logical_and(latgrid <= self.Lat + self.RangeBB, latgrid >= self.Lat - self.RangeBB) * \
                         np.logical_and(longrid <= self.Lon + self.RangeBB, longrid >= self.Lon - self.RangeBB)
            outforcing = list(np.nanmean(ncdat[selectarea, :], axis=0))
            return outforcing

        elif self.varname == 'p0x':
            ncfile = netcdf.netcdf_file(self.fordir + 'WOA2018/PhosphateAboveMLD_WOA.nc', 'r')
            nclat = ncfile.variables['lat'].data.copy()
            nclon = ncfile.variables['lon'].data.copy()
            ncdat = ncfile.variables['p0'].data.copy()
            ncfile.close()
            longrid, latgrid = np.meshgrid(nclon, nclat)
            selectarea = np.logical_and(latgrid <= self.Lat + self.RangeBB, latgrid >= self.Lat - self.RangeBB) * \
                         np.logical_and(longrid <= self.Lon + self.RangeBB, longrid >= self.Lon - self.RangeBB)
            outforcing = list(np.nanmean(ncdat[selectarea, :], axis=0))
            return outforcing

        elif self.varname == 'si0x':
            ncfile = netcdf.netcdf_file(self.fordir + 'WOA2018/SilicateAboveMLD_WOA.nc', 'r')
            nclat = ncfile.variables['lat'].data.copy()
            nclon = ncfile.variables['lon'].data.copy()
            ncdat = ncfile.variables['si0'].data.copy()
            ncfile.close()
            longrid, latgrid = np.meshgrid(nclon, nclat)
            selectarea = np.logical_and(latgrid <= self.Lat + self.RangeBB, latgrid >= self.Lat - self.RangeBB) * \
                         np.logical_and(longrid <= self.Lon + self.RangeBB, longrid >= self.Lon - self.RangeBB)
            outforcing = list(np.nanmean(ncdat[selectarea, :], axis=0))
            return outforcing

        elif self.varname == 'tmld':
            ncfile = netcdf.netcdf_file(self.fordir + 'WOA2018/TempAboveMLD_WOA.nc', 'r')
            nclat = ncfile.variables['lat'].data.copy()
            nclon = ncfile.variables['lon'].data.copy()
            ncdat = ncfile.variables['t_mld'].data.copy()
            ncfile.close()
            longrid, latgrid = np.meshgrid(nclon, nclat)
            selectarea = np.logical_and(latgrid <= self.Lat + self.RangeBB, latgrid >= self.Lat - self.RangeBB) * \
                         np.logical_and(longrid <= self.Lon + self.RangeBB, longrid >= self.Lon - self.RangeBB)
            outforcing = list(np.nanmean(ncdat[selectarea, :], axis=0))
            return outforcing
        else:
            return 'Please specify either mld, par, n0x or tmld'


class VerifData:
    """
    initializes and reads verification data, contained in ncdf files
    """

    def __init__(self, lat, lon, rbb, kostadinov=False):
        self.Lat = lat
        self.Lon = lon
        self.RangeBB = rbb
        self.fordir = str(Path(__file__).resolve().parents[3]) + '/phydra-data/'

        self.chla = self.readchla()
        #self.chlaint = self.interplt(self.chla)
        self.N = self.readnutrientaboveMLD('n0')
        #self.Nint = self.interplt(self.N)

        if kostadinov:
            self.carbontotal = self.readKostadinovCarbonSize('C_biomass_total')
            self.c_microp = self.readKostadinovCarbonSize('C_biomass_microplankton')
            self.c_nanop = self.readKostadinovCarbonSize('C_biomass_nanoplankton')
            self.c_picop = self.readKostadinovCarbonSize('C_biomass_picoplankton')

            self.carbontotal_sd = self.readKostadinovCarbonSize('Composite_standard_deviation_C_biomass_total')
            self.c_microp_sd = self.readKostadinovCarbonSize('Composite_standard_deviation_C_biomass_microplankton')
            self.c_nanop_sd = self.readKostadinovCarbonSize('Composite_standard_deviation_C_biomass_nanoplankton')
            self.c_picop_sd = self.readKostadinovCarbonSize('Composite_standard_deviation_C_biomass_picoplankton')

        print('VerifData forcing created')

    def readKostadinovCarbonSize(self, var):
        ncfile = netcdf.netcdf_file(self.fordir + 'Kostadinov2016/Kostadinov_' + var + '.nc', 'r')
        # ncfile = netcdf.netcdf_file(self.fordir + '/Kostadinov_'+ var + '.nc', 'r')
        nclat = ncfile.variables['Latitude'].data.copy()
        nclon = ncfile.variables['Longitude'].data.copy()
        ncdat = ncfile.variables[var].data.copy()
        ncdat[ncdat < 0] = np.nan
        ncfile.close()
        longrid, latgrid = np.meshgrid(nclon, nclat)
        selectarea = np.logical_and(longrid <= self.Lon + self.RangeBB, longrid >= self.Lon - self.RangeBB) * \
                     np.logical_and(latgrid <= self.Lat + self.RangeBB, latgrid >= self.Lat - self.RangeBB)
        outforcing = list(np.nanmean(ncdat[:, selectarea.T], axis=1))
        return outforcing

    def readchla(self):
        ncfile = netcdf.netcdf_file(self.fordir + 'MODISaqua/ChlAclimatology_MODISaqua_L3.nc', 'r')
        nclat = ncfile.variables['lat'].data.copy()
        nclon = ncfile.variables['lon'].data.copy()
        ncdat = ncfile.variables['chlor_a'].data.copy()
        ncfile.close()
        longrid, latgrid = np.meshgrid(nclon, nclat)
        selectarea = np.logical_and(longrid <= self.Lon + self.RangeBB, longrid >= self.Lon - self.RangeBB) * \
                     np.logical_and(latgrid <= self.Lat + self.RangeBB, latgrid >= self.Lat - self.RangeBB)
        outforcing = list(np.nanmean(ncdat[:, selectarea], axis=1))
        return outforcing

    def readnutrientaboveMLD(self, var):
        if var == 'n0':
            ncfile = netcdf.netcdf_file(self.fordir + 'WOA2018/NitrateAboveMLD_WOA.nc', 'r')
        elif var == 'p0':
            ncfile = netcdf.netcdf_file(self.fordir + 'WOA2018/Phosphate_WOA.nc', 'r')
        elif var == 'si0':
            ncfile = netcdf.netcdf_file(self.fordir + 'WOA2018/SilicateAboveMLD_WOA.nc', 'r')

        nclat = ncfile.variables['lat'].data.copy()
        nclon = ncfile.variables['lon'].data.copy()
        ncdat = ncfile.variables[var].data.copy()
        ncfile.close()
        longrid, latgrid = np.meshgrid(nclon, nclat)
        selectarea = np.logical_and(latgrid <= self.Lat + self.RangeBB, latgrid >= self.Lat - self.RangeBB) * \
                     np.logical_and(longrid <= self.Lon + self.RangeBB, longrid >= self.Lon - self.RangeBB)
        outforcing = list(np.nanmean(ncdat[selectarea, :], axis=0))
        return outforcing

    def interplt(self, dat):
        dayspermonth = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        dpm_3 = dayspermonth * 3
        dpm_cumsum = np.cumsum(dpm_3) - np.array(dpm_3) / 2
        dat_dpm = pd.DataFrame(np.column_stack([dat * 3, dpm_cumsum]), columns=['Value', 'yday'])
        tm_dat_conc = np.arange(0., 3 * 365., 1.0)

        dat_pad = dat_dpm.set_index('yday').reindex(tm_dat_conc).reset_index()
        dat_int = dat_pad.Value.interpolate().values[365:365 + 365]
        return dat_int