import os
import numpy as np

from scipy.io import netcdf

# get phydra-data from context for now
from pathlib import Path

class WOAForcing:
    """
    initializes and reads forcing from a certain location in the WOA 2009 data, contained in ncdf files

    """
    def __init__(self, lat, lon, rangebb, varname):
        self.Lat = lat
        self.Lon = lon
        self.RangeBB = rangebb
        self.varname = varname
        self.fordir = str(Path(__file__).resolve().parents[3]) + '/phydra-data'
        self.outForcing = self.spatialave()

    def spatialave(self):
        """
        Method to extract spatially averaged environmental forcing.

        Returns
        -------
        The spatial average of the respective environmental forcing per month.
        """

        if self.varname == 'mld':
            ncfile = netcdf.netcdf_file(self.fordir +'/MLDclimatology_DeBoyerMontegut/mld_mindtr02_l3_nc3.nc', 'r')
            # print(ncfile.dimensions)
            nclat = ncfile.variables['lat'].data.copy()
            nclon = ncfile.variables['lon'].data.copy()
            ncdat = ncfile.variables['mld_mindtr02_rmoutliers_smth_okrg'].data.copy()
            ncfile.close()
            longrid, latgrid = np.meshgrid(nclon, nclat)
            selectarea = np.logical_and(longrid <= self.Lon + self.RangeBB, longrid >= self.Lon - self.RangeBB) * \
                         np.logical_and(latgrid <= self.Lat + self.RangeBB, latgrid >= self.Lat - self.RangeBB)
            outforcing = list(np.nanmean(ncdat[:, selectarea], axis=1))
            return outforcing * 3

        elif self.varname == 'par':
            ncfile = netcdf.netcdf_file(self.fordir + '/MODISaqua/PARclimatology_MODISaqua_L3_nc3.nc', 'r')
            nclat = ncfile.variables['lat'].data.copy()
            nclon = ncfile.variables['lon'].data.copy()
            ncdat = ncfile.variables['par'].data.copy()
            ncfile.close()
            longrid, latgrid = np.meshgrid(nclon, nclat)
            selectarea = np.logical_and(longrid <= self.Lon + self.RangeBB, longrid >= self.Lon - self.RangeBB) * \
                         np.logical_and(latgrid <= self.Lat + self.RangeBB, latgrid >= self.Lat - self.RangeBB)
            outforcing = list(np.nanmean(ncdat[:, selectarea], axis=1))
            return outforcing * 3

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
            return outforcing * 3

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
            return outforcing * 3

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
            return outforcing * 3

        elif self.varname == 'sst':
            ncfile = netcdf.netcdf_file(self.fordir + 'WOA2018/TempAboveMLD_WOA.nc', 'r')
            nclat = ncfile.variables['lat'].data.copy()
            nclon = ncfile.variables['lon'].data.copy()
            ncdat = ncfile.variables['t_mld'].data.copy()
            ncfile.close()
            longrid, latgrid = np.meshgrid(nclon, nclat)
            selectarea = np.logical_and(latgrid <= self.Lat + self.RangeBB, latgrid >= self.Lat - self.RangeBB) * \
                         np.logical_and(longrid <= self.Lon + self.RangeBB, longrid >= self.Lon - self.RangeBB)
            outforcing = list(np.nanmean(ncdat[selectarea, :], axis=0))
            return outforcing * 3
        else:
            return 'Please specify either mld, par, n0x or sst'