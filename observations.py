import os
import numpy as np
import xarray as xr
import pandas as pd

class SurfaceObservations(object):
    
    def __init__(self, variable, **kwds):
        self.variable = variable
        self.filepath = ''.join(["/nobackup/users/straaten/E-OBS/", variable, ".nc"])
        for key in kwds.keys():
            setattr(self, key, kwds[key])
    
    def downloadraw(self):
        """
        Downloads daily highres observations on regular 0.25 degree lat-lon grid
        """
        import urllib3
        
        urls = {"tg":"https://www.ecad.eu/download/ensembles/data/Grid_0.25deg_reg/tg_0.25deg_reg_v17.0.nc.gz",
        "tn":"https://www.ecad.eu/download/ensembles/data/Grid_0.25deg_reg/tn_0.25deg_reg_v17.0.nc.gz",
        "tx":"https://www.ecad.eu/download/ensembles/data/Grid_0.25deg_reg/tx_0.25deg_reg_v17.0.nc.gz",
        "rr":"https://www.ecad.eu/download/ensembles/data/Grid_0.25deg_reg/rr_0.25deg_reg_v17.0.nc.gz",
        "pp":"https://www.ecad.eu/download/ensembles/data/Grid_0.25deg_reg/pp_0.25deg_reg_v17.0.nc.gz"}
        
        zippath = self.filepath + '.gz'
        f = open(zippath, 'wb')
        http = urllib3.PoolManager()
        u = http.request('GET', urls[self.variable], preload_content = False)
        filesize = int(u.info().getheaders("Content-Length")[0])
        print("saving", filesize, "to", f.name)
        
        filesize_done = 0
        blocksize = 8192
        while True:
            buffer = u.read(blocksize)
            if not buffer:
                break
            filesize_done += len(buffer)
            f.write(buffer)
            status = r"%10d  [%3.2f%%]" % (filesize_done, filesize_done * 100. / filesize)
            print(status)
        
        u.release_conn()
        f.close()
        os.system("gunzip -k " + zippath) # Results in file written at filepath
        
    def loadraw(self, tmin = None, tmax = None, llcrnr = (None, None), rucrnr = (None,None)):
        """
        Loads the netcdf (possibly delimited by maxtime and corners). Corners need to be given as tuples (lat,lon)
        Creates new attributes: the array with the data, units, and coordinates
        """
    
        #import scipy.io as sio
        
        if not os.path.isfile(self.filepath):
            self.downloadraw()
            
        full = xr.open_dataarray(self.filepath)
        if tmin is None:
            tmin = np.min(full.coords['time'].values)
        if tmax is None:
            tmax = np.max(full.coords['time'].values)
        self.array = full.sel(time = pd.date_range(tmin, tmax, freq = 'D'), longitude = slice(llcrnr[1], rucrnr[1]), latitude = slice(llcrnr[0], rucrnr[0])) # slice gives an inexact lookup, everything within the range
        
        #ifile = sio.netcdf_file(self.filepath, mode = 'r')
        #templons = ifile.variables['longitude'].data
        #templats = ifile.variables['latitude'].data
        #temptime = ifile.variables['time']
        
        
        #variable = ifile.variables[self.variable] # Netcdf variable object referring to file location
        #missval = variable._get_missing_value()
        #self.units = variable.units.decode()

        #self.marray = np.ma.masked_array(variable.data, np.equal(variable.data, missval))
        #del(variable, missval)
        #ifile.close()
    
    def aggregatespace(self, step, by_degree = False):
        """
        Regular lat lon or gridbox aggregation by creating new single coordinate which is used for grouping.
        In the case of degree grouping the groups might not be of equal size.
        """
        if by_degree:
            binlon = pd.cut(self.array.longitude, bins = np.arange(self.array.longitude.min(), self.array.longitude.max(), step))
            binlat = pd.cut(self.array.latitude, bins = np.arange(self.array.latitude.min(), self.array.latitude.max(), step))
        else:
            lon_n, lon_rem = divmod(self.array.longitude.size, step)
            binlon = np.repeat(np.arange(0, lon_n), repeats = step)
            binlon = np.append(binlon, np.repeat(np.NaN, lon_rem))
            lat_n, lat_rem = divmod(self.array.latitude.size, step)
            binlat = np.repeat(np.arange(0, lat_n), repeats = step)
            binlat = np.append(binlat, np.repeat(np.NaN, lat_rem))
        
        # Initialize the grouped extra variables
        lon2 = xr.DataArray(binlon, [self.array.coords['longitude']])
        lat2 = xr.DataArray(binlat, [self.array.coords['latitude']])
        
        stacked = self.array.stack()
        
        nl_avg_lon = self.array.groupby_bins('longitude', np.arange()).mean()
        
    
    def aggregatetime(self, freq = 'w' , method = 'mean'):
        """
        Uses the pandas frequency indicators. Method can be mean, min, max, std
        """
        tempattr = self.array.attrs
        f = getattr(self.array.resample(time = freq), method) # timestamp can be set with label = 'right'
        self.array = f('time') # Removes the attributes.
        tempattr['timemethod'] = '_'.join([freq,method])
        self.array.attrs = tempattr
        # To access days of week: self.array.time.dt.timeofday
        # Also possible is self.array.time.dt.floor('D')
        
    # Define a 'save' method? possibly using the experiment name given through kwargs?   
    # Define a detrend method.
        
# I want to keep the real advanced operations seperate, like PCA, clustering, 
        
# I want to make a class for heatevent. within the definitionmethods I want to have an option to convert the array to a flat dataframe. remove NA'. Then based on previously clustering I can also aggregate by spatial grouping?

# Later on I want to make an experiment class. Combining surface observations on a certain aggregation, with 
