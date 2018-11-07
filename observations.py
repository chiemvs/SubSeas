import os
import numpy as np
import xarray as xr
import pandas as pd
import itertools

class SurfaceObservations(object):
    
    def __init__(self, variable, **kwds):
        self.variable = variable
        self.storagedir = "/nobackup/users/straaten/E-OBS/"
        self.filepath = ''.join([self.storagedir, variable, ".nc"])
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
        
        if not os.path.isfile(zippath):
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

        
    def load(self, lazy = True, tmin = None, tmax = None, llcrnr = (None, None), rucrnr = (None,None)):
        """
        Loads the netcdf (possibly delimited by maxtime and corners). Corners need to be given as tuples (lat,lon)
        Creates new attributes: the array with the data, units, and coordinates, and possibly a timemethod.
        """
        
        if not os.path.isfile(self.filepath):
            self.downloadraw()
        if lazy:
            full = xr.open_dataarray(self.filepath, chunks={'time': 3650}) # ten year chunks
        else:
            full = xr.open_dataarray(self.filepath)
            
        if tmin is None:
            tmin = np.min(full.coords['time'].values)
        if tmax is None:
            tmax = np.max(full.coords['time'].values)
        
        freq = pd.infer_freq(full.coords['time'].values)
        self.array = full.sel(time = pd.date_range(tmin, tmax, freq = freq), longitude = slice(llcrnr[1], rucrnr[1]), latitude = slice(llcrnr[0], rucrnr[0])) # slice gives an inexact lookup, everything within the range
        if not hasattr(self.array, 'timemethod'):
            self.array.attrs['timemethod'] = freq # Useful for saving method later on, so original data of full temporal range is not overwritten.

    def aggregatespace(self, step, method = 'mean', by_degree = False):
        """
        Regular lat lon or gridbox aggregation by creating new single coordinate which is used for grouping.
        In the case of degree grouping the groups might not contain an equal number of cells.
        Completely lazy when loading was lazy
        """
        if by_degree:
            binlon = pd.cut(self.array.longitude, bins = np.arange(self.array.longitude.min(), self.array.longitude.max(), step), include_lowest = True) # Lowest is included because otherwise NA's arise at begin and end, making a single group
            binlat = pd.cut(self.array.latitude, bins = np.arange(self.array.latitude.min(), self.array.latitude.max(), step), include_lowest = True)
        else:
            lon_n, lon_rem = divmod(self.array.longitude.size, step)
            binlon = np.repeat(np.arange(0, lon_n), repeats = step)
            binlon = np.append(binlon, np.repeat(np.NaN, lon_rem))
            lat_n, lat_rem = divmod(self.array.latitude.size, step)
            binlat = np.repeat(np.arange(0, lat_n), repeats = step)
            binlat = np.append(binlat, np.repeat(np.NaN, lat_rem))
        
        # Concatenate as strings to a group variable
        combined = np.char.array(binlat)[:, None] + np.char.array(binlon)[None, :] # Numpy broadcasting behaviour. Kind of an outer product
        combined = xr.DataArray(combined, [self.array.coords['latitude'], self.array.coords['longitude']], name = 'latlongroup')
        
        # Compute grouped values. This stacks the dimensions to one spatial and one temporal
        f = getattr(self.array.groupby(combined), method)
        grouped = f('stacked_latitude_longitude', keep_attrs=True)        
        
        # Compute new coordinates, and construct a spatial multiindex with lats and lons for each group
        newlat = self.array.latitude.to_pandas().groupby(np.char.array(binlat)).mean()
        newlon = self.array.longitude.to_pandas().groupby(np.char.array(binlon)).mean()
        newlatlon = pd.MultiIndex.from_tuples(list(itertools.product(newlat, newlon)), names=('latitude', 'longitude'))
        
        # Prepare the coordinates of stack dimension and replace the internal array
        grouped['latlongroup'] = newlatlon        
        self.array = grouped.unstack('latlongroup')
        #self.array = xr.DataArray(grouped, coords = [grouped.time, newlatlon], dims = ['time','latlon']).unstack('latlon')
        self.array.attrs['spacemethod'] = '_'.join([str(step), 'cells', method]) if not by_degree else '_'.join([str(step), 'degrees', method])
        
    
    def aggregatetime(self, freq = 'w' , method = 'mean'):
        """
        Uses the pandas frequency indicators. Method can be mean, min, max, std
        Completely lazy when loading was lazy
        """
        tempattr = self.array.attrs
        f = getattr(self.array.resample(time = freq), method) # timestamp can be set with label = 'right'
        self.array = f('time') # Removes the attributes.
        tempattr['timemethod'] = '_'.join([freq,method])
        self.array.attrs = tempattr
        # To access days of week: self.array.time.dt.timeofday
        # Also possible is self.array.time.dt.floor('D')
    
    def savechanges(self):
        """
        Uses the timemethod and spaceaverage attributes (when created) to create a new variable name. 
        Then writes the dask array to this file. Clears the internal array. Can give a awrning of all NaN slices encountered during writing.
        Possibly: add option for experiment name?
        """
        timemethod = getattr(self.array, 'timemethod') if hasattr(self.array, 'timemethod') else ''
        spacemethod = getattr(self.array, 'spacemethod') if hasattr(self.array, 'spacemethod') else ''
        self.variable = self.variable + timemethod + spacemethod
        self.filepath = ''.join([self.storagedir, self.variable, ".nc"])
        # invoke the computation (if loading was lazy) and writing
        self.array.to_netcdf(self.filepath)
        delattr(self, 'array')
    # Define a detrend method? Look at the Toy weather data documentation of xarray.
        
# I want to keep the real advanced operations seperate, like PCA, clustering,

# I want to make a class for heatevent. within the definitionmethods I want to have an option to convert the array to a flat dataframe. remove NA'. Then based on previously clustering I can also aggregate by spatial grouping?

# Later on I want to make an experiment class. Combining surface observations on a certain aggregation, with 
    
self = SurfaceObservations(variable = 'rr')
self.load(tmax = "1960-03-02", lazy = True)
self.aggregatetime(freq = 'w', method = 'mean')
self.savechanges()
self.load(lazy = False)
self.aggregatespace(step = 5, method = 'max', by_degree=False)

self.savechanges()
#self.load()

class HeatEvent(object):
    
    def __init__(self, observations):
        self.obs = observations
        