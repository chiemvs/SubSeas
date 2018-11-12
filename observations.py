import os
import numpy as np
import xarray as xr
import pandas as pd
import itertools

class SurfaceObservations(object):
    
    def __init__(self, alias, **kwds):
        """
        Sets the base E-OBS variable alias. And sets the base storage directory. 
        Additionally you can supply 'timemethod', 'spacemethod', 'tmin' and 
        'tmax' if you want to load a pre-existing adapted file.
        """
        self.basevar = alias
        self.basedir = "/nobackup/users/straaten/E-OBS/"
        for key in kwds.keys():
            setattr(self, key, kwds[key])
    
    def construct_name(self):
        """
        Name and filepath are based on the base variable and the relevant attributes (if present)
        Uses the timemethod and spacemethod attributes
        """
        keys = ['tmin','tmax','timemethod','spacemethod']
        attrs = [ getattr(self,x) if hasattr(self, x) else '' for x in keys]
        attrs.insert(0,self.basevar) # Prepend with alias
        self.name = ''.join(attrs)
        self.filepath = ''.join([self.basedir, self.name, ".nc"])
    
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

        
    def load(self, lazychunk = None, tmin = None, tmax = None, llcrnr = (None, None), rucrnr = (None,None)):
        """
        Loads the netcdf (possibly delimited by maxtime and corners). Corners need to be given as tuples (lat,lon)
        Creates new attributes: the array with the data, units, and coordinates, and possibly a timemethod.
        If a lazychunk is given (e.g. dictionary: {'time': 3650}) a dask array will be loaded
        """
        
        self.construct_name()
        
        # Check file existence. Download if base variable
        if not os.path.isfile(self.filepath):
            if (self.name == self.basevar):
                self.downloadraw()
            else:
                raise FileNotFoundError("File for non-basevariable not found")
        
        if lazychunk is None:
            full = xr.open_dataarray(self.filepath)
        else:
            full = xr.open_dataarray(self.filepath,  chunks= lazychunk)
        
        # Full range if no timelimits were given
        if tmin is None:
            tmin = pd.Series(full.coords['time'].values).min()
        if tmax is None:
            tmax = pd.Series(full.coords['time'].values).max()
        
        freq = pd.infer_freq(full.coords['time'].values)
        self.array = full.sel(time = pd.date_range(tmin, tmax, freq = freq), longitude = slice(llcrnr[1], rucrnr[1]), latitude = slice(llcrnr[0], rucrnr[0])) # slice gives an inexact lookup, everything within the range
        
        self.tmin = tmin[0:10] if isinstance(tmin, str) else tmin.strftime('%Y-%m-%d')
        self.tmax = tmax[0:10] if isinstance(tmax, str) else tmax.strftime('%Y-%m-%d')

    def aggregatespace(self, step, method = 'mean', by_degree = False):
        """
        Regular lat lon or gridbox aggregation by creating new single coordinate which is used for grouping.
        In the case of degree grouping the groups might not contain an equal number of cells.
        Completely lazy when loading is set to lazy
        """
        
        # Check if already loaded.
        if not hasattr(self, 'array'):
            self.load(lazychunk = {'latitude': 50, 'longitude': 50})
        
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
        self.spacemethod = '_'.join([str(step), 'cells', method]) if not by_degree else '_'.join([str(step), 'degrees', method])
        
    
    def aggregatetime(self, freq = 'w' , method = 'mean'):
        """
        Uses the pandas frequency indicators. Method can be mean, min, max, std
        Completely lazy when loading is lazy
        """
        if not hasattr(self, 'array'):
            self.load(lazychunk = {'time': 365})
        
        f = getattr(self.array.resample(time = freq), method) # timestamp can be set with label = 'right'
        self.array = f('time', keep_attrs=True) 
        self.timemethod = '_'.join([freq,method])

        # To access days of week: self.array.time.dt.timeofday
        # Also possible is self.array.time.dt.floor('D')
    
    def savechanges(self):
        """
        Calls new name creation. Then writes the dask array to this file. 
        Can give a awrning of all NaN slices encountered during writing.
        Possibly: add option for experiment name?. Clear internal array?
        """
        self.construct_name()
        # invoke the computation (if loading was lazy) and writing
        self.array.to_netcdf(self.filepath)
        #delattr(self, 'array')
    
    # Define a detrend method? Look at the Toy weather data documentation of xarray.
        
# I want to keep the real advanced operations seperate, like PCA, clustering,

# I want to make a class for heatevent. within the definitionmethods I want to have an option to convert the array to a flat dataframe. remove NA'. Then based on previously clustering I can also aggregate by spatial grouping?

# Later on I want to make an experiment class. Combining surface observations on a certain aggregation, with 
    
#test1 = SurfaceObservations(alias = 'rr')
#test1.load(lazychunk = {'latitude': 50, 'longitude': 50}, tmax = '1960-01-01')
#test1.aggregatetime() # 1.32 sec

test2 = SurfaceObservations(alias = 'rr')
#test2.load(lazychunk = {'time':3650}, tmax = '1990-01-01')
#test2.aggregatetime(freq = 'M') # Small chuncks seem to work pretty well. For .sum() the nan are not correctly processed (become 0)
#test2.savechanges()

#del test2

#recover = SurfaceObservations(alias = 'rr', **{'tmin' : '1950-01-01', 'tmax' : '1960-01-01', 'timemethod' : 'w_mean'})
#recover.load()


class HeatEvent(object):
    
    def __init__(self, observations):
        self.obs = observations
    
    
        