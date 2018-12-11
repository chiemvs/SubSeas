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
        Downloads daily highres observations on regular 0.25 degree lat-lon grid.
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
            u = http.request('GET', urls[self.basevar], preload_content = False)
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
        
        self.construct_name() # To take account of attributes given at initialization.
        
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

    def aggregatespace(self, step, method = 'mean', by_degree = False):
        """
        Regular lat lon or gridbox aggregation by creating new single coordinate which is used for grouping.
        In the case of degree grouping the groups might not contain an equal number of cells.
        Completely lazy when supplied array is lazy.
        """
        from helper_functions import agg_space
        
        # Check if already loaded.
        if not hasattr(self, 'array'):
            self.load(lazychunk = {'latitude': 50, 'longitude': 50})
        
        self.array, self.spacemethod = agg_space(array = self.array, 
                                                 orlats = self.array.latitude.load(),
                                                 orlons = self.array.longitude.load(),
                                                 step = step, method = method, by_degree = by_degree)
    
    def aggregatetime(self, freq = 'w' , method = 'mean'):
        """
        Uses the pandas frequency indicators. Method can be mean, min, max, std
        Completely lazy when loading is lazy.
        """
        from helper_functions import agg_time
        
        if not hasattr(self, 'array'):
            self.load(lazychunk = {'time': 365})
        
        self.array, self.timemethod = agg_time(array = self.array, freq = freq, method = method)

        # To access days of week: self.array.time.dt.timeofday
        # Also possible is self.array.time.dt.floor('D')
    
    def savechanges(self):
        """
        Calls new name creation after writing tmin and tmax as attributes. Then writes the dask array to this file. 
        Can give a warning of all NaN slices encountered during writing.
        Possibly: add option for experiment name?. Clear internal array?
        """
        self.tmin = pd.Series(self.array.time).min().strftime('%Y-%m-%d')
        self.tmax = pd.Series(self.array.time).max().strftime('%Y-%m-%d')
        self.construct_name()
        # invoke the computation (if loading was lazy) and writing
        self.array.to_netcdf(self.filepath)
        #delattr(self, 'array')
    
    # Define a detrend method? Look at the Toy weather data documentation of xarray.
        
# I want to keep the real advanced operations seperate, like PCA, clustering,

# I want to make a class for heatevent. within the definitionmethods I want to have an option to convert the array to a flat dataframe. remove NA'. Then based on previously clustering I can also aggregate by spatial grouping?

# Later on I want to make an experiment class. Combining surface observations on a certain aggregation, with 

test1 = SurfaceObservations(alias = 'rr')
test1.load(tmax = '1980-02-03') #lazychunk = {'latitude': 50, 'longitude': 50}  
#test1.aggregatespace(step = 5)
#test2 = SurfaceObservations(alias = 'rr')
#test2.load(tmax = '1950-01-03', lazychunk = {'time':300})
    
# To matrix with (observations, locations), where NaN is filtered
#dsflat = xr.Dataset({'rr' :test1.array.stack(latlon = ['latitude', 'longitude'])})
#nona = dsflat.dropna(dim = 'latlon')
#obsmat = nona.rr.values

#U, s, Vt = np.linalg.svd(obsmat, full_matrices=False)
# s are the singular values. Square them for the eigenvalues of np.dot(obsmat.T, obsmat). Already ordered.
# eigenvectors are the columns in V = Vt.T (just like regular eigenvector matrix) or in the rows of Vt

# plot the first eigenvector
#pc1 = xr.DataArray(Vt[0,:], coords = [nona.latlon], dims = ['latlon'])
#import matplotlib.pyplot as plt
#axes = plt.axes()
#pc1.unstack('latlon').plot(ax = axes)
#plt.savefig('test')


class EventClassification(object):
    
    def __init__(self, obs_array, obs_daskarray = None):
        """
        Supply the observations xarray: Normal array for (memory efficient) grouping and fast explicit climatology computation
        If the dataset is potentially large to load completely then also supply a
        dask array for the anomaly computation which probably will not fit into memory.
        """
        self.obs = obs_array
        if obs_daskarray is not None:
            self.obsd = obs_daskarray
    
    def localclim(self, daysbefore = 0, daysafter = 0, mean = True, quant = None):
        """
        Construct local climatological distribution within a rolling window, but with pooled years. 
        Extracts mean (for anomaly computation) or for instance a quantile. Returns fields of this for all supplied time and space.
        Daysbefore and daysafter are inclusive. 
        """ 
        if quant is not None:
            from helper_functions import nanquantile
        doygroups = self.obs.groupby('time.dayofyear')
        doygroups = {str(key):value for key,value in doygroups} # Change to string
        maxday = 366
        results = []
        for doy in doygroups.keys():        
            doy = int(doy)
            window = np.arange(doy - daysbefore, doy + daysafter + 1, dtype = 'int64')
            # small corrections for overshooting into previous or next year.
            window[ window < 1 ] += maxday
            window[ window > maxday ] -= maxday
            
            complete = xr.concat([doygroups[str(key)] for key in window if str(key) in doygroups.keys()], dim = 'time')

            if mean:
                reduced = complete.mean('time')
            elif quant is not None:
                # nanquantile method based on sorting. So not compatible with xarray. Therefore feed values and restore coordinates.
                reduced = xr.DataArray(data = nanquantile(array = complete.values, q = quant),
                                        coords = [complete.coords['latitude'], complete.coords['longitude']],
                                        dims = ['latitude','longitude'])
                reduced.attrs['quantile'] = quant
            else:
                raise ValueError('Provide a quantile if mean is set to False')
            
            print('computed clim of', doy)
            reduced.coords['doy'] = doy
            results.append(reduced)
        
        self.clim = xr.concat(results, dim = 'doy')
    
    def climexceedance(self):
        """
        Substracts the climatological value for the associated day of year from the observations.
        This leads to anomalies when the climatological mean was taken. Exceedances become sorted by doy.
        """
        if hasattr(self, 'obsd'):
            doygroups = self.obsd.groupby('time.dayofyear')
        else:
            doygroups = self.obs.groupby('time.dayofyear')
        results = []
        for doy, fields in doygroups:
            results.append(fields - self.clim.sel(doy = doy))
            print('computed exceedance of', doy)
        self.exceedance = xr.concat(results, dim = 'time')
    
#self = EventClassification(obs_array=test1.array, obs_daskarray=test2.array)
#self.localclim(daysbefore=2, daysafter=2, mean = False, quant = 0.95)

