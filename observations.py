#!/usr/bin/env python3
import os
import sys
import numpy as np
import xarray as xr
import pandas as pd
import dask.array as da
import warnings
from helper_functions import agg_space, agg_time, monthtoseasonlookup, vcorrcoef3D, vcorrcoef2D

obs_netcdf_encoding = {'rr': {'dtype': 'int16', 'scale_factor': 0.05, '_FillValue': -32767},
                   'tx': {'dtype': 'int16', 'scale_factor': 0.002, '_FillValue': -32767},
                   'tg': {'dtype': 'int16', 'scale_factor': 0.002, '_FillValue': -32767},
                   'rr-anom': {'dtype': 'int16', 'scale_factor': 0.05, '_FillValue': -32767},
                   'tx-anom': {'dtype': 'int16', 'scale_factor': 0.002, '_FillValue': -32767},
                   'tg-anom': {'dtype': 'int16', 'scale_factor': 0.002, '_FillValue': -32767},
                   'rr-pop': {'dtype': 'int16', 'scale_factor': 0.0001, '_FillValue': -32767},
                   'rr-pod': {'dtype': 'int16', 'scale_factor': 0.0001, '_FillValue': -32767},
                   'time': {'dtype': 'int64'},
                   'latitude': {'dtype': 'float32'},
                   'longitude': {'dtype': 'float32'},
                   'doy': {'dtype': 'int16'},
                   'number': {'dtype': 'int16'},
                   'clustid': {'dtype': 'int16', '_FillValue': -32767},
                   'dissim_threshold': {'dtype':'float32'}} #{'dtype':'int16', 'scale_factor':0.0001,  '_FillValue': -32767}

class SurfaceObservations(object):
    
    def __init__(self, basevar, **kwds):
        """
        Sets the base E-OBS variable alias. And sets the base storage directory. 
        Additionally you can supply 'timemethod', 'spacemethod', 'tmin' and 
        'tmax', or the comple filename if you want to load a pre-existing adapted file.
        """
        self.basevar = basevar
        self.basedir = "/nobackup/users/straaten/E-OBS/"
        for key in kwds.keys():
            setattr(self, key, kwds[key])
    
    def construct_name(self, force = False):
        """
        Name and filepath are based on the base variable (or new variable) and the relevant attributes (if present).
        If name is already present as an attribute given upon initialization: reverse engineer the other attributes
        """
        keys = ['var','tmin','tmax','timemethod','spacemethod']
        if hasattr(self, 'name') and (not force):
            values = self.name.split(sep = '_')
            for key in keys[1:]:
                setattr(self, key, values[keys.index(key)])
        else:
            values = [ getattr(self,key) for key in keys[1:] if hasattr(self, key)]
            try:
                values.insert(0,'-'.join([self.basevar,self.newvar]))
            except AttributeError:
                values.insert(0,self.basevar)
            self.name = '_'.join(values)
        
        self.filepath = ''.join([self.basedir, self.name, ".nc"])
    
    def downloadraw(self):
        """
        Downloads daily highres observations on regular 0.25 degree lat-lon grid.
        """
        import urllib3
        
        urls = {"tg":"https://www.ecad.eu/download/ensembles/data/Grid_0.25deg_reg_ensemble/tg_ens_mean_0.25deg_reg_v19.0e.nc",
        "tn":"https://www.ecad.eu/download/ensembles/data/Grid_0.25deg_reg_ensemble/tn_ens_mean_0.25deg_reg_v19.0e.nc",
        "tx":"https://www.ecad.eu/download/ensembles/data/Grid_0.25deg_reg_ensemble/tx_ens_mean_0.25deg_reg_v19.0e.nc",
        "rr":"https://www.ecad.eu/download/ensembles/data/Grid_0.25deg_reg_ensemble/rr_ens_mean_0.25deg_reg_v19.0e.nc",
        "pp":"https://www.ecad.eu/download/ensembles/data/Grid_0.25deg_reg_ensemble/pp_ens_mean_0.25deg_reg_v19.0e.nc"}
        
        if not os.path.isfile(self.filepath):
            f = open(self.filepath, 'wb')
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
            
    def load(self, lazychunk = None, tmin = None, tmax = None, llcrnr = (None, None), rucrnr = (None,None)):
        """
        Loads the netcdf (possibly delimited by maxtime and corners). Corners need to be given as tuples (lat,lon)
        Creates new attributes: the array with the data, units, and coordinates, and possibly a timemethod.
        If a lazychunk is given (e.g. dictionary: {'time': 3650}) a dask array will be loaded
        """
        
        self.construct_name() # To take account of attributes or even name given at initialization.
        
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
        
        if not hasattr(self, 'timemethod'): # Attribute is only missing if it has not been given upon initialization.
            self.timemethod = '1D'
        if not hasattr(self, 'spacemethod'):
            self.spacemethod = '0.25-degrees'
        
        self.tmin = pd.Series(self.array.time).min().strftime('%Y-%m-%d')
        self.tmax = pd.Series(self.array.time).max().strftime('%Y-%m-%d')

    def minfilter(self, season, n_min_per_seas = 50):
        """
        Requires on average a minimum amount of daily observations per season of interest (91 days). Assigns NaN all year round when this is not the case.
        """
        seasonindex = self.array.time.dt.season == season
        n_seasons = np.diff(seasonindex).sum() / 2 # Number of transitions between season and non-season divided by 2 
        seasononly = self.array.sel(time = seasonindex)
        
        self.array = self.array.where(seasononly.count('time') >= n_seasons * n_min_per_seas, np.nan)
        
    def aggregatespace(self, step, method = 'mean', by_degree = False, skipna = False, rolling = False):
        """
        Regular lat lon or gridbox aggregation by creating new single coordinate which is used for grouping.
        In the case of degree grouping the groups might not contain an equal number of cells.
        Completely lazy when supplied array is lazy.
        """
        # Check if already loaded.
        if not hasattr(self, 'array'):
            self.load(lazychunk = {'latitude': 50, 'longitude': 50})
        
        self.array, self.spacemethod = agg_space(array = self.array, 
                                                 orlats = self.array.latitude.load(),
                                                 orlons = self.array.longitude.load(),
                                                 step = step, method = method, by_degree = by_degree, skipna = skipna, rolling = rolling)
    
    def aggregatetime(self, freq = '7D' , method = 'mean', rolling = False):
        """
        Uses the pandas frequency indicators. Method can be mean, min, max, std
        Completely lazy when loading is lazy.
        """
        
        if not hasattr(self, 'array'):
            self.load(lazychunk = {'time': 365})
        
        self.array, self.timemethod = agg_time(array = self.array, freq = freq, method = method, rolling = rolling)

        # To access days of week: self.array.time.dt.timeofday
        # Also possible is self.array.time.dt.floor('D')
    

    
    def savechanges(self):
        """
        Calls new name creation after writing tmin and tmax as attributes. Then writes the dask array to this file. 
        Can give a warning of all NaN slices encountered during writing.
        """
        self.construct_name(force = True)
        # invoke the computation (if loading was lazy) and writing
        particular_encoding = {key : obs_netcdf_encoding[key] for key in self.array.to_dataset().keys()} 
        self.array.to_netcdf(self.filepath, encoding = particular_encoding)
        #delattr(self, 'array')
    
    # Define a detrend method? Look at the Toy weather data documentation of xarray. Scipy has a detrend method.


class Climatology(object):
    
    def __init__(self, alias, **kwds):
        """
        Class to contain local climatologies. Construct from observations or load a previously constructed one
        """
        self.var = alias
        self.basedir = "/nobackup/users/straaten/climatology/"
        for key in kwds.keys():
            setattr(self, key, kwds[key])
    
    def construct_name(self, force = False):
        """
        Name and filepath are based on the base variable (or new variable) and the relevant attributes (if present).
        The second position in the name is just 'clim' for easy recognition in the directories.
        """
        keys = ['var','tmin','tmax','timemethod','spacemethod', 'daysbefore', 'daysafter', 'climmethod']
        if hasattr(self, 'name') and (not force):
            values = self.name.split(sep = '_')
            values.pop(1)
            for key in keys:
                setattr(self, key, values[keys.index(key)])
        else:
            values = [ str(getattr(self,key)) for key in keys if hasattr(self, key)]
            values.insert(1,'clim') # at the second spot, after the basevar
            self.name = '_'.join(values)
        
        self.filepath = ''.join([self.basedir, self.name, ".nc"])
        
    def localclim(self, obs = None, tmin = None, tmax = None, timemethod = None, spacemethod = None, daysbefore = 0, daysafter = 0, mean = True, quant = None, n_draws = None, daily_obs = None):
        """
        Load a local clim if one with corresponding basevar, tmin, tmax and method is found, or if name given at initialization is found.
        Construct local climatological distribution within a rolling window, but with pooled years. 
        Extracts mean (giving probabilities if you have a binary variable). Or a random number of draws if these are given.
        It can also compute a quantile on a continuous variables. Returns fields of this for all supplied day-of-year (doy) and space.
        Daysbefore and daysafter are inclusive.
        For non-daily aggregations the procedure is the same, as the climatology needs to be still derived from daily values. 
        Therefore the amount of aggregated dats is inferred from the timemethod frequency. And also if rolling needs to take place. (rolling obs gives rolling time agg in the doy windows)
        For event-based variables: the obs should have the transformation already. 
        The daily obs should have the original continuous variable (only space aggregated) and is transformed here 
        """
        
        keys = ['tmin','tmax','timemethod','spacemethod', 'daysbefore', 'daysafter']
        for key in keys:
            try:
                setattr(self, key, getattr(obs, key))
            except AttributeError:
                setattr(self, key, locals()[key])

        if mean:
            self.climmethod = 'mean'
        elif quant is not None:
            self.climmethod = 'q' + str(quant)
        else:
            self.climmethod = 'rand'
        
        # Overwrites possible nonsense attributes
        self.construct_name(force = False)
                
        try:
            self.clim = xr.open_dataarray(self.filepath)
            print('climatology directly loaded')
        except OSError:
            self.ndayagg = int(pd.date_range('2000-01-01','2000-12-31', freq = obs.timemethod.split('-')[0]).to_series().diff().dt.days.mode())
        
            if quant is not None:
                from helper_functions import nanquantile
            
            if (self.ndayagg > 1):
                freq, rolling, method = obs.timemethod.split('-')
                doygroups = daily_obs.array.groupby('time.dayofyear')
            else:
                doygroups = obs.array.groupby('time.dayofyear')
            
            doygroups = {str(key):value for key,value in doygroups} # Change to string
            maxday = 366
            results = []
            for doy in doygroups.keys():        
                doy = int(doy)
                # for aggregated values the day of year is the first day of the period. 
                window = np.arange(doy - daysbefore, doy + daysafter + self.ndayagg, dtype = 'int64')
                # small corrections for overshooting into previous or next year.
                window[ window < 1 ] += maxday
                window[ window > maxday ] -= maxday
                
                complete = xr.concat([doygroups[str(key)] for key in window if str(key) in doygroups.keys()], dim = 'time')
                # Call for the same aggregation on the daily complete by slicing up each past sequence of our doy-window, and progressively removing them from the complete set, such that the minimum time can be used for each slice. However, the aggregation is always non-rolling even though obs might be.
                if (self.ndayagg > 1):
                    aggregated_slices = []
                    while len(complete.time) > 0:
                        slice_tmin = complete.time.min().values
                        print(slice_tmin)
                        slice_arr = complete.sortby('time').sel(time = slice(str(slice_tmin), str(slice_tmin + np.timedelta64(len(window), 'D')))) # Soft searching method. Based on the minimum found in the set. Does not crash if certain doys are less present (like 366)
                        if len(slice_arr.time) >= self.ndayagg:
                            aggregated_slices.append(agg_time(array = slice_arr, freq = freq, method = method, ndayagg = self.ndayagg, rolling = (rolling == 'roll'))[0]) # [0] for only the returned array. not the timemethod             
                        complete = complete.drop(slice_arr.time.values, dim = 'time') # remove so new minimum can be found.
                    complete = xr.concat(aggregated_slices, dim = 'time')
                    
                    # Possible classification in the aggregated slices if the supplied obs was transformed, and this is not the same (pre-aggregation) transformation already present on the daily-obs.
                    try:
                        obsnewvar = getattr(obs, 'newvar')
                        try:
                            dailyobsnewvar = getattr(daily_obs, 'newvar')
                            if obsnewvar != dailyobsnewvar:
                                raise AttributeError
                        except AttributeError:
                            tempobs = SurfaceObservations(basevar= obs.basevar) # Assign to the class for alteraton
                            tempobs.array = complete
                            getattr(EventClassification(tempobs),getattr(obs, 'newvar'))() # Get the classifier capable of transforming the class
                            complete = tempobs.array
                    except AttributeError:
                        pass
    
                if mean:
                    reduced = complete.mean('time', keep_attrs = True)
                elif quant is not None:
                    # nanquantile method based on sorting. So not compatible with xarray. Therefore feed values and restore coordinates.
                    reduced = xr.DataArray(data = nanquantile(array = complete.values, q = quant),
                                            coords = [complete.coords['latitude'], complete.coords['longitude']],
                                            dims = ['latitude','longitude'], name = self.var)
                    reduced.attrs = complete.attrs
                    reduced.attrs['quantile'] = quant
                else:
                    # Random sampling with replacement if not enough fields in the complete array
                    try:
                        draws = complete.sel(time = np.random.choice(complete.time, size = n_draws, replace = False))
                    except ValueError:
                        draws = complete.sel(time = np.random.choice(complete.time, size = n_draws, replace = True))
                    # Assign a number dimension to the draws.
                    reduced = xr.DataArray(data = draws, coords = [np.arange(n_draws), complete.coords['latitude'], complete.coords['longitude']], dims = ['number','latitude','longitude'], name = self.var)
                
                # Setting a minimum on the amount of observations that went into the mean and quantile computation, and report the number of locations that went to NaN
                number_nan = reduced.isnull().sum(['latitude','longitude']).values
                reduced = reduced.where(complete.count('time') >= 10, np.nan)
                number_nan = reduced.isnull().sum(['latitude','longitude']).values - number_nan
                
                
                print('computed clim of', doy, ', removed', number_nan, 'locations with < 10 obs.')
                reduced.coords['doy'] = doy
                results.append(reduced)
            
            self.clim = xr.concat(results, dim = 'doy')
    
    
    def savelocalclim(self):
        
        self.construct_name(force = True)
        particular_encoding = {key : obs_netcdf_encoding[key] for key in self.clim.to_dataset().variables.keys()} 
        self.clim.to_netcdf(self.filepath, encoding = particular_encoding)
        
        
class EventClassification(object):
    
    def __init__(self, obs, obs_dask = None, **kwds):
        """
        Aimed at on-the-grid classification of continuous variables. By modifying the array
        attribute of observation or forecast objects. Normal array for (memory efficient) grouping 
        and fast explicit computation
        If the dataset is potentially large then one can supply a version withy dask array
        dask array for the anomaly computation which probably will not fit into memory.
        """
        self.obs = obs
        if obs_dask is not None:
            self.obsd = obs_dask
            
        self.old_units = getattr(obs.array, 'units')
            
        for key in kwds.keys():
            setattr(self, key, kwds[key])
    
    def pop(self, threshold = 0.3, inplace = True):
        """
        Method to change rainfall accumulation arrays to a boolean variable of whether it rains or not. Unit is mm / day.
        Because we still like to incorporate np.NaN on the unobserved areas, the array has to be floats of 0 and 1
        """
        if hasattr(self, 'obsd'):
            data = da.where(da.isnan(self.obsd.array.data), self.obsd.array.data, self.obsd.array.data > threshold)
        else:
            data = np.where(np.isnan(self.obs.array), self.obs.array, self.obs.array.data > threshold)
        
        result = xr.DataArray(data = data, coords = self.obs.array.coords, dims= self.obs.array.dims,
                                attrs = {'long_name':'probability_of_precipitation', 'threshold_mm_day':threshold, 'units':self.old_units, 'new_units':''},
                                name = '-'.join([self.obs.basevar, 'pop']))
        
        if inplace:
            self.obs.array = result
            self.obs.newvar = 'pop'
        else:
            return(result)

    def pod(self, threshold = 0.3, inplace = True):
        """
        Method to change rainfall accumulation arrays to a boolean variable of whether it is dry or not. Unit is mm / day.
        Because we still like to incorporate np.NaN on the unobserved areas, the array has to be floats of 0 and 1
        """
        if hasattr(self, 'obsd'):
            data = da.where(da.isnan(self.obsd.array.data), self.obsd.array.data, self.obsd.array.data < threshold)
        else:
            data = np.where(np.isnan(self.obs.array), self.obs.array, self.obs.array.data < threshold)
        
        result = xr.DataArray(data = data, coords = self.obs.array.coords, dims= self.obs.array.dims,
                                attrs = {'long_name':'probability_of_dryness', 'threshold_mm_day':threshold, 'units':self.old_units, 'new_units':''},
                                name = '-'.join([self.obs.basevar, 'pod']))
        
        if inplace:
            self.obs.array = result
            self.obs.newvar = 'pod'
        else:
            return(result)
            
    def stefanon2012(self):
        """
        First a climatology. Then local exceedence, 4 consecutive days, 60% in sliding square of 3.75 degree.
        Not the connection yet of events within neighbouring squares yet.
        """
        # check for variable is tmax, construct quantile climatology by calling the Climatology class. get.
    
    def zscheischler2013(self):
        """
        time/space greedy fill of local exceedence of climatological quantile.
        """
    
    def anom(self, inplace = True):
        """
        Substracts the climatological value for the associated day of year from the observations.
        This leads to anomalies when the climatological mean was taken. Exceedances become sorted by doy.
        The climatology object needs to have been supplied at initialization.
        """
        if not hasattr(self, 'climatology'):
            raise AttributeError('provide climatology at initialization please')
        
        if not (self.obs.array.units == self.climatology.clim.units):
            raise ValueError('supplied object and climatology do not have the same units')
        
        if hasattr(self, 'obsd'):
            doygroups = self.obsd.array.groupby('time.dayofyear')
        else:
            doygroups = self.obs.array.groupby('time.dayofyear')
        
        def subtraction(inputarray):
            doy = int(np.unique(inputarray['time.dayofyear']))
            if hasattr(inputarray, 'leadtime') and hasattr(self.climatology.clim, 'leadtime'):
                climatology = self.climatology.clim.sel(doy = doy, leadtime = inputarray['leadtime'], drop = True)
            else:
                warnings.warn('anomaly subtraction not leadtime dependent')
                climatology = self.climatology.clim.sel(doy = doy, drop = True)
            return(inputarray - climatology)
        
        result = xr.DataArray(data = doygroups.apply(subtraction), coords = self.obs.array.coords, dims= self.obs.array.dims,
                                attrs = {'long_name':'-'.join([self.obs.basevar, 'anomalies']), 'units':self.old_units, 'new_units':self.old_units},
                                name = '-'.join([self.obs.basevar, 'anom']))
        
        if inplace:
            self.obs.array = result
            self.obs.newvar = 'anom'
        else:
            return(result)


class Clustering(object):
    
    def __init__(self, season, **kwds):
        """
        Class to compute, contain and load hierarchal clusters of gridded observations.
        Possibility to supply a name here for direct loading of a saved hierarchal cluster object.
        """
        self.season = season
        self.basedir = '/nobackup/users/straaten/clusters/'
        self.lags = list(range(-20,21)) # possible lags used in the association between gridpoints
        self.dissim_thresholds = [0,0.005,0.01,0.025,0.05,0.1,0.2,0.3,0.4,0.5,1] # Average dissimilarity thresholds to cut the tree, into n clusters
        for key in kwds.keys():
            setattr(self, key, kwds[key])
            
    def compute_cormat(self, obs, mapmemory = True, vectorize_lags = False):
        """
        Method to compute the 1D triangular correlation matrix between all cells in the loaded array of the supplied observation class
        The metric is the maximum correlation in a set of lagged timeseries between each pair of cells
        Cells without any observations are removed, otherwise there is no minimum amount.
        Only on timeseries within the season.
        TODO: add parallel computation option
        """
        # Storing information from the obs.
        keys = ['basevar','tmin','tmax']
        for key in keys:
            setattr(self,key,getattr(obs,key))        
        
        # Select season, flatten the space dimension to ncells and store the spatial index for later reconstruction
        subsetaxis = obs.array.time.dt.season == self.season
        subset = obs.array.sel(time = subsetaxis).stack({'latlon':['latitude','longitude']}).dropna('latlon','all')
        self.spaceindex = subset.coords['latlon'].copy()
        
        # Created multiple shifted version of the timeseries by shifting and reindexing
        # because we are dealing with a seasonal subset so a non-continuous time axis
        ori_timeaxis = subset.coords['time'].copy()
        laggedmatrices = [None] * len(self.lags)
        for lag in self.lags:
            lag_timeaxis = ori_timeaxis - pd.Timedelta(str(lag) + 'D')
            subset.coords['time'] = lag_timeaxis # Assign the shifted timeaxis
            laggedmatrices[self.lags.index(lag)] = subset.reindex_like(ori_timeaxis).values
        if mapmemory:
            f = np.memmap('/tmp/memmapped.dat', dtype=np.float32, mode='w+', shape=(len(self.lags),) + laggedmatrices[0].shape)
            np.stack(laggedmatrices, axis = 0, out = f)
            laggedmatrices = f
        else:
            laggedmatrices = np.stack(laggedmatrices, axis = 0) # The lagged matrices object might get too large for memory (in that case memory mapping?)
        
        # Initialization of the square maximum correlation matrix, only the upper triangle, excluding the diagonal
        ncells = len(self.spaceindex)
        n_triangular = int((ncells**2 - ncells)/2)
        self.maxcormat = np.full((n_triangular), -1.0, dtype = 'float32') # Initialize at the worst possible similarity
                
        # triangular loop. Write to the 1D maxcormat matrix
        # This is all sequential code. Both finding the maxima in non vectorized lags and the 
        firstemptycell = 0
        for i in range(ncells - 1):
            colindices = slice(i+1,ncells) # Not the correlation with itself.
            writelength = ncells - 1 - i
            cormatindices = slice(firstemptycell, firstemptycell + writelength) # write indices to the 1D triangular matrix.
            cellseries = laggedmatrices[self.lags.index(0),:,i]
            if vectorize_lags:
                self.maxcormat[cormatindices] = vcorrcoef3D(laggedmatrices[:,:,colindices], cellseries)
            else:
                for lag in self.lags:
                    lagcor = vcorrcoef2D(laggedmatrices[self.lags.index(lag),:,colindices], cellseries)
                    self.maxcormat[cormatindices] = np.maximum(self.maxcormat[cormatindices], lagcor)
            firstemptycell += writelength
            print('computed', str(writelength), 'links for cell', str(i + 1), 'of', str(ncells))
    
    def hierarchal_clustering(self):
        """
        Hierarchal clustering with a lagged correlation based distance metric (0 is perfect correlated, 2 is perfectly anticorrelated)
        The triangular matrix is supplied to the scipy algorithm linking the clusters with the least average distance.
        The z matrix is stored for dendogram inspection. Cuts of the tree (based on dissimilarity heights) are put into the array format.
        """
        import scipy.cluster.hierarchy as sch
        self.Z = sch.linkage(y = 1 - self.maxcormat, method = 'average')
        #sch.dendrogram(self.Z, truncate_mode='lastp')
        
        # Extract the clusters from the tree at the average dissimilarity thresholds
        ids = np.squeeze(sch.cut_tree(self.Z, height=self.dissim_thresholds))
        self.clusters = xr.DataArray(ids, dims = ['latlon','dissim_threshold'], coords = {'dissim_threshold':self.dissim_thresholds,'latlon':self.spaceindex}, name = 'clustid').unstack('latlon')
        
    def get_clusters_at(self, level = 0):
        """
        Method to get the georeferenced cluster array at a certain dissimilarity level
        """
        self.construct_name(force = False)
        if not hasattr(self,'clusters'):
            self.clusters = xr.open_dataarray(self.filepath)
        try:
            return(self.clusters.sel(dissim_threshold = level, method = 'nearest', tolerance = 1e-7))
        except KeyError:
            raise KeyError('desired level not present in the loaded clustering dataset')
    
    def construct_name(self, force = False):
        """
        Name and filepath are based on the base variable (or new variable) and the relevant attributes (if present).
        The second position in the name is just 'clim' for easy recognition in the directories.
        """
        keys = ['basevar','season','tmin','tmax']
        if hasattr(self, 'name') and (not force):
            values = self.name.split(sep = '_')
            for key in keys:
                setattr(self, key, values[keys.index(key)])
        else:
            values = [ str(getattr(self,key)) for key in keys if hasattr(self, key)]
            self.name = '_'.join(values)
        
        self.filepath = ''.join([self.basedir, self.name, ".nc"])
    
    def save_clusters(self):
        self.construct_name(force = True)
        particular_encoding = {key : obs_netcdf_encoding[key] for key in self.clusters.to_dataset().variables.keys()} 
        self.clusters.to_netcdf(self.filepath, encoding = particular_encoding)
       
#obs = SurfaceObservations('tg')
#obs.load(tmin = '1980-01-01', tmax = '1985-01-01', llcrnr= (64,40))
#obs.minfilter(season='JJA', n_min_per_seas=88)
#self = Clustering(season = 'JJA', **{'name':'tx_JJA_1980-01-01_1985-01-01'})
#self.save_clusters()
#self.compute_cormat(obs = obs, mapmemory=False, vectorize_lags=True)
#self.hierarchal_clustering()
#self.save_clusters()
