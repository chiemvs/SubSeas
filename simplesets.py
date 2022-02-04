import numpy as np
import xarray as xr
import pandas as pd

from pathlib import Path
from datetime import datetime

from forecasts import Forecast, ModelClimatology, for_netcdf_encoding
from observations import SurfaceObservations, EventClassification, Climatology
from comparison import ForecastToObsAlignment

import scipy.cluster.vq as vq
from scipy.signal import detrend

from sklearn.linear_model import LinearRegression

def merge_soilm(directory: Path, variables: list = ['swvl1','swvl2','swvl3'], weights: list = [7,21,72], newvar: str = 'swvl13'):
    """
    Iterates through for(hind)casts present in the directory
    Computes a weighted average of the listed variables
    Writes a new variables to the file
    """
    paths = directory.glob('*_processed.nc') # Both prefix for_ and hin_
    weights = xr.DataArray(weights, dims = ('dummy',)) # Preparation of weighted mean outside loop
    for path in paths: 
        print(path)
        ds = xr.open_dataset(path)
        if newvar in ds.variables:
            print(f'{newvar} already present')
            ds.close()
        else:
            temp = xr.concat([ds[var] for var in variables], dim = 'dummy') # becomes zeroth axis
            temp_weighted = temp.weighted(weights) 
            new = ds.assign({newvar:temp_weighted.mean('dummy')})
            new[newvar].attrs = new[variables[0]].attrs # Make sure that units are retained
            new.load()
            ds.close() # Close before rewriting (bit ugly, netCDF4 could do appending to file)
            particular_encoding = {key : for_netcdf_encoding[key] for key in new.variables.keys()}
            new.to_netcdf(path, encoding = particular_encoding)
            print(f'{newvar} computed and file rewritten')

"""
Get daily forecasts at a specific lead time. 

Make them into anomalies with the model climatologies (trend is dealt with in statistical modeling)

Perhaps to this only upon processing. Perhaps not so useful to have a whole extra directory with anomalies

What is time aggregation of predictands?

Certainly no regime information on mean Z300. Then better work with frequencies of daily regimes within some period 
"""

# Creation of anomalies for all leadtimes
# Model drift in the mean is accounted for by constructing anomalies this way

varib_unit = {'swvl13':'m**3 m**-3', 'z':'m**2 s**-2'}

def make_anomalies(f: Forecast, var: str, highresmodelclim: ModelClimatology, loadkwargs: dict = {}):
    """ Inplace transformation to anomalies """
    f.load(var, **loadkwargs)
    f.array.attrs.update({'units':varib_unit[var]}) # Avoid the current absence of units
    if not hasattr(highresmodelclim, 'clim'):
        highresmodelclim.local_clim()
        highresmodelclim.clim.attrs.update({'units':varib_unit[var]}) # Avoid the current absence of units
    method = getattr(EventClassification(obs = f, **{'climatology':highresmodelclim}), 'anom')
    method(inplace = True)

def lead_time_1_z300(ndays = 1) -> xr.DataArray:
    """
    attempt to get the daily 'analysis' dataset (time,space) of Z300 anomalies. For later regime computations
    Truest to the analysis in this data is leadtime 1 day (12 UTC), control member
    But to enlarge dataset the amount of days after initialization to extract can be chosen
    """
    variable = 'z'
    forbasedir = '/nobackup/users/straaten/EXT_extra/'
    cycle = '45r1'
    forecast_paths = (Path(forbasedir) / cycle).glob('*_processed.nc')  # Both for- and hindcasts
    listofforecasts = [Forecast(indate = path.name.split('_')[1], prefix = path.name[:4], cycle = cycle, basedir = forbasedir) for path in forecast_paths]

    # Reference model climatology
    modelclim = ModelClimatology(cycle='45r1', variable = variable, **{'name':f'{variable}_45r1_1998-06-07_2019-08-31_1D_1.5-degrees_5_5_mean'}) # Name for loading

    # Pre-allocate an array for anomalies
    example = listofforecasts[0]
    example.load(variable)
    lats = example.array.coords['latitude']
    lons = example.array.coords['longitude']

    data = np.full((len(listofforecasts)*3, len(lats), len(lons)), np.nan, dtype = np.float32)

    # Extract!
    timestamps = pd.DatetimeIndex([]) # Possibly non-unique
    for forc in listofforecasts:
        mintime = pd.Timestamp(forc.indate)
        maxtime = mintime + pd.Timedelta(ndays - 1, 'D')
        timestamps = timestamps.append(pd.date_range(mintime, maxtime)) # dates to be added are determined
        make_anomalies(f = forc, var = variable, highresmodelclim = modelclim, loadkwargs = {'n_members':1, 'tmax':maxtime.strftime('%Y-%m-%d')}) # To limit the amount of data loaded into memory
        data[(len(timestamps)-ndays):len(timestamps),:,:] = forc.array.sel(number = 0) # number 0 is the control member

    almost_analysis = xr.DataArray(data, name = f'{variable}-anom', dims = ('time','latitude','longitude'), coords = {'time':timestamps,'latitude':lats,'longitude':lons}, attrs = {'units':varib_unit[variable]})
    # Duplicate dates are possible
    return almost_analysis

def era5_z300_resample() -> xr.DataArray:
    """
    Spatially resampling the ERA5 anomalie data at the VU
    Should be loaded such that it resembles the domain of EXT_extra hindcasts
    Resampling is not by averaging.
    Should be okay for z300 which is a relatively smooth field.
    """
    z300_anom_path = '/scistor/ivm/jsn295/processed/z300_nhnorm.anom.nc'
    
    sample_path = '/scistor/ivm/jsn295/backup/EXT_extra/45r1/hin_2005-07-09_processed.nc'
    anoms = xr.open_dataarray(z300_anom_path)
    example = xr.open_dataset(sample_path)['z'][0,:,:,0]
    example = example.sortby('latitude') # Reversing the order, forecasts are stored with decreasing latitudes.
    return anoms.reindex_like(example, method = 'nearest')

def subset_and_prepare(arr: xr.DataArray, months: int = None, buffermonths: int = None, detrend_method: str = None) -> xr.Dataset:
    """
    Selection of the temporal slice for which regimes are prepared. 
    either a single month (integer), or multiple months jointly (list of integers)
    If using hindcasts you probably want to exclude march:
    Counts array([3, 4, 5, 6, 7, 8]), array([ 42, 189, 189, 210, 189, 106]))
    possible to detrend the slice linearly (pooling data) with either scipy or sklearn
    Perhaps I want to add filtering. See e.g. https://github.com/fujiisoup/xr-scipy/blob/master/xrscipy/signal/filters.py
    Also possible to supply buffermonths (this leads to the returning of a larger second set, whose extra months do not count for the detrending and component extraction but will be projected)
    """
    assert detrend_method in [None,'scipy','sklearn','sklearnpool'], 'Only scipy and sklearn or None are valid detrending options'

    if isinstance(months, int):
        months = [months]
    anom = arr[arr.time.dt.month.isin(months),...].copy()

    if not detrend_method is None:
        if detrend_method == 'scipy':
            anom.values = detrend(anom.values, axis = 0)
            final = anom.to_dataset()
        elif detrend_method == 'sklearn':
            time_axis = anom.time.to_pandas().index.to_julian_date().values[:,np.newaxis] # (samples, features)
            stacked = anom.stack({'latlon':['latitude','longitude']})
            lr = LinearRegression(n_jobs = 15)
            lr.fit(X = time_axis, y = stacked.values)
            trend = lr.predict(X = time_axis)
            stacked.values = stacked.values - trend
            final = xr.Dataset({'coef':('latlon',lr.coef_.squeeze()), 'intercept':('latlon',lr.intercept_.squeeze())}, coords = {'latlon':stacked.coords['latlon']})
            final = final.assign({anom.name:stacked}).unstack('latlon')
        elif detrend_method == 'sklearnpool':
            fully_stacked = anom.to_series()  # (nsamples,)
            time_axis = fully_stacked.index.get_level_values('time').to_julian_date().values[:,np.newaxis] # (samples, features)
            lr = LinearRegression(n_jobs = 15)
            lr.fit(X = time_axis, y = fully_stacked.values)
            trend = lr.predict(time_axis)
            detrended = (fully_stacked - trend).to_xarray() 
            detrended.attrs.update(anom.attrs)
            final = xr.Dataset({'coef':(('latitude','longitude'), np.full(anom.shape[1:], lr.coef_[0])), 'intercept':(('latitude','longitude'), np.full(anom.shape[1:], lr.intercept_))}, coords = {'latitude':anom.coords['latitude'],'longitude':anom.coords['longitude']})
            final = final.assign({detrended.name:detrended})
    else:
        final = anom.to_dataset()

    if not buffermonths is None:
        assert detrend_method != 'scipy', 'buffermonths is not compatible with scipy detrending as no coefficients are stored'
        if isinstance(buffermonths, int):
            buffermonths = [buffermonths]
        extended_anom = arr[arr.time.dt.month.isin(months + buffermonths),...].copy()
        if detrend_method.startswith('sklearn'):
            extended_time_axis = extended_anom.time.to_pandas().index.to_julian_date().values[:,np.newaxis,np.newaxis] # (samples, features)
            extended_trend = extended_time_axis * final['coef'].values[np.newaxis,...] + final['intercept'].values[np.newaxis,...]
            extended_anom.values = extended_anom.values - extended_trend
        return final, extended_anom
    else:
        return final
        
def extract_components(arr: xr.DataArray, extended_arr: xr.DataArray = None, ncomps: int = 10, nclusters: int = 4, seed = None):
    """
    arr needs to be subset to the correct subset, also already detrended if desired
    returns 2 datasets
    - dataset with result of EOF (eigvectors, eigvalues, eof projections timeseries of arr into vectors) 
    - dataset with cluster results (EOF centroids, and composite mean arr fields of the clusters)
    If an extended array is supplied (optional), it will only be used for projection
    """
    data = arr.values.reshape((arr.shape[0],-1)) # Raveling the spatial dimension, keeping only the values
    U, s, Vt = np.linalg.svd(data, full_matrices=False) # export OPENBLAS_NUM_THREADS=25 put upfront.
    
    component_coords = pd.Index(list(range(ncomps)), dtype = np.int64)

    eigvals = (s**2)[:ncomps]
    eigvectors = Vt[:ncomps,:].reshape((ncomps,) + arr.shape[1:])
    if extended_arr is None:
        eof = arr.coords.to_dataset()
    else:
        eof = extended_arr.coords.to_dataset()
    eof.coords['component'] = component_coords
    eof['eigvectors'] = (('component','latitude','longitude'),eigvectors)
    eof['eigvalues'] = (('component',),eigvals)
    # Projection of the 3D data into 10 component timeseries
    projection = data @ Vt.T[:,:ncomps]
    if not extended_arr is None:
        extended_projection = extended_arr.values.reshape((extended_arr.shape[0],-1)) @ Vt.T[:,:ncomps]
        eof['projection'] = (('time','component'),extended_projection)
    else:
        eof['projection'] = (('time','component'),projection)
    eof['projection'].attrs['units'] = ''

    #k-means clustering
    k_coords = pd.Index(list(range(nclusters)), dtype = np.int64)
    centroids, assignments = vq.kmeans2(projection, k = nclusters, seed = seed, minit = '++') 
    composites = [data[assignments == cluster,...].mean(axis = 0) for cluster in k_coords] # Composite of the anomalies
    composites = np.concatenate(composites, axis = 0).reshape((nclusters,) + arr.shape[1:]) # new zeroth axis is clusters
    clusters = arr.coords.to_dataset().drop_dims('time')
    clusters.coords['clustid'] = k_coords
    clusters.coords['component'] = component_coords
    clusters['z_comp'] = (('clustid','latitude','longitude'), composites)
    clusters['centroid'] = (('clustid','component'), centroids)

    return eof, clusters

class RegimeAssigner(object):
    def __init__(self, at_KNMI: bool = True, max_distance: float = None):
        """
        Assignment itself needs access to multiple sources
        needed for preparation of the model fields
        - Forecasts (indexed with decreasing lats) 
        - z model climatology (lead time dependent) for anomalies (indexed with decreasing lats)
        - coefficients and intercepts for detrending (indexed with increasing lats)
        needed for the distance and classification:
        - eigenvectors for projection (1 field to 10 timeseries) (indexed with increasing lats)
        - centroids for distance computation, (1 value per centroid / cluster)
        """
        self.max_distance = max_distance
        self.timeagg = 1 # initial, gets overwritten by frequency_in_window
        self.name ='z300_1D_months_5678_sklearnpool_detrended2'
        #self.name ='z300_1D_months_5678_sklearnpool_detrended_ncl5_s1'
        if at_KNMI:
            self.simplepredspath = Path('/nobackup/users/straaten/predsets')
            self.for_basedir = Path('/nobackup/users/straaten/EXT_extra')
            self.highresmodelclim = ModelClimatology(cycle = '45r1', variable = 'z', name = 'z_45r1_1998-06-07_2019-08-31_1D_1.5-degrees_5_5_mean', basedir = '/nobackup/users/straaten/modelclimatology/')
        else:
            self.simplepredspath = Path('/scistor/ivm/jsn295/backup/predsets')
            self.for_basedir = Path('/scistor/ivm/jsn295/backup/EXT_extra')
            self.highresmodelclim = ModelClimatology(cycle = '45r1', variable = 'z', name = 'z_45r1_1998-06-07_2019-08-31_1D_1.5-degrees_5_5_mean', basedir = '/scistor/ivm/jsn295/backup/modelclimatology/')
        self.highresmodelclim.local_clim()
        
        try:
            self.coefset = xr.open_dataset(self.simplepredspath / f'{self.name}_coefs.nc') 
        except OSError:
            print('coefficient set not found, assuming detrending before assignment will no be needed later')
        self.eigvectors = xr.open_dataset(self.simplepredspath / f'{self.name}_patterns.nc')['eigvectors']
        self.centroids = xr.open_dataset(self.simplepredspath / f'{self.name}_clusters.nc')['centroid']


    def anomalize(self, f: Forecast):
        """
        Preparation of z300 to make into anomalies
        Remove seasonal cycle by using the modelclimatology (lead time dependent)
        """
        # f and highresmodelclim have to be loaded already
        method = getattr(EventClassification(obs = f, **{'climatology':self.highresmodelclim}), 'anom')
        method(inplace = True) # still indexed with decreasing latitude
        f.array = f.array.sortby('latitude') # Correct the sortorder to work well with eigvector and coefs

    def detrend(self, f: Forecast):
        """
        Preparation of z300 to make into anomalies
        Remove the linear (thermal expansion) trend with coefficients determined on ERA5
        Do this inplace
        """
        assert not 'det' in f.array.name, f'variable {f.array.name} seems already detrended, can only happen once'
        # just a manual a + bx?
        timeaxis = f.array['time'].to_pandas().index.to_julian_date().values
        trend = self.coefset['intercept'].values[:,:,np.newaxis] + self.coefset['coef'].values[:,:,np.newaxis] * timeaxis # (latitude,longitude,time) # with increasing latitude
        trend_with_coords = f.array.coords.to_dataset().drop('number')
        trend_with_coords = trend_with_coords.assign({f.array.name:xr.Variable(dims = ('latitude','longitude','time'), data = trend)})
        # array (time, latitude, longitude, number)
        oldattrs = f.array.attrs # procedure below does not keep units etc.
        f.array = f.array - trend_with_coords[f.array.name] # xarray will handle the remaining coordinate, and ordering of the dims
        f.array.attrs = oldattrs 
        f.array.name = f'{f.array.name}-det'
        f.array.attrs['long_name'] = f'{f.array.attrs["long_name"]}-detrended'

    def project(self, f: Forecast, unstack: bool = False) -> xr.DataArray:
        """
        Needs eigenvectors (latitude, logitude) to project
        (time, latitude, longitude, number) into either:
        (n_comps,time,number) if unstack, else 
        ([time,number],n_comps) for easy further computation
        """
        for_flattened = f.array.stack({'timenumber': ['time','number'],'latlon':['latitude','longitude']}) # flattening time/number into one because of ease of 2D*2D matrix multiplication
        eig_flattened = self.eigvectors.stack({'latlon':['latitude','longitude']}) # stacks into last dimension but we need it first to be the inner dimension for the matrix multiplication, so transpose
        projected = np.matmul(for_flattened.values, eig_flattened.values.T) # very important here that the stacked sort-order is correct, as there is no formal coordinate handling.
        projected = xr.DataArray(projected, dims = ('timenumber','component'), coords = {'timenumber':for_flattened.coords['timenumber'],'component':eig_flattened.coords['component']})
        if unstack:
            return projected.unstack('timenumber')
        else:
            return projected

    def assign(self, component_timeseries: xr.DataArray) -> tuple:
        """
        Component timeseries (n_samples, n_comps)
        where n_samples is pontentially stacked (not a pure timeseries)
        centroids (n_clusters, n_comps)
        returns 1) distances to each centroid  (n_samples, n_clusters)
        and 2) the assigned (integer) clusters (n_samples,)
        Also possible to supply a maximum allowed distance to the nearest cluster
        if no distance is below that, we assign NO cluster (id = -1) for that sample
        """
        # vq is an option but does not return all distances.
        # Says it requires unit variance in the documentation. Obviously not the case.
        # clusters, distance_to_nearest = vq.vq(obs = component_timeseries, code_book = centroids)

        minus = component_timeseries - self.centroids.T # (N,n_comps,n_clusters)
        squared = minus**2
        distances = np.sqrt(squared.sum(axis = 1)) # Shows that vq also does euclidian distance, but returns only the nearest
        clusters = distances.argmin(axis =1) # These are column indices, will only correspond to cluster id if cluster id coords are sorted [0,n_clusters - 1]
        clusters.values = self.centroids.coords['clustid'].values[clusters.values] # Make use of array indexing of the small cluster id coordinate array
        if not self.max_distance is None:
            all_exceeding = (distances > self.max_distance).all(axis = 1)
            clusters = clusters.where(~all_exceeding, other = -1) # assign a -1 when distances all are too large
        distances.name = 'forecast'
        clusters.name = 'forecast'
        return distances.unstack(), clusters.unstack() # After this unstacking we may want to restore leadtime.

    def find_forecasts(self, cycle = '45r1') -> list:
        """
        Find forecasts and prepare initialization arguments for each (returned as list)
        No initialization here and attachment of forecasts to the self
        because when loaded (lots of data) this is hard to parallelize
        """
        self.cycle = cycle
        present_forecasts = (self.for_basedir / cycle ).glob('*processed.nc')
        initkwargs = []
        for path in present_forecasts:
            fortype, date, _ = path.name.split('_')  # fortype is whether forecast or hindcast
            initkwargs.append(dict(indate = date, prefix = f'{fortype}_', cycle = cycle, basedir = f'{self.for_basedir}/'))
        return initkwargs

    def process_forecast(self, forecast_init_kwargs: dict, detrend: bool = True, return_distances_too: bool = True):
        """
        called once per forecast
        Returns dataframes: one for the regime ids
        and optionally one for the distances
        Should in the end be indexed similar to 'aligned data'
        So [time, clustid, leadtime, forecast, observation] with the extra level 'number'
        Clustid here can be location (or in this case the regime id)
        """
        f = Forecast(**forecast_init_kwargs)
        f.load(variable = 'z', n_members = 11)
        self.anomalize(f = f) # returns nothing
        if detrend:
            self.detrend(f = f) # returns nothing
        eofs = self.project(f = f, unstack = False)
        distances, regimeids = self.assign(component_timeseries = eofs)
        # Add leadtimes again as index variable, needs to happen in a dataset
        leadtimes = np.array(f.array.coords['leadtime'].values.astype('timedelta64[D]'), dtype = 'int8') # To integer (days)
        regimeids = regimeids.to_dataset()
        regimeids = regimeids.assign({'leadtime':('time', leadtimes)})
        regimeids = regimeids.set_coords('leadtime').to_dataframe()
        regimeids.set_index('leadtime', inplace = True, append = True) 
        if return_distances_too:
            distances = distances.to_dataset()
            distances = distances.assign({'leadtime':('time', leadtimes)})
            distances = distances.set_coords('leadtime').to_dataframe()
            distances.set_index('leadtime', inplace = True, append = True) 
            
            return regimeids.unstack('number'), distances.unstack('number')
        else:
            return regimeids.unstack('number')

    def add_observation(self, forecast_df: pd.DataFrame, how = 'left') -> pd.DataFrame:
        """
        Reads the projected observed fields and assigns them 
        Then adds those as an extra column to the dataframe
        Tries to infer whether to add the distances or the clusters
        Handy get the forecast and corresponding observation in the same frame
        """
        projected_obs = xr.open_dataset(self.simplepredspath / f'{self.name}_patterns.nc')['projection']

        obs_distances, obs_regimes = self.assign(projected_obs) # Also obeys the self.maxdistance
        if 'clustid' in forecast_df.index.names:
            print('inferred through the clustid-index that a distance forecast_df is supplied, adding observed distance per cluster')
            to_merge = obs_distances.to_dataframe() 

        else:
            print('inferred that a regimeid forecast_df is supplied, adding observed regimes')
            to_merge = obs_regimes.to_dataframe()
        
        to_merge.columns = pd.MultiIndex.from_tuples([('observation',0)], names = [None,'number']) # Adding a fake number dimension
        result = forecast_df.join(to_merge, how = how) # Joining on the index
        return result

    def associate_all(self):
        """
        Finds all forecasts and associates them sequentually
        """
        list_of_initkwargs = self.find_forecasts()
        all_assigned, all_distances = [], []
        for initkwargs in list_of_initkwargs:
            regime, distance = self.process_forecast(initkwargs, detrend = True, return_distances_too = True)
            print(f'processed {initkwargs["indate"]}')
            all_assigned.append(regime)
            all_distances.append(distance)
        all_assigned = pd.concat(all_assigned, axis = 0)
        all_assigned = self.add_observation(all_assigned, how = 'inner') # Stricter join, we don't want to add observed nans in april
        all_distances = pd.concat(all_distances, axis = 0)
        all_distances = self.add_observation(all_distances, how = 'inner') # Stricter join, we don't want to add observed nans in april
        return all_assigned.sort_index(), all_distances.sort_index()

    def two_dimensional_rolling(self, init_dates: pd.DatetimeIndex, timeagg: int, array: xr.DataArray, pre_allocated_frame: pd.DataFrame, reduce_number_dim: bool):
        """
        For time aggregation, we need to start per forecast initialization date
        and simultaneously move along the time and leadtime axes
        This is easier in an array, which we can index (non-orthogonally) along both axes
        Array is either (time, leadtime) for obs or (time, leadtime, number) for forecasts
        The pre_allocated_frame is just np.nan indexed with all the present (time,leadtime) combinations
        Writes entries with frequencies per clustid inplace (columns of the pre_allocated_frame)
        """
        for firstday in init_dates:
            timeslice = slice(firstday, firstday + pd.Timedelta('45D'), None) # Might point into autumn, where some leadtimes are not available
            subset = array.sel(time = timeslice) # Therefore slicing and check availability with shape:
            print(f'value sequence starting {firstday} has len {subset.shape[0]} in MJJA')
            leadtime_index_slice = xr.DataArray(np.arange(subset.shape[0]), dims = ('leadtime',)) # Coordinates for vectorized indexing of the available data.
            start_to_end = subset[leadtime_index_slice, leadtime_index_slice,...] # Vectorized (non-orthogonal indexing) in 2D
            temp = [] # We'll be creating a new clustid axis
            for clustid in pre_allocated_frame.columns.get_level_values('clustid').unique():
                if ('number' in start_to_end.dims) and reduce_number_dim: # Dealing with forecasts and aggregating over the members
                    count = (start_to_end == clustid).sum('number') # counting over the members
                else: # Dealing with observations, or wanting the forecast count per member
                    count = start_to_end == clustid
                # Now the temporal aggregation by counting over the time window.
                total_window_occurrence = count.rolling({'leadtime':timeagg}).sum() # leadtime indexed or leadtime and number indexed
                temp.append(total_window_occurrence)
            comb = xr.concat(temp, dim = pre_allocated_frame.columns.get_level_values('clustid').unique()) # Added as axis = 0
            # Left stamping
            comb = comb.assign_coords({'time':comb.time - pd.Timedelta(str(timeagg - 1) + 'D')})
            comb = comb.assign_coords({'leadtime':comb.leadtime - (timeagg - 1)})
            comb = comb.isel(leadtime = slice(timeagg - 1, None))
            if 'number' in comb.dims:
                stacked = comb.stack({'stackdim':['clustid','number']}) # Makes a multiindex (as last axis) just like the columns of the dataframe
                pre_allocated_frame.loc[list(zip(comb.time.values, comb.leadtime.values)),:] = stacked.values # Setting values
            else:
                pre_allocated_frame.loc[list(zip(comb.time.values, comb.leadtime.values)),:] = comb.values.T # Setting values, while getting clustid from zeroth axis to the first (i.e. columns)

    def frequency_in_window(self, assigned: pd.DataFrame, nday_window : int, per_member: bool = False) -> pd.DataFrame:
        """
        Time aggregation, by means of frequency of regime occurrence in a window.
        Takes in a dataframe with the daily assignments,
        Will call the 2D rolling on the forecasts (and observations if present)
        with windowsize in days. Results will be left stamped, meaning trailing leadtime/day combinations
        with not enough values following them, will disappear. 
        """
        self.timeagg = nday_window
        clustids = pd.Index(np.unique(assigned['forecast']), name = 'clustid')
        assert ('time' in assigned.index.names) and ('leadtime' in assigned.index.names), 'needs to have an indexed dataframe'

        forecasts = assigned['forecast'].stack('number')
        firstdays = forecasts.loc[(slice(None),1,0)].index.get_level_values('time') # leadtime 1, and just the control member
        forecasts = forecasts.to_xarray()
        nmembers = len(forecasts.coords['number'])
        if per_member:
            forecast_count_in_window = pd.DataFrame(np.nan, index = assigned.index, columns = pd.MultiIndex.from_product([clustids,forecasts.number.values], names = ['clustid','number'])) # Pre-allocation , ([time, leadtime],[clustid, number])
        else:
            forecast_count_in_window = pd.DataFrame(np.nan, index = assigned.index, columns = clustids) # Pre-allocation , ([time, leadtime],clustid)
        self.two_dimensional_rolling(init_dates = firstdays , timeagg = nday_window, array = forecasts, pre_allocated_frame = forecast_count_in_window, reduce_number_dim = not per_member) # Modifies the frame inplace
        # Set extra index level and Normalize the counts to frequencies, but adding 1 to all classes, so we don't get zero's
        if per_member:
            forecast_count_in_window.columns = pd.MultiIndex.from_tuples([('forecast',) + key for key in forecast_count_in_window.columns], names = [None,'clustid','number']) # Already a two-layer index
            n_counts = nday_window + len(clustids) 
        else:
            forecast_count_in_window.columns = pd.MultiIndex.from_product([['forecast'],forecast_count_in_window.columns])
            n_counts = (nmembers * nday_window) + len(clustids) 
        forecast_count_in_window = forecast_count_in_window + 1
        probability_in_window = forecast_count_in_window / n_counts 
        probability_in_window = probability_in_window.dropna(axis = 0, how = 'all') # Slicing of the trailing NaN due to pre-allocation
        assert np.allclose(probability_in_window.sum(axis = 1), 1.0) or np.allclose(probability_in_window.sum(axis = 1), 1.0 *nmembers) , 'distribution should sum up to one, check missing forecast values'

        if 'observation' in assigned.columns:
            observations = assigned['observation'].iloc[:,0].to_xarray()
            observed_freq_in_window = pd.DataFrame(np.nan, index = assigned.index, columns = clustids) # Pre-allocation , ([time, leadtime],clustid)
            self.two_dimensional_rolling(init_dates = firstdays , timeagg = nday_window, array = observations, pre_allocated_frame = observed_freq_in_window, reduce_number_dim = not per_member) # Modifies the frame inplace
            # Build the required multiindex
            if per_member:
                observed_freq_in_window.columns = pd.MultiIndex.from_product([['observation'],observed_freq_in_window.columns,[0]], names = [None,'clustid','number'])
            else:
                observed_freq_in_window.columns = pd.MultiIndex.from_product([['observation'],observed_freq_in_window.columns])
            # Normalize the counts to frequencies
            observed_freq_in_window = observed_freq_in_window / nday_window # nmembers * ndays is total
            observed_freq_in_window = observed_freq_in_window.dropna(axis = 0, how = 'all') # Slicing of the trailing NaN due to pre-allocation
            assert np.allclose(observed_freq_in_window.sum(axis = 1), 1.0), 'distribution should sum up to one, check missing observation values (e.g. april)'
            joined = probability_in_window.join(observed_freq_in_window, how = 'left')
        else:
            joined = probability_in_window

        return joined

    def save(self, df: pd.DataFrame, expname: str = 'paper3-4-regimes', what: str = 'ids' ,basepath: Path = Path('/nobackup_1/users/straaten/match')):
        """
        Ready the dataframe to match previous formats
        Also with a booksfile. We'll be always overwriting
        """
        if 'clustid' in df.columns.names:
            df = df.stack('clustid')
        if not 'clustid' in df.index.names:
            df['clustid'] = -99 # Just a placeholder to match the formats
        if not 'number' in df.columns.names:
            df.columns = pd.MultiIndex.from_product([df.columns, ['']], names = list(df.columns.names) + ['number'])

        name = f'{expname}_z-anom_JJA_{self.cycle}_{self.timeagg}D-frequency_{what}'
        outfilepath = basepath / f'{name}.h5'
        books_name = f'books_{name}.csv'
        books_path = basepath / books_name

        books = pd.DataFrame({'file':[outfilepath],
                              'tmax':[df.index.get_level_values('time').max().strftime('%Y-%m-%d')],
                              'tmin':[df.index.get_level_values('time').min().strftime('%Y-%m-%d')],
                              'unit':[''],
                              'write_date':[datetime.now().strftime('%Y-%m-%d_%H:%M:%S')]})

        df = df.reset_index(inplace = False)
        df.to_hdf(outfilepath, key = 'intermediate', mode = 'w', format = 'table')
        
        with open(books_path, 'w') as f:
            books.to_csv(f, header=True, index = False)
        print('written out', outfilepath)
        return books_name


if __name__ == '__main__':
    """
    Creation of the weighted soil moisture 
    """
    #merge_soilm(directory = Path('/nobackup/users/straaten/EXT_extra/45r1')) 
    
    """
    Regime predictors I: Computation of the clusters in EOF-transformed observed Z300
    """
    #ndays = 3
    #arr = lead_time_1_z300(ndays = ndays)
    #arr = arr.sortby('time') # Potentially, .drop_duplicates('time')
    #arr.to_netcdf(f'/nobackup/users/straaten/predsets/z300-leadtime-dep-anom_{ndays}D_control.nc')

    #arr = xr.open_dataarray(f'/nobackup/users/straaten/predsets/z300-leadtime-dep-anom_{ndays}D_control.nc')

    #arr = era5_z300_resample()
    #monthsets = [list(range(5,9))]
    ####monthsets = [[i] for i in range(5,9)]
    #detrend_method = 'sklearnpool' #None
    #seed = 2
    #nclusters = 5
    #basedir = Path('/scistor/ivm/jsn295/backup/predsets')
    #for months in monthsets:
    #    monthscoord = "".join([str(i) for i in months])
    #    basename = f'z300_1D_months_{monthscoord}_{detrend_method}_detrended_ncl{nclusters}_s{seed}'
    #    dataset, dataarray = subset_and_prepare(arr = arr, months = months, buffermonths = 9, detrend_method = detrend_method)
    #    eofs, clusters = extract_components(arr = dataset[arr.name], extended_arr = dataarray, ncomps = 10, nclusters = nclusters, seed = seed) # Following Ferranti 2015 in ncomps and nclusters (4, or 5 from zampieri 2017) #export OPENBLAS_NUM_THREADS=25 put upfront.
    #    eofs.to_netcdf(basedir / f'{basename}_patterns.nc')
    #    clusters.to_netcdf(basedir / f'{basename}_clusters.nc')
    #    if 'coef' in dataset:
    #        dataset.drop(arr.name).to_netcdf(basedir / f'{basename}_coefs.nc')

    """
    Regime predictors II: assignment of forecast Z300 (initializtion,leadtime,members) to found clusters
    """
    #ra = RegimeAssigner(at_KNMI = True, max_distance = 60000) # close to the median distance to all
    #assigned_ids, distances = ra.associate_all()
    #bookfile1 = ra.save(assigned_ids, what = 'ids', expname = 'paper3-4-4regimes')
    #bookfile2 = ra.save(distances, what = 'distances', expname = 'paper3-4-4regimes')
    #assigned_ids = pd.read_hdf('/nobackup_1/users/straaten/match/paper3-4-4regimes_z-anom_JJA_45r1_1D-frequency_ids.h5').set_index(['time','leadtime'])
    #nday_window = 21
    #assigned_ids_agg = ra.frequency_in_window(assigned_ids, nday_window = nday_window, per_member = False) # Time aggregation
    #bookfile3 = ra.save(assigned_ids_agg, what = 'ids', expname = 'paper3-4-4regimes')
    #assigned_ids_agg_pm = ra.frequency_in_window(assigned_ids, nday_window = nday_window, per_member = True) # Time aggregation per member
    #bookfile4 = ra.save(assigned_ids_agg_pm, what = 'ids_per_member', expname = 'paper3-4-4regimes')
    
    """
    Spatiotemporal mean anomaly simplesets
    Should be doable by creating ERA5 anom observations with a certain clustering and timeagg
    Writing out the clusterarray too, and then loading + matching with the modelclim for that variable.
    - downside = sampling and re-indexing. currently one season, and a mixture of leadtimes. No continuous availability of forecasts with a certain leadtime
    - upside = produces a set with 'observations' included. Could be helpful for investigating biases in simulated variables themselves
    block-cluster made in swvl-simple.nc 
    """
    combinations = pd.DataFrame({'obs':['swvl13-anom_1981-01-01_2019-09-30_1D_1.5-degrees','swvl4-anom_1981-01-01_2019-09-30_1D_1.5-degrees','z300-anom_1979-01-01_2019-12-31_1D_1.5-degrees','sst-anom_1979-01-01_2019-12-31_1D_1.5-degrees'],
        'block':['swvl-local','swvl-local','swvl-local','sst-local'],
        'modelclim':['swvl13_45r1_1998-06-07_2019-08-31_1D_1.5-degrees_5_5_mean','swvl4_45r1_1998-06-07_2019-08-31_1D_1.5-degrees_5_5_mean','z_45r1_1998-06-07_2019-08-31_1D_1.5-degrees_5_5_mean','sst_45r1_1998-06-07_2019-08-31_1D_1.5-degrees_5_5_mean'],
        'expname':['paper3-3-simple','paper3-3-simple','paper3-3-simple','paper3-3-simple'],
        }, index = pd.Index(['swvl13','swvl4', 'z','sst'], name = 'basevar'))

    def create_blocks(basevar, timeagg: int):
        obs = SurfaceObservations(basevar = basevar, basedir = '/nobackup/users/straaten/ERA5/', name = combinations.loc[basevar,'obs']) # Override normal E-OBS directory
        obs.load(tmin = '1998-06-07', tmax = '2019-08-31')
        obs.aggregatespace(clustername = combinations.loc[basevar,'block'], level = 1)
        obs.aggregatetime(freq = f'{timeagg}D', method = 'mean', rolling = True)
        obs.newvar = 'anom' # was already an anomalie
        obs.savechanges()

        # loading of the constructed observations, streamlines the clusterarray for matching
        obs = SurfaceObservations(basevar = basevar, basedir = '/nobackup/users/straaten/ERA5/', name = f'{basevar}-anom_1998-06-07_2019-08-31_{timeagg}D-roll-mean_1-{combinations.loc[basevar,"block"]}-mean') # Override normal E-OBS directory
        obs.load()
        obs.newvar = 'anom' # As this happened at the VU already
        
        modelclim = ModelClimatology(cycle='45r1', variable = obs.basevar, **{'name':combinations.loc[basevar,'modelclim']}) # Name for loading
        modelclim.local_clim()
        newvarkwargs={'climatology':modelclim}
        if basevar != 'sst':
            loadkwargs = {'llcrnr':(30,None),'rucrnr':(None,42)} # Limiting the domain a bit.
        else:
            loadkwargs = {}

        alignment = ForecastToObsAlignment(season = 'JJA', observations=obs, cycle='45r1', n_members = 11, **{'expname':combinations.loc[basevar,'expname']}) # Season subsets the obs
        alignment.match_and_write(newvariable = True, 
                                  newvarkwargs = newvarkwargs,
                                  loadkwargs = loadkwargs,
                                  matchtime = True, 
                                  matchspace= True)

    def create_clim(basevar, timeagg: int):
        """
        Making climatologies of the newly created block-average observations
        """
        obs = SurfaceObservations(basevar = basevar, basedir = '/nobackup/users/straaten/ERA5/', name = f'{basevar}-anom_1998-06-07_2019-08-31_{timeagg}D-roll-mean_1-{combinations.loc[basevar,"block"]}-mean') # Override normal E-OBS directory
        obs.load()
        clim = Climatology(f'{basevar}-anom')
        clim.localclim(obs = obs, daysbefore = 15, daysafter = 15, mean = False, quant = 0.75)
        clim.savelocalclim()

    #timeagg = 21
    #for var in combinations.index[::-1]:
    #    create_blocks(var, timeagg = timeagg)
    #    create_clim(var, timeagg = timeagg)

    """
    Global Mean Surface temperature prediction (only trend)
    Downloaded from the climate explorer
    Can be used to filter other predictors (that are only associated because of trend in predictor and target)
    """
    #import netCDF4 as nc
    #from sklearn.linear_model import LinearRegression
    #gmst_path = '/nobackup/users/straaten/predsets/tg_monthly_global_mean_surface.nc'
    #ds = nc.Dataset(gmst_path, mode = 'r') # automatic time decoding via xarray does not work.
    #timeunits = ds['time'].getncattr('units') # months since 1880-01
    #time = ds['time'][:] # first value is 0, so start in 
    #timeaxis = pd.date_range(start = '1880-01-15', freq = 'M', periods = len(time))
    #ts = pd.Series(ds['Ta'][:], index = timeaxis)
    #ts = ts.loc[slice('1970-01-01', None,None)].dropna(how = 'any') # Linear regime
    #regressor = LinearRegression()
    #regressor.fit(X = ts.index.to_julian_date().values.reshape((len(ts),1)), y = ts)
    #
    ## Highres 1D trend prediction (of the 31day global mean surface temp prediction)
    #timerange = pd.date_range(start = '1970-01-01',end = '2019-12-31')
    #continuous_trend = regressor.predict(X = timerange.to_julian_date().values.reshape((len(timerange),1)))
    #continuous_trend = xr.DataArray(continuous_trend, dims = ('time',), coords = {'time':timerange}, name = 'tg-anom')
    #continuous_trend.attrs.update({'units':ds['Ta'].getncattr('units'),'coef':regressor.coef_, 'intercept':regressor.intercept_,'description':'Trend (according linear regression against julian date) in monthly GMST, in quasi-linear domain from 1970-01-01 till 2021-06-15', 'title':ds.title})
    #out_path = '/nobackup/users/straaten/predsets/tg_monthly_global_mean_surface_only_trend.nc'
    #continuous_trend.to_netcdf(out_path)
    #

    """
    Processing raw MJO data of BOM
    daily RMM mjo index, originally from http://www.bom.gov.au/climate/mjo/graphics/rmm.74toRealtime.txt
    """
    #def prelag(separations : list, frame: pd.DataFrame):
    #    """
    #    Create len(separations) versions of the same frame
    #    But with values lagged by changing the index and reindexing
    #    """
    #    dfs_per_separation = []
    #    for sep in separations: 
    #       df = frame.copy()
    #       df.index = df.index + pd.Timedelta(value = sep, unit = 'day')
    #       dfs_per_separation.append(df.reindex_like(frame))

    #    combined = pd.concat(dfs_per_separation, axis = 1, keys =pd.Index(separations, name = 'separation'))
    #    return combined

    #rawpath = '/nobackup/users/straaten/predsets/rmm_bom_raw.txt'
    #test = pd.read_fwf(rawpath, skiprows = 2, colspecs = [(8,13),(22,25),(34,37),(37,52),(53,70),(79,81),(82,96)] , names = ['year', 'month', 'day', 'rmm1', 'rmm2', 'phase', 'amplitude'], parse_dates = {'time':['year','month','day']})
    #test = test.iloc[(test['rmm1'] < 999).values,:].set_index('time') #Dropping the NANs (1e36 or 999)
    #test = test.loc['1979-01-01':,:] # This leads to a 291 day gap around 1978, so select data afterwards (uninterrupted daily)
    #test.columns.name = 'metric'
    #separations = np.arange(0,32,1) # Separation in amount of days
    #total = prelag(separations, frame = test)
    #total.columns = pd.MultiIndex.from_frame(total.columns.to_frame().assign(clustid = 1).assign(timeagg = 1).assign(variable = 'mjo'))
    #total.columns = total.columns.reorder_levels(['separation','variable','timeagg','clustid','metric'])
    #total = total.stack('separation') 
    #outpath = '/nobackup/users/straaten/predsets/mjo_daily.h5'
    #total.to_hdf(outpath, key = 'index')

    """
    Processing of PDO daily
    Send to me by Sem. He attached multiple types.
    """
    #rawpath = '/nobackup/users/straaten/predsets/df_PDOs_daily.h5'
    #pdo = pd.read_hdf(rawpath)
    #pdo = pdo[['PDO']]
    #pdo.columns = pd.MultiIndex.from_tuples([('pdo',1,1,'eof')], names = ['variable','timeagg','clustid','metric'])
    #pdo.index.name = 'time'
    #separations = np.arange(0,32,1) # Separation in amount of days
    #total = prelag(separations, frame = pdo).stack('separation')
    #outpath = '/nobackup/users/straaten/predsets/pdo_daily.h5'
    #total.to_hdf(outpath, key = 'index')
