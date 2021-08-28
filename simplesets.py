import numpy as np
import xarray as xr
import pandas as pd

from pathlib import Path

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

# Perhaps I want to add filtering. See e.g. https://github.com/fujiisoup/xr-scipy/blob/master/xrscipy/signal/filters.py 
        
def extract_components(arr: xr.DataArray, ncomps: int = 10, nclusters: int = 4):
    """
    arr needs to be subset to the correct subset, also already detrended if desired
    returns 2 datasets
    - dataset with result of EOF (eigvectors, eigvalues, eof projections timeseries of arr into vectors) 
    - dataset with cluster results (EOF centroids, and composite mean arr fields of the clusters)
    """
    data = arr.values.reshape((arr.shape[0],-1)) # Raveling the spatial dimension, keeping only the values
    U, s, Vt = np.linalg.svd(data, full_matrices=False) # export OPENBLAS_NUM_THREADS=25 put upfront.
    
    component_coords = pd.Index(list(range(ncomps)), dtype = np.int64)

    eigvals = (s**2)[:ncomps]
    eigvectors = Vt[:ncomps,:].reshape((ncomps,) + arr.shape[1:])
    eof = arr.coords.to_dataset()
    eof.coords['component'] = component_coords
    eof['eigvectors'] = (('component','latitude','longitude'),eigvectors)
    eof['eigvalues'] = (('component',),eigvals)
    # Projection of the 3D data into 10 component timeseries
    projection = data @ Vt.T[:,:ncomps]
    eof['projection'] = (('time','component'),projection)
    eof['projection'].attrs['units'] = ''

    #k-means clustering
    k_coords = pd.Index(list(range(nclusters)), dtype = np.int64)
    centroids, assignments = vq.kmeans2(projection, k = nclusters) 
    composites = [data[assignments == cluster,...].mean(axis = 0) for cluster in k_coords] # Composite of the anomalies
    composites = np.concatenate(composites, axis = 0).reshape((nclusters,) + arr.shape[1:]) # new zeroth axis is clusters
    clusters = arr.coords.to_dataset().drop_dims('time')
    clusters.coords['cluster'] = k_coords
    clusters.coords['component'] = component_coords
    clusters['z_comp'] = (('cluster','latitude','longitude'), composites)
    clusters['centroid'] = (('cluster','component'), centroids)

    return eof, clusters

if __name__ == '__main__':
    """
    Creation of the weighted soil moisture 
    """
    #merge_soilm(directory = Path('/nobackup/users/straaten/EXT_extra/45r1')) 
    
    """
    Regime predictors
    """
    #ndays = 3
    ##arr = lead_time_1_z300(ndays = ndays)
    ##arr = arr.sortby('time') # Potentially, .drop_duplicates('time')
    ##arr.to_netcdf(f'/nobackup/users/straaten/predsets/z300-leadtime-dep-anom_{ndays}D_control.nc')

    #arr = xr.open_dataarray(f'/nobackup/users/straaten/predsets/z300-leadtime-dep-anom_{ndays}D_control.nc')
    def subset_and_prepare(arr: xr.DataArray, months: int = None, detrend_method: str = None) -> xr.Dataset:
        """
        Selection of the temporal slice for which regimes are prepared. 
        either a single month (integer), or multiple months jointly (list of integers)
        If using hindcasts you probably want to exclude march:
        Counts array([3, 4, 5, 6, 7, 8]), array([ 42, 189, 189, 210, 189, 106]))
        possible to detrend the slice linearly (pooling data) with either scipy or sklearn
        """
        assert detrend_method in [None,'scipy','sklearn'], 'Only scipy and sklearn or None are valid detrending options'
        if isinstance(months, int):
            months = list(months)
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
        else:
            final = anom.to_dataset()
        return final

    #arr = era5_z300_resample()
    #monthsets = [list(range(5,9))]
    ##monthsets = [[i] for i in range(5,9)]
    #detrend_method = None #'sklearn'
    #basedir = Path('/scistor/ivm/jsn295/backup/predsets')
    #for months in monthsets:
    #    monthscoord = "".join([str(i) for i in months])
    #    basename = f'z300_1D_months_{monthscoord}_{detrend_method}_detrended'
    #    dataset = subset_and_prepare(arr, months, detrend_method)
    #    eofs, clusters = extract_components(dataset[arr.name], ncomps = 10, nclusters = 4) # Following Ferranti 2015 in ncomps and nclusters
    #    eofs.to_netcdf(basedir / f'{basename}_patterns.nc')
    #    clusters.to_netcdf(basedir / f'{basename}_clusters.nc')
    #    if 'coef' in dataset:
    #        dataset.drop(arr.name).to_netcdf(basedir / f'{basename}_coefs.nc')

    """
    Assignment itself
    needed for preparation of fields:
    - Forecasts 
    - z model climatology (lead time dependent) for anomalies
    - coefficients and intercepts for detrending
    needed for the distance and classification:
    - eigenvectors for projection (1 field to 10 timeseries)
    - centroids for distance computation, (1 value per centroid / cluster)
    """
    
    f = Forecast(indate = '2005-07-09', prefix = 'hin_', cycle = '45r1', basedir = '/scistor/ivm/jsn295/backup/EXT_extra/')
    f.load('z')
    simplepredspath = Path('/scistor/ivm/jsn295/backup/predsets')
    coefset = xr.open_dataset(simplepredspath / 'z300_1D_months_5678_sklearn_detrended_coefs.nc') 
    eigvectors = xr.open_dataset(simplepredspath / 'z300_1D_months_5678_sklearn_detrended_patterns.nc')['eigvectors']
    centroids = xr.open_dataset(simplepredspath / 'z300_1D_months_5678_sklearn_detrended_clusters.nc')['centroid']

    modelclim = ModelClimatology(cycle = '45r1', variable = 'z', name = 'z_45r1_1998-06-07_2019-08-31_1D_1.5-degrees_5_5_mean', basedir = '/scistor/ivm/jsn295/backup/modelclimatology/')
    modelclim.local_clim()

    def anomalize_and_detrend(f: Forecast, modelclim: ModelClimatology, coefset: xr.Dataset) -> xr.DataArray:
        """
        Preparation of z300 to make into anomalies
        Remove seasonal cycle by using the modelclimatology (lead time dependent)
        Then remove the linear (thermal expansion) trend with coefficients determined on ERA5
        """
        # Call upon event-classification
        # just a manual a + bx?
        timeaxis = f.array['time'].to_pandas().index.to_julian_date().values
        trend = coefset['intercept'].values[:,:,np.newaxis] + coefset['coef'].values[:,:,np.newaxis] * timeaxis # (latitude,longitude,time)
        trend_with_coords = f.array.coords.to_dataset().drop('number')
        trend_with_coords = trend_with_coords.assign({f.array.name:xr.Variable(dims = ('latitude','longitude','time'), data = trend)})
        # array (time, latitude, longitude, number)
        detrended = f.array - trend_with_coords[f.array.name] # xarray will handle the remaining coordinate, and ordering of the dims
        return detrended

    def project():
        pass

    def assign(component_timeseries, centroids):
        """
        Component timeseries (n_steps, n_comps)
        centroids (n_clusters, n_comps)
        """
        # Says it requires unit variance in the documentation. Obviously not the case.
        clusters, distance_to_nearest = vq.vq(obs = component_timeseries, code_book = centroids)
        # perhaps, but not correct we want euclidian distance
        distances = np.matmul(component_timeseries.values, centroids.values.T) #(N,n_comps) * (n_comps*n_clusters) = (N,n_clusters)
        minus = component_timeseries.values[:,:,np.newaxis] - centroids.values.T # (N,n_comps,n_clusters)
        squared = minus**2
        distances = np.sqrt(squared.sum(axis = 1)) # Shows that vq also does euclidian distance, but returns only the nearest
        clusters = distances.argmin(axis =1)
        # Perhaps assign a -1 when distances are too large?

    """
    Spatiotemporal mean anomaly simplesets
    Should be doable by creating ERA5 anom observations with a certain clustering and timeagg
    Writing out the clusterarray too, and then loading + matching with the modelclim for that variable.
    - downside = sampling and re-indexing. currently one season, and a mixture of leadtimes. No continuous availability of forecasts with a certain leadtime
    - upside = produces a set with 'observations' included. Could be helpful for investigating biases in simulated variables themselves
    block-cluster made in swvl-simple.nc 
    """
    combinations = pd.DataFrame({'obs':['swvl13-anom_1981-01-01_2019-09-30_1D_1.5-degrees','swvl4-anom_1981-01-01_2019-09-30_1D_1.5-degrees','z300-anom_1979-01-01_2019-12-31_1D_1.5-degrees','sst-anom_1979-01-01_2019-12-31_1D_1.5-degrees'],
        'block':['swvl-simple','swvl-simple','swvl-simple','sst-simple'],
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
        clim.localclim(obs = obs, daysbefore = 5, daysafter = 5, mean = False, quant = 0.75)
        clim.savelocalclim()

    #timeagg = 14
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
