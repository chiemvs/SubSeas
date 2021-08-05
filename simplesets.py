import numpy as np
import xarray as xr
import pandas as pd

from pathlib import Path

from forecasts import Forecast, ModelClimatology, for_netcdf_encoding
from observations import SurfaceObservations, EventClassification, Climatology
from comparison import ForecastToObsAlignment

import scipy.cluster.vq as vq
from scipy.signal import detrend

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
    modelclim = ModelClimatology(cycle='45r1', variable = variable, **{'name':f'{variable}_45r1_1998-06-07_2018-08-31_1D_1.5-degrees_5_5_mean'}) # Name for loading

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
    #
    ## Want to do the decomposition per month
    #months = [4,5,6,7,8] # March excluded only 42 starting points. Counts array([3, 4, 5, 6, 7, 8]), array([ 42, 189, 189, 210, 189, 106]))
    #months = [5]
    #all_months_patterns = []
    #all_months_clustered = []
    #for month in months:
    #    #anom = arr[arr.time.dt.month == month,...]
    #    anom = arr[arr.time.dt.month.isin([5,6,7,8]),...].copy()
    #    # Anomalies are not detrended. Should they be? perhaps.
    #    # Can perhaps be done by expanding the time dimension if that fits in memory. And then scipy detrend
    #    # No true, because of Nana. Also it is not really reproducible when assigning a single forecasts based on its anoms.
    #    # unless I encode something with coefficients
    #    #anom.values = detrend(anom.values, axis = 0)

    #    eofs, clusters = extract_components(anom, ncomps = 10, nclusters = 4) # Following Ferranti 2015 in ncomps and nclusters
    #    all_months_patterns.append(eofs)
    #    all_months_clustered.append(clusters)

    #
    ##basename = f'z300_{ndays}D_detrended'
    ##basename = f'z300_MJJA_detrended'
    #basename = f'z300_MJJA_trended_test'
    ##basename = f'z300_{ndays}D_trended'
    #all_months_patterns = xr.concat(all_months_patterns, pd.Int64Index(months, name = 'month'))
    ## actually the temporal projections can just be concatenated over time
    #all_months_patterns.to_netcdf(f'/nobackup/users/straaten/predsets/{basename}_patterns.nc')
    #
    #all_months_clustered = xr.concat(all_months_clustered, pd.Int64Index(months, name = 'month'))
    #all_months_clustered.to_netcdf(f'/nobackup/users/straaten/predsets/{basename}_clusters.nc')

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
        'modelclim':['swvl13_45r1_1998-06-07_2018-08-31_1D_1.5-degrees_5_5_mean','swvl4_45r1_1998-06-07_2018-08-31_1D_1.5-degrees_5_5_mean','z_45r1_1998-06-07_2018-08-31_1D_1.5-degrees_5_5_mean','sst_45r1_1998-06-07_2018-08-31_1D_1.5-degrees_5_5_mean'],
        'expname':['paper3-3-simple','paper3-3-simple','paper3-3-simple','paper3-3-simple'],
        }, index = pd.Index(['swvl13','swvl4', 'z','sst'], name = 'basevar'))

    def create_blocks(basevar):
        obs = SurfaceObservations(basevar = basevar, basedir = '/nobackup/users/straaten/ERA5/', name = combinations.loc[basevar,'obs']) # Override normal E-OBS directory
        obs.load(tmin = '1998-06-07', tmax = '2019-08-31')
        obs.aggregatespace(clustername = combinations.loc[basevar,'block'], level = 1)
        obs.aggregatetime(freq = '7D', method = 'mean', rolling = True)
        obs.newvar = 'anom' # was already an anomalie
        obs.savechanges()

        # loading of the constructed observations, streamlines the clusterarray for matching
        obs = SurfaceObservations(basevar = basevar, basedir = '/nobackup/users/straaten/ERA5/', name = f'{basevar}-anom_1998-06-07_2019-08-31_7D-roll-mean_1-{combinations.loc[basevar,"block"]}-mean') # Override normal E-OBS directory
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

    def create_clim(basevar):
        """
        Making climatologies of the newly created block-average observations
        """
        obs = SurfaceObservations(basevar = basevar, basedir = '/nobackup/users/straaten/ERA5/', name = f'{basevar}-anom_1998-06-07_2019-08-31_7D-roll-mean_1-{combinations.loc[basevar,"block"]}-mean') # Override normal E-OBS directory
        obs.load()
        clim = Climatology(f'{basevar}-anom')
        clim.localclim(obs = obs, daysbefore = 5, daysafter = 5, mean = False, quant = 0.75)
        clim.savelocalclim()

    #for var in combinations.index:
    #    create_blocks(var)
    #    create_clim(var)
