import numpy as np
import xarray as xr
import pandas as pd

from pathlib import Path

from forecasts import Forecast, ModelClimatology
from observations import EventClassification

import scipy.cluster.vq as vq
from scipy.signal import detrend
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
    Regime predictors
    """
    ndays = 3
    #arr = lead_time_1_z300(ndays = ndays)
    #arr = arr.sortby('time') # Potentially, .drop_duplicates('time')
    #arr.to_netcdf(f'/nobackup/users/straaten/predsets/z300-leadtime-dep-anom_{ndays}D_control.nc')

    arr = xr.open_dataarray(f'/nobackup/users/straaten/predsets/z300-leadtime-dep-anom_{ndays}D_control.nc')
    
    # Want to do the decomposition per month
    months = [4,5,6,7,8] # March excluded only 42 starting points. Counts array([3, 4, 5, 6, 7, 8]), array([ 42, 189, 189, 210, 189, 106]))
    months = [5]
    all_months_patterns = []
    all_months_clustered = []
    for month in months:
        #anom = arr[arr.time.dt.month == month,...]
        anom = arr[arr.time.dt.month.isin([5,6,7,8]),...].copy()
        # Anomalies are not detrended. Should they be? perhaps.
        # Can perhaps be done by expanding the time dimension if that fits in memory. And then scipy detrend
        # No true, because of Nana. Also it is not really reproducible when assigning a single forecasts based on its anoms.
        # unless I encode something with coefficients
        #anom.values = detrend(anom.values, axis = 0)

        eofs, clusters = extract_components(anom, ncomps = 10, nclusters = 4) # Following Ferranti 2015 in ncomps and nclusters
        all_months_patterns.append(eofs)
        all_months_clustered.append(clusters)

    
    #basename = f'z300_{ndays}D_detrended'
    #basename = f'z300_MJJA_detrended'
    basename = f'z300_MJJA_trended_test'
    #basename = f'z300_{ndays}D_trended'
    all_months_patterns = xr.concat(all_months_patterns, pd.Int64Index(months, name = 'month'))
    # actually the temporal projections can just be concatenated over time
    all_months_patterns.to_netcdf(f'/nobackup/users/straaten/predsets/{basename}_patterns.nc')
    
    all_months_clustered = xr.concat(all_months_clustered, pd.Int64Index(months, name = 'month'))
    all_months_clustered.to_netcdf(f'/nobackup/users/straaten/predsets/{basename}_clusters.nc')

    """
    Spatiotemporal mean anomaly simplesets
    Should be doable by creating ERA5 anom observations with a certain clustering and timeagg
    Writing out the clusterarray too, and then loading + matching with the modelclim for that variable.
    - downside = sampling and re-indexing. currently one season, and a mixture of leadtimes. No continuous availability of forecasts with a certain leadtime
    - upside = produces a set with 'observations' included. Could be helpful for investigating biases in simulated variables themselves
    """

    #e.g. clustid 0 for a certain square soilm region?

