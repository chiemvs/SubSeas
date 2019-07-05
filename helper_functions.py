#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 13:55:37 2018

@author: straaten
"""
import numpy as np
import pandas as pd
import xarray as xr

def agg_space2(array, orlats, orlons, step, skipna = False, method = 'mean', by_degree = False):
    """
    Regular lat lon or gridbox aggregation by creating new single coordinate which is used for grouping.
    In the case of degree grouping the groups might not contain an equal number of cells.
    Returns an adapted array and spacemethod string for documentation.
    TODO: make efficient handling of dask arrays. Ideas for this are commented, problem is that stacking and groupby fuck up the efficiency.
    """
    
    # Binning. The rims are added to the closest bin
    if by_degree:
        binlon = np.digitize(x = orlons, bins = np.arange(orlons.min(), orlons.max(), step)) + 200 # + 200 to make sure that both groups are on a different scale, needed for when we want to derive unique (integer) combinations. 
        binlat = np.digitize(x = orlats, bins = np.arange(orlats.min(), orlats.max(), step))
    else:
        lon_n, lon_rem = divmod(orlons.size, step)
        binlon = np.repeat(np.arange(1, lon_n + 1), repeats = step)
        binlon = np.append(binlon, np.repeat(binlon[-1], lon_rem)) + 200
        lat_n, lat_rem = divmod(orlats.size, step)
        binlat = np.repeat(np.arange(1, lat_n + 1), repeats = step)
        binlat = np.append(binlat, np.repeat(binlat[-1], lat_rem))
    
    # Counts along axis
    #binlatcount = np.unique(np.char.array(binlat), return_counts=True)[1]
    #binloncount = np.unique(np.char.array(binlon), return_counts=True)[1]
    
    # Concatenate as strings to a group variable
    combined = np.core.defchararray.add(binlat.astype(np.unicode_)[:, None], binlon.astype(np.unicode_)[None, :])
    combined = xr.DataArray(combined.astype(np.int), [orlats, orlons], name = 'latlongroup')
    
    # if array is dask array: Re-chunking such that the each group is in a single chunk.
    #if array.chunks:
    #    temp = xr.Dataset({'vals':array,'cor':combined})
    #    temp = temp.chunk({'latitude':tuple(binlatcount), 'longitude':tuple(binloncount)})
    #    temp.set_coords('cor', inplace = True)
    #    array = array.chunk({'latitude':tuple(binlatcount), 'longitude':tuple(binloncount)})
        
    
    #test = array.stack(latlon = ['latitude', 'longitude']) # This fucks up the chunking.
    #groups = array.groupby(combined)
    #for key, arr in groups:
    #    f = getattr(arr, method)
    #    res = f('stacked_latitude_longitude', keep_attrs=True)
    
    # Compute grouped values. This stacks the dimensions to one spatial and one temporal.
    f = getattr(array.groupby(combined), method) # This loads the data into memory if a non-dask array is supplied. Put into try except framework for personally defined functions.
    grouped = f('stacked_latitude_longitude', keep_attrs=True) # if very strict: skipna = False
    
    # Maximum amount of missing values is 40% in space  = 0.4 * stepsize^2 (in case of cells)
    if not skipna:
        if by_degree:
            maxnacells = int(0.4 * (len(binlat[binlat == 1]))**2)
        else:
            maxnacells = int(0.4 * step**2)
        toomanyna = np.isnan(array).groupby(combined).sum('stacked_latitude_longitude') > maxnacells
        grouped.values[toomanyna.values] = np.nan # Set the values. NOTE: an error might occur here for dataarray with a limited length first axis (time). Then change toomanyna to toomanyna.values
    
    # Compute new coordinates, and construct a spatial multiindex with lats and lons for each group
    newlat = orlats.to_series().groupby(binlat).mean()
    newlon = orlons.to_series().groupby(binlon).mean()
    newlatlon = pd.MultiIndex.from_product([newlat, newlon], names=('latitude', 'longitude'))
    
    # Prepare the coordinates of stack dimension and replace the array
    grouped['latlongroup'] = newlatlon        
    array = grouped.unstack('latlongroup')
    spacemethod = '-'.join([str(step), 'cells', method]) if not by_degree else '-'.join([str(step), 'degrees', method])
    return(array, spacemethod)

def agg_space(array, orlats, orlons, step, skipna = False, method = 'mean', by_degree = False, rolling = False):
    """
    Regular lat lon or gridbox aggregation by creating new single coordinate which is used for grouping.
    In the case of degree grouping the groups might not contain an equal number of cells.
    Returns an adapted array and spacemethod string for documentation.
    """
    # Maximum amount of missing values is 40% in space
    maxnafrac = 0.4
    if rolling:
        # If rolling and bydegree then translate the number of degrees to the number of cells within a window.
        # Approach of rolling first along one dimension then along the other spatial one. (Will slightly affect the weighting of values near coastlines as we have non-equal group-sizes)
        # For even nr of cells stepsizes the rolling operator takes [i-1, i] for 2 and [i-2, i-1, i, i+1] for 4 
        if by_degree:
            original_size = np.diff(orlons)[int(len(orlons)/2)]
            cellstep = int(step // original_size) # conversion of degree step to step of ncells which are completely contained in the degree window.
        else:
            cellstep = int(step)
            
        if cellstep <= 1:
            raise ValueError('Stepsize in rolling aggregation should contain more than one cell, for by degree this means fully containing')
        
        attrs = array.attrs
        name = array.name
        f1 = getattr(array.rolling(dim = {'latitude':cellstep}, min_periods = int(np.ceil((1 - maxnafrac) * cellstep)), center = True), method)
        f2 = getattr(f1().rolling(dim = {'longitude':cellstep}, min_periods = int(np.ceil((1 - maxnafrac) * cellstep)), center = True), method)
        array = f2()
        array.attrs = attrs
        array.name = name
        # Other approaches, based on multi-dimensional array methods:
        #from scipy.ndimage.filters import generic_filter
        #f = getattr(np, 'nan'+method)
        #out = generic_filter(array, f, size = (1,3,3))
        # Or check this: https://stackoverflow.com/questions/8174467/vectorized-moving-window-on-2d-array-in-numpy
    else:
        # Binning. The rims are added to the closest bin
        if by_degree:
            binlon = np.digitize(x = orlons, bins = np.arange(orlons.min(), orlons.max(), step)) + 200 # + 200 to make sure that both groups are on a different scale, needed for when we want to derive unique (integer) combinations. 
            binlat = np.digitize(x = orlats, bins = np.arange(orlats.min(), orlats.max(), step))
        else:
            lon_n, lon_rem = divmod(orlons.size, step)
            binlon = np.repeat(np.arange(1, lon_n + 1), repeats = step)
            binlon = np.append(binlon, np.repeat(binlon[-1], lon_rem)) + 200
            lat_n, lat_rem = divmod(orlats.size, step)
            binlat = np.repeat(np.arange(1, lat_n + 1), repeats = step)
            binlat = np.append(binlat, np.repeat(binlat[-1], lat_rem))
        
       # Concatenate as strings to a group variable
        combined = np.core.defchararray.add(binlat.astype(np.unicode_)[:, None], binlon.astype(np.unicode_)[None, :])
        combined = xr.DataArray(combined.astype(np.int), [orlats, orlons], name = 'latlongroup')
    
        # Compute grouped values. This stacks the dimensions to one spatial and one temporal.
        f = getattr(array.groupby(combined), method) # This loads the data into memory if a non-dask array is supplied. Put into try except framework for personally defined functions.
        grouped = f('stacked_latitude_longitude', keep_attrs=True) # if very strict: skipna = False
        
        # Maximum amount of missing values is 40% in space  = 0.4 * stepsize^2 (in case of cells)
        if not skipna:
            if by_degree:
                maxnacells = int(maxnafrac * (len(binlat[binlat == 1]))**2)
            else:
                maxnacells = int(maxnafrac * step**2)
            toomanyna = np.isnan(array).groupby(combined).sum('stacked_latitude_longitude') > maxnacells
            grouped.values[toomanyna.values] = np.nan # Set the values. NOTE: an error might occur here for dataarray with a limited length first axis (time). Then change toomanyna to toomanyna.values
        
        # Compute new coordinates, and construct a spatial multiindex with lats and lons for each group
        newlat = orlats.to_series().groupby(binlat).mean()
        newlon = orlons.to_series().groupby(binlon).mean()
        newlatlon = pd.MultiIndex.from_product([newlat, newlon], names=('latitude', 'longitude'))
        
        # Prepare the coordinates of stack dimension and replace the array
        grouped['latlongroup'] = newlatlon        
        array = grouped.unstack('latlongroup')
    
    # Some bookkeeping
    char_degree = 'degrees' if by_degree else 'cells'
    char_rol = 'roll' if rolling else 'norm'
    spacemethod = '-'.join([str(step), char_degree, char_rol, method])
    return(array, spacemethod)

def agg_time(array, freq = 'w', method = 'mean', ndayagg = None, returnndayagg = False, rolling = False):
    """
    Assumes an input array with a regularly spaced daily time index. Uses the pandas frequency indicators. Method can be mean, min, max, std
    Completely lazy when loading is lazy. Returns an adapted array and a timemethod string for documentation.
    Skipna is false so no non-observations within the period allowed.
    Returns array with the last (incomplete) interval removed if the input length is not perfectly divisible.
    For rolling the first incomplete fields (that lie previous in time) are also removed.
    """
    if rolling:
        if ndayagg is None:
            # Translate the frequency to a number of days window.
            ndayagg = int(pd.date_range('2000-01-01','2000-12-31', freq = freq).to_series().diff().dt.days.mode())
        
        name = array.name
        attrs = array.attrs
        f = getattr(array.rolling({'time':ndayagg}, center = False), method) # Stamped right
        array = f()
        # Left timestamping, and keeping the attributes
        array = array.assign_coords(time = array.time - pd.Timedelta(str(ndayagg - 1) + 'D')).isel(time = slice(ndayagg - 1, None))
        array.name = name
        array.attrs = attrs
        
    else:
        input_length = len(array.time)
        f = getattr(array.resample(time = freq, closed = 'left', label = 'left'), method) # timestamp is left and can be changed with label = 'right'
        array = f('time', keep_attrs=True, skipna = False)
        if ndayagg is None:
            ndayagg = (array.time.values[1] - array.time.values[0]).astype('timedelta64[D]').item().days # infer the interval length.
        if (input_length % ndayagg) != 0:
            array = array.isel(time = slice(0,-1,None))
    
    timemethod = '-'.join([freq,'roll',method]) if rolling else '-'.join([freq,'norm',method]) 
    if returnndayagg:
        return(array, timemethod, ndayagg)
    else:    
        return(array, timemethod)
    

def unitconversionfactors(xunit, yunit):
    """
    Provides the correct factors to linearly convert the units of x to the unit of y: y = ax + b, thus returning a and b.
    The desired outcome should be the units supplied at y.
    """
    library = {
            'KtoCelsius':(1,-273.15),
            'CelsiustoK':(1,273.15),
            'mtomm':(1000,0),
            'mmtom':(0.001,0)}
    try:
        return(library[xunit + 'to' + yunit])
    except KeyError:
        if yunit == xunit:
            return((1,0))
        else:
            print('Check supplied units, they are not equal and conversion could not be found in library')

def monthtoseasonlookup(months):
    """
    Character array output of integer array input
    """
    array = np.array([
    None,
    'DJF', 'DJF',
    'MAM', 'MAM', 'MAM',
    'JJA', 'JJA', 'JJA',
    'SON', 'SON', 'SON',
    'DJF'
    ])
    return(array[months])       
    
def nanquantile(array, q):
    """
    Get quantile along the first axis of the array. Faster than numpy, because it has only a quantile function ignoring nan's along one dimension.
    Quality checked against numpy native method.
    """
    # amount of valid (non NaN) observations along the first axis. Plus repeated version
    valid_obs = np.sum(np.isfinite(array), axis=0)
    valid_obs_3d = np.repeat(valid_obs[np.newaxis, :, :], array.shape[0], axis=0)
    # replace NaN with maximum, but only for slices with more than one valid observation along the first axis.
    max_val = np.nanmax(array)
    array[np.logical_and(np.isnan(array), valid_obs_3d > 0 )] = max_val
    # sort - former NaNs will move to the end
    array = np.sort(array, axis=0)

    quant_arr = np.zeros(shape=(array.shape[1], array.shape[2]))

    # desired position as well as floor and ceiling of it
    k_arr = (valid_obs - 1) * q
    f_arr = np.floor(k_arr).astype(np.int32)
    c_arr = np.ceil(k_arr).astype(np.int32)
    fc_equal_k_mask = f_arr == c_arr

    # linear interpolation (like numpy percentile) takes the fractional part of desired position
    floor_val = _zvalue_from_index(arr=array, ind=f_arr) * (c_arr - k_arr)
    ceil_val = _zvalue_from_index(arr=array, ind=c_arr) * (k_arr - f_arr)

    quant_arr = floor_val + ceil_val
    quant_arr[fc_equal_k_mask] = _zvalue_from_index(arr=array, ind=k_arr.astype(np.int32))[fc_equal_k_mask]  # if floor == ceiling take floor value

    return quant_arr

def _zvalue_from_index(arr, ind):
    """private helper function to work around the limitation of np.choose() by employing np.take()
    arr has to be a 3D array
    ind has to be a 2D array containing values for z-indicies to take from arr
    See: http://stackoverflow.com/a/32091712/4169585
    This is faster and more memory efficient than using the ogrid based solution with fancy indexing.
    """
    # get number of columns and rows
    _,nC,nR = arr.shape

    # get linear indices and extract elements with np.take()
    idx = nC*nR*ind + np.arange(nC*nR).reshape((nC,nR))
    return np.take(arr, idx)

def auto_cor(df, column, cutofflag = 15, return_lag_last_abovezero = False, return_char_length = False):
    """
    Computes the lagged autocorrelation in df[column] starting at lag one till cutofflag.
    It is assumed that the rows in the dataframe are unique and sorted by time.
    For the scoresets written in experiment 2 this is a correct assumption.
    """
    res = np.zeros((cutofflag,), dtype = 'float32')
    for i in range(cutofflag):
        res[i] = df.loc[:,column].autocorr(lag = i + 1)
        if return_lag_last_abovezero and (res[i] <= 0): # Will return previous lag if this is the first lag (i+1) below zero
            return(i)
            
    if return_lag_last_abovezero: # If not returned previously in the loop.
        return(cutofflag)
    elif return_char_length: # Formula by Leith 1973, referenced in Feng (2011)
        return(((1 - np.arange(1, cutofflag + 1)/cutofflag) * res * 2).sum() + 1)
    else:
        return(res)

def lastconsecutiveabove(series, threshold = 0):
    """
    Provides the leadtime index of the last positive value in the first consecutively positive series
    So input is a multi-index pandas series (of which one level is the leadtime index)
    Returns a float or integer.
    """
    gtzero = series.gt(threshold)
    leadtimeindexlevel = series.index.names.index('leadtime')
    if not gtzero.iloc[0]:
        leadtime = 0
    elif gtzero.all():
        leadtime = gtzero.index.get_level_values(level = leadtimeindexlevel)[-1]
    else:
        tupfirst = gtzero.idxmin()
        locfirst = gtzero.index.get_loc(tupfirst)
        tup = gtzero.index.get_values()[locfirst - 1]
        if isinstance(tup, tuple):
            leadtime = tup[leadtimeindexlevel]
        else: # In this case we actually didn't have a multi-index.
            leadtime = tup
    return(leadtime)

def assignmidpointleadtime(frame, timeagg = None):
    """
    To correct the leadtime index for the fact that leadtime was assigned to the first day.
    Now we assign it to the midpoint of the temporal aggregation. Acts on groupedby timeagg.
    Or on a unique frame and with the string timeagg supplied.
    """
    temp = frame.reset_index()
    pos_uncor_indices = ['spaceagg','latitude','longitude','quantile']
    try:
        midpointday = (int(temp['timeagg'].values[0][0]) - 1) / 2
    except KeyError:
        midpointday = (int(timeagg[0]) - 1) / 2
    print(midpointday)
    correctedleadtimes = temp['leadtime'].values + midpointday
    frame.index = pd.MultiIndex.from_arrays([temp[i].values for i in pos_uncor_indices if (i in temp.columns)] + [correctedleadtimes], names = [i for i in pos_uncor_indices if (i in temp.columns)] + ['leadtime'])
    return(frame)
