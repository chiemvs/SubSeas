#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 13:55:37 2018

@author: straaten
"""
import numpy as np
import pandas as pd
import xarray as xr

def agg_space(array, orlats, orlons, step, skipna = False, method = 'mean', by_degree = False):
    """
    Regular lat lon or gridbox aggregation by creating new single coordinate which is used for grouping.
    In the case of degree grouping the groups might not contain an equal number of cells.
    Returns an adapted array and spacemethod string for documentation.
    TODO: make efficient handling of dask arrays. Ideas for this are commented, problem is that stacking and groupby fuck up the efficiency.
    TODO: enable minimum number of cells for computation.
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
        grouped.values[toomanyna] = np.nan # Set the values.
    
    # Compute new coordinates, and construct a spatial multiindex with lats and lons for each group
    newlat = orlats.to_series().groupby(binlat).mean()
    newlon = orlons.to_series().groupby(binlon).mean()
    newlatlon = pd.MultiIndex.from_product([newlat, newlon], names=('latitude', 'longitude'))
    
    # Prepare the coordinates of stack dimension and replace the array
    grouped['latlongroup'] = newlatlon        
    array = grouped.unstack('latlongroup')
    spacemethod = '-'.join([str(step), 'cells', method]) if not by_degree else '-'.join([str(step), 'degrees', method])
    return(array, spacemethod)

def agg_time(array, freq = 'w' , method = 'mean'):
    """
    Uses the pandas frequency indicators. Method can be mean, min, max, std
    Completely lazy when loading is lazy. Returns an adapted array and a timemethod string for documentation.
    """
    f = getattr(array.resample(time = freq, closed = 'left', label = 'left'), method) # timestamp is left and can be changed with label = 'right'
    array = f('time', keep_attrs=True, skipna = False) 
    timemethod = '-'.join([freq,method])
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
