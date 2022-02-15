#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 09:31:16 2018

@author: straaten
"""
import os
import sys
import numpy as np
import xarray as xr
import pandas as pd
import dask.dataframe as dd
import itertools
import properscoring as ps
import uuid
import multiprocessing
from datetime import datetime
from observations import SurfaceObservations, Climatology, EventClassification, Clustering
from forecasts import Forecast, ModelClimatology
from helper_functions import monthtoseasonlookup, unitconversionfactors, lastconsecutiveabove, assignmidpointleadtime
from fitting import NGR, Logistic, ExponentialQuantile

def matchforecaststoobs(datesubset, variable, outfilepath, time_agg = 31, n_members = 11, cycle = '45r1', maxleadtime = 46, newvarkwargs = {}, loadkwargs = {}, leadtimerange = None):
    """
    Function to act as a child process, matching forecasts to the obs until a certain size dataframe is reached. Then that file is written, and the booksfile is updated.
    Neirest neighbouring to match the forecast grid to the observed grid in the form of the clusterarray belonging to the observations. Be careful when domain of the (clustered) observations is larger.
    Determines the order in which space-time aggregation and classification is done based on the potential newvar.
    Also calls unit conversion for the forecasts. (the observed units of a newvariable are actually its old units)
    Loadkwargs can carry the delimiting spatial corners if only a part of the domain is desired.
    """
    def find_forecasts(date):
        """
        Here the forecasts corresponding to the observations are determined, by testing their existence.
        Hindcasts and forecasts might be mixed. But they are in the same class.
        Leadtimes may differ.
        Returns an empty list if non were found
        """       
        # Find forecast initialization times that fully contain the observation date (including its window for time aggregation)
        containstart = date + pd.Timedelta(str(time_agg) + 'D') - pd.Timedelta(str(maxleadtime) + 'D')
        containend = date
        contain = pd.date_range(start = containstart, end = containend, freq = 'D').strftime('%Y-%m-%d')
        if variable in ['tg','tx','rr']:
            forbasedir = '/nobackup/users/straaten/EXT/'
        else:
            forbasedir = '/nobackup/users/straaten/EXT_extra/'
        forecasts = [Forecast(indate, prefix = 'for_', cycle = cycle, basedir = forbasedir) for indate in contain]
        hindcasts = [Forecast(indate, prefix = 'hin_', cycle = cycle, basedir = forbasedir) for indate in contain]
        # select from potential forecasts only those that exist.
        forecasts = [f for f in forecasts if os.path.isfile(f.basedir + f.processedfile)]
        hindcasts = [h for h in hindcasts if os.path.isfile(h.basedir + h.processedfile)]
        return(forecasts + hindcasts)
    
    def load_forecasts(date, listofforecasts):
        """
        Gets the daily processed forecasts into memory. Delimited by the left timestamp and the aggregation time.
        This is done by using the load method of each Forecast class in the dictionary. They are stored in a list.
        """ 
        tmin = date
        tmax = date + pd.Timedelta(str(time_agg - 1) + 'D') # -1 because date itself also counts for one day in the aggregation.
       
        for forecast in listofforecasts:
            forecast.load(variable = variable, tmin = tmin, tmax = tmax, n_members = n_members, **loadkwargs)
     
    def force_resolution(forecast, time = True):
        """
        Force the observed resolution onto the supplied Forecast. Checks if the same resolution and force spatial/temporal aggregation if that is not the case. Checks will fail on the roll-norm difference, e.g. 2D-roll-mean observations and .
        Makes use of the methods of each Forecast class. Checks can be switched on and off. Time-rolling is never applied to the forecasts as for each date already the precise window was loaded, but space-rolling is.
        """                
        if time:
            # Check time aggregation
            obstimemethod = '31D-roll-mean' 
            try:
                fortimemethod = getattr(forecast, 'timemethod')
                if not fortimemethod == obstimemethod:
                    raise AttributeError
                else:
                    print('Time already aligned')
            except AttributeError:
                print('Aligning time aggregation')
                freq, rolling, method = obstimemethod.split('-')
                forecast.aggregatetime(freq = freq, method = method, keep_leadtime = True, ndayagg = time_agg, rolling = False)
                      
    def force_new_variable(forecast, newvarkwargs, inplace = True, newvariable: str = None):
        """
        Call upon event classification on the forecast object to get the on-the-grid conversion of the base variable.
        This is classification method is the same as applied to obs and is determined by the similar name.
        Possibly returns xarray object if inplace is False. If newvar is anom and model climatology is supplied through newvarkwargs, it should already have undergone the change in units.
        """
        if newvariable is None:
            newvariable = obs.newvar
        method = getattr(EventClassification(obs = forecast, **newvarkwargs), newvariable)
        return(method(inplace = inplace))
    
    p = multiprocessing.current_process()
    print('Starting:', p.pid)
    
    aligned_basket = {}
    aligned_basket_size = 0 # Size of the content in the basket. Used to determine when to write
    
    while (aligned_basket_size < 5*10**8) and (not datesubset.empty):
        
        date = datesubset.iloc[0]
        listofforecasts = find_forecasts(date) # Results in empty list if none were found
        
        if listofforecasts:
            
            load_forecasts(date = date, listofforecasts = listofforecasts) # Does this modify listofforecasts inplace?
            
            for forecast in listofforecasts:
                newvar = 'anom'
                force_new_variable(forecast, newvarkwargs = newvarkwargs, inplace = True, newvariable = newvar) # If newvar is anomaly then first new variable and then aggregation. If e.g. newvar is pop then first aggregation then transformation.
                force_resolution(forecast, time = True)
                forecast.array = forecast.array.swap_dims({'time':'leadtime'}) # So it happens inplace
            
            allleadtimes = xr.concat(objs = [f.array for f in listofforecasts], 
                                             dim = 'leadtime') # concatenates over leadtime dimension.
            if not leadtimerange is None:
                present = [l for l in list(leadtimerange) if l in allleadtimes.leadtime]
                subleadtimes = allleadtimes.sel(leadtime = present)
                print(f'{len(present)} of {leadtimerange} present in {date}')
            else:
                subleadtimes = allleadtimes
            if subleadtimes.size != 0:
                aligned_basket.update({date:subleadtimes})
                print(date, 'added')
                # If aligned takes too much system memory (> 500Mb) . Write it out
                aligned_basket_size += sys.getsizeof(subleadtimes)

        datesubset = datesubset.drop(date)
    
    if aligned_basket:
        pass
        # Writing files
        
    return aligned_basket

def select_highest_member(field: xr.DataArray, at_lat: float = 52.0, at_lon: float = 4.0):
    """
    From a field forecast by multiple members it selects the member
    with the highest value in the nearest-neighbor gridcell at a location
    field is indexed as (leadtime/time, latitude,longitude,n_member)
    will return as (leadtime/time, latitude,longitude)
    Possible that multiple leadtimes are supplied, therefore a loop
    """
    highest = []
    for l in field.leadtime:
        cell_values = field.sel(leadtime = l, latitude = at_lat, longitude = at_lon, method = 'nearest')
        where_max = (cell_values == cell_values.max()).values
        print(cell_values.number.values[where_max])
        selection = field.sel(leadtime = [l]).isel(number = where_max).squeeze('number', drop = True)
        highest.append(selection)
    return xr.concat(highest, dim = 'leadtime')

        
if __name__ == '__main__':
    dates = pd.date_range('2000-07-01','2000-07-03').to_series()

    highresmodelclim = ModelClimatology(cycle = '45r1', variable = 'z', name = 'z_45r1_1998-06-07_2019-08-31_1D_1.5-degrees_5_5_mean', basedir = '/nobackup/users/straaten/modelclimatology/')
    highresmodelclim.local_clim()
    newvarkwargs = {'climatology':highresmodelclim}

    test = matchforecaststoobs(datesubset = dates, variable = 'z', outfilepath = '', time_agg = 31, n_members = 11, leadtimerange = range(12,16), newvarkwargs = newvarkwargs)
    #test = matchforecaststoobs(datesubset = dates, variable = 'z', outfilepath = '', time_agg = 31, n_members = 11, leadtimerange = None, newvarkwargs = newvarkwargs)

    highest = {}
    for date in dates:
        highest.update({date:select_highest_member(test[date])})

    total = np.concatenate(list(highest.values()), axis = 0)


