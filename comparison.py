#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 09:31:16 2018

@author: straaten
"""
import os
import numpy as np
import xarray as xr
import pandas as pd
import itertools
from observations import SurfaceObservations
from forecasts import Forecast

class ForecastToObsAlignment(object):
    """
    Idea is: you have already prepared an observation class, with a certain variable, temporal extend and space/time aggregation.
    This searches the corresponding forecasts, and possibly forces the same aggregations.
    TODO: not sure how to handle groups yet.
    """
    def __init__(self, season, observations):
        """
        Temporal extend can be read from the attributes of the observation class. 
        Specify the season under inspection for subsetting.
        """
        self.season = season
        self.obs = observations
        self.dates = self.obs.array.coords['time'].to_series() # Left stamped
        # infer dominant time aggregation in days
        self.time_agg = int(self.dates.diff().dt.days.mode())
        self.maxleadtime = 46
    
    def find_forecasts(self):
        """
        Here the forecasts corresponding to the observations are determined, by testing their existence.
        Hindcasts and forecasts might be mixed. But they are in the same class.
        Leadtimes may differ.
        TODO: Include seasonal subsetting in this method.
        """
        self.forecasts = {}
        for date in self.dates.tolist():
            # Find forecast initialization times that fully contain the observation date (including its window for time aggregation)
            containstart = date + pd.Timedelta(str(self.time_agg) + 'D') - pd.Timedelta(str(self.maxleadtime) + 'D')
            containend = date
            contain = pd.date_range(start = containstart, end = containend, freq = 'D').strftime('%Y-%m-%d')
            forecasts = [Forecast(indate, prefix = 'for_') for indate in contain]
            hindcasts = [Forecast(indate, prefix = 'hin_') for indate in contain]
            # select from potential forecasts only those that exist.
            forecasts = [f for f in forecasts if os.path.isfile(f.basedir + f.processedfile)]
            hindcasts = [h for h in hindcasts if os.path.isfile(h.basedir + h.processedfile)]
            self.forecasts.update({date : forecasts + hindcasts})   
        
    def load_forecasts(self, n_members):
        """
        Gets the daily processed forecasts into memory. Delimited by the left timestamp and the aggregation time.
        This is done by using the load method of each Forecast class in the dictionary. They are stored in a list.
        """
        for date, listofforecasts in self.forecasts.items():
            
            if listofforecasts: # empty lists are skipped
                tmin = date
                tmax = date + pd.Timedelta(str(self.time_agg - 1) + 'D') # -1 because date itself also counts for one day in the aggregation.
                
                for forecast in listofforecasts:
                    forecast.load(variable = self.obs.basevar, tmin = tmin, tmax = tmax, n_members = n_members)
    
    def force_resolution(self):
        """
        Check if the same resolution and force spatial/temporal aggregation if that is not the case.
        Makes use of the methods of each Forecast class
        """
        
    def match(self):
        """
        Neirest neighbouring to match pairs. Also converts forecast units to observed units.
        Creates the dataset. Possibly writes to disk too if intermediate results press too much on memory?.
        """
        from helper_functions import unitconversionfactors
        
        self.aligned = {}
        
        for date, listofforecasts in self.forecasts.items():
            
            if listofforecasts:
                fieldobs = self.obs.array.sel(time = date).expand_dims('time')
                
                allleadtimes = xr.concat(objs = [f.array.swap_dims({'time':'leadtime'}) for f in listofforecasts], dim = 'leadtime') # concatenates over leadtime dimension.
                a,b = unitconversionfactors(xunit = allleadtimes.units, yunit = fieldobs.units)
                exp = allleadtimes.reindex_like(fieldobs, method='nearest') * a + b
                
                temp = exp.drop('time').to_dataframe().unstack(['number'])
                # exp to pandas dataframe with number variable as columns. Ideally also masking.
                # obs as another corresponding column.
                dat = fieldobs.to_dataframe()
                #Do some join operation such that time dimension is kept.
                #temp = temp.assign(obs = dat)
        #pointer = xr.open_mfdataset()

obs = SurfaceObservations(alias = 'rr')
obs.load(tmin = '1995-05-14', tmax = '1995-06-02')
#obs.aggregatetime(freq = 'w', method = 'mean') 

test = ForecastToObsAlignment(season = 'JJA', observations=obs)
test.find_forecasts()
test.load_forecasts(n_members=11)

#test2 = SurfaceObservations(alias = 'rr', tmin = '1950-01-01', tmax = '1990-01-01', timemethod = 'M_mean')
#test2.load()
        
class Comparison(object):
    
    def __init__(self, alignedobject):
        """
        Observation | members | 
        """
    
    def post_process():
        """
        jpsjods
        """
    
    def setup_cv():
        """
        okeedk
        """
    
    def score():
        """
        Check requirements for the forecast horizon of Buizza.
        """
    
        