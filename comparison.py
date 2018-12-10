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
            self.forecasts.update({date : forecasts.append(hindcasts)})   
        
    def force_resolution(self):
        """
        Check if the same resolution and force spatial/temporal aggregation if that is not the case.
        """
        
    def load_and_match(self, n_members):
        """
        Neirest neighbouring to match pairs.
        Creates the dataset. Possibly writes to disk too.
        """
        #pointer = xr.open_mfdataset()

obs = SurfaceObservations(alias = 'rr')
obs.load(tmax = '1950-05-03')

test = ForecastToObsAlignment(season = 'JJA', observations=obs)

test2 = SurfaceObservations(alias = 'rr', tmin = '1950-01-01', tmax = '1990-01-01', timemethod = 'M_mean')
test2.load()
        
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
    
        