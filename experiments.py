#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 12:11:08 2019

@author: straaten
"""

import os
import numpy as np
import xarray as xr
import pandas as pd
import dask.dataframe as dd
from observations import SurfaceObservations, Climatology, EventClassification
#from forecasts import Forecast
from comparison import ForecastToObsAlignment, Comparison
import itertools

class Experiment(object):
    
    def __init__(self, expname, basevar, cycle, season, method = 'mean', timeaggregations = ['1D', '2D', '3D', '4D', '7D'], spaceaggregations = [0.25, 0.75, 1.5, 3], quantiles = [0.5, 0.9, 0.95]):
        """
        Setting the relevant attributes. Timeaggregations are pandas frequency strings, spaceaggregations are floats in degrees.
        """
        self.resultsdir = '/nobackup/users/straaten/results/'
        self.expname = expname
        self.basevar = basevar
        self.cycle = cycle
        self.season = season
        self.method = method
        self.timeaggregations = timeaggregations
        self.spaceaggregations = spaceaggregations
        self.quantiles = quantiles
    
    def setuplog(self):
        """
        Load an experiment log if it is present. Otherwise create one.
        """
        self.logpath = self.resultsdir + expname + '.h5'
        try:
            self.log = pd.read_hdf(self.logpath, key = 'exp')
        except OSError:
            self.log = pd.DataFrame(data = object, index = pd.MultiIndex.from_product([self.spaceaggregations, self.timeaggregations, self.quantiles], names = ['spaceagg','timeagg','quantile']), columns = ['climname','scores'])
            self.log = self.log.unstack(level = -1)
            self.log = self.log.assign(**{'obsname':'','booksname':''})
    
    def iterateaggregations(self, func, column = None, kwargs = {}):
        """
        Wrapper that calls the function with a specific spaceagg and timeagg and writes the returns of the function to the column if a name was given
        """
        for spaceagg, timeagg in itertools.product(self.spaceaggregations,self.timeaggregations):
            f = getattr(self, func)
            ret = f(spaceagg, timeagg, **kwargs)
            if (ret is not None) and (column is not None):
                self.log.loc[(spaceagg, timeagg),column] = ret
    
    def prepareobs(self, spaceagg, timeagg, tmin, tmax):
        """
        Writes the observations that will be matched to forecasts
        """
        obs = SurfaceObservations(self.basevar)
        obs.load(tmin = tmin, tmax = tmax, llcrnr = (25,-30), rucrnr = (75,75))
        if timeagg != '1D':
            obs.aggregatetime(freq = timeagg, method = self.method)
        if spaceagg != 0.25:
            obs.aggregatespace(step = spaceagg, method = self.method, by_degree = True)
        obs.savechanges()
        return(obs.name)
    
    def match(self, spaceagg, timeagg):
        """
        Writes the intermediate files. And returns the (possibly appended) booksname
        """
        obs = SurfaceObservations(self.basevar, **{'name':self.log.loc[(spaceagg, timeagg),'obsname']})
        obs.load()
        alignment = ForecastToObsAlignment(season = self.season, observations=obs, cycle=self.cycle)
        alignment.find_forecasts()
        alignment.load_forecasts(n_members = 11)
        alignment.force_resolution(time = (timeagg != '1D'), space = (spaceagg != 0.25))
        alignment.match_and_write()
        return(alignment.books_name)
    
    def makeclim(self, spaceagg, timeagg, climtmin, climtmax):
        """
        Make climatologies based on a period of 30 years, longer than the 5 years in matching. Should daysbefore/daysafter be an attribute of the class?
        """
        obs = SurfaceObservations(self.basevar)
        obs.load(tmin = climtmin, tmax = climtmax, llcrnr = (25,-30), rucrnr = (75,75))
        dailyobs = SurfaceObservations(self.basevar)
        dailyobs.load(tmin = climtmin, tmax = climtmax, llcrnr = (25,-30), rucrnr = (75,75))
        if timeagg != '1D':
            obs.aggregatetime(freq = timeagg, method = self.method)
        if spaceagg != 0.25:
            obs.aggregatespace(step = spaceagg, method = self.method, by_degree = True)
            dailyobs.aggregatespace(step = spaceagg, method = self.method, by_degree = True) # Aggregate the dailyobs for the climatology
        
        climnames = np.repeat('',len(self.quantiles))
        for quantile in self.quantiles:
            climatology = Climatology(self.basevar)
            climatology.localclim(obs = obs, daysbefore = 5, daysafter=5, mean = False, quant = quantile, daily_obs_array = dailyobs.array)
            climatology.savelocalclim()
            climnames[self.quantiles.index(quantile)] = climatology.name
        
        return(climnames)
            
    def score(self, spaceagg, timeagg)
        """
        Read the obs and clim. make a comparison object which computes the scores in the dask dataframe. Returns a list of dataframes with raw and climatological scores. Perhaps write intermediate scores to a file? Distributed computation can take very long.
        """
        alignment = ForecastToObsAlignment(season = self.season, cycle=self.cycle)
        alignment.recollect(booksname = experiment_log.loc[(spaceagg, timeagg),'booksname'])
        
        result = []
        for quantile in self.quantiles:    
            climatology = Climatology(self.basevar, **{'name':self.log.loc[(spaceagg, timeagg),('climname', quantile)]})
            climatology.localclim() # loading in this case. Creation was done in the makeclim method.
            comp = Comparison(alignedobject= alignment.alignedobject, climatology = climatology.clim)
            scores = comp.brierscore(groupers=['latitude','longitude'])
            
            # rename the columns
            scores.columns = scores.columns.droplevel(1)
            result.append(scores)
        
        return(result)

"""
Mean temperature benchmarks.
"""
# Calling of the class        
test1 = Experiment(expname = 'test1', basevar = 'tx', cycle = '41r1', season = 'JJA', method = 'max', timeaggregations = ['1D', '2D', '3D', '4D', '7D'], spaceaggregations = [0.25, 0.75, 1.5, 3], quantiles = [0.5, 0.9, 0.95])
test1.setuplog()
test.iterateaggregations(func = 'prepareobs', column = 'obsname', kwargs = {'tmin':'1996-05-30','tmax':'2006-08-31'}
test.iterateaggregations(func = 'match', column = 'booksname')
test.iterateaggregations(func = 'makeclim', column = 'climname', kwargs = {'climtmin':'1980-05-30','climtmax':'2010-08-31'})
test.iterateaggregations(func = 'score', column = 'scores')
        
resultsdir = '/nobackup/users/straaten/results/'
experiment = 'test1' # not used yet for directory based writing.
season = 'JJA'
cycle = '41r1'
basevar = 'tx'
method = 'max'
timeaggregations = ['1D', '2D', '3D', '4D', '7D'] # Make this smoother? daily steps?
spaceaggregations = [0.25, 0.75, 1.5, 3] # In degrees, in case of None, we take the raw res 0.25 of obs and raw res 0.38 of forecasts. Space aggregation cannot deal with minimum number of cells yet.
experiment_log = pd.DataFrame(data = '', index = pd.MultiIndex.from_product([spaceaggregations, timeaggregations], names = ['spaceagg','timeagg']), columns = ['obsname','booksname'])
quantiles = [0.1, 0.5, 0.9]

# Writing of the files.
for spaceagg, timeagg in []: #itertools.product(spaceaggregations, timeaggregations):
    
    obs = SurfaceObservations(basevar)
    obs.load(tmin = '1996-05-30', tmax = '2006-08-31', llcrnr = (25,-30), rucrnr = (75,75))
    if timeagg != '1D':
        obs.aggregatetime(freq = timeagg, method = method)
    if spaceagg != 0.25:
        obs.aggregatespace(step = spaceagg, method = method, by_degree = True)
    obs.savechanges()
    experiment_log.loc[(spaceagg, timeagg),'obsname'] = obs.name
    
    alignment = ForecastToObsAlignment(season = season, observations=obs, cycle=cycle)
    alignment.find_forecasts()
    alignment.load_forecasts(n_members = 11)
    alignment.force_resolution(time = (timeagg != '1D'), space = (spaceagg != 0.25))
    alignment.match_and_write()
    experiment_log.loc[(spaceagg, timeagg),'booksname'] = alignment.books_name
    

experiment_log.to_hdf(resultsdir + experiment + '.h5', key = 'exp')

#experiment_log = pd.read_hdf(resultsdir + experiment + '.h5', key = 'exp')

# Reading and scoring of the files. Make climatologies based on a period of 30 years, longer than the 5 years in matching.
for spaceagg, timeagg in []: #itertools.product(spaceaggregations, timeaggregations):
    
    obs = SurfaceObservations(basevar)
    obs.load(tmin = '1980-05-30', tmax = '2010-08-31', llcrnr = (25,-30), rucrnr = (75,75))
    dailyobs = SurfaceObservations(basevar)
    dailyobs.load(tmin = '1980-05-30', tmax = '2010-08-31', llcrnr = (25,-30), rucrnr = (75,75))
    if timeagg != '1D':
        obs.aggregatetime(freq = timeagg, method = method)
    if spaceagg != 0.25:
        obs.aggregatespace(step = spaceagg, method = method, by_degree = True)
        dailyobs.aggregatespace(step = spaceagg, method = method, by_degree = True) # Aggregate the dailyobs for the climatology
    
    for quantile in quantiles:
        climatology = Climatology(basevar)
        climatology.localclim(obs = obs, daysbefore = 5, daysafter=5, mean = False, quant = quantile, daily_obs_array = dailyobs.array)
        climatology.savelocalclim()
        alignment = ForecastToObsAlignment(season = season, observations=obs, cycle=cycle)
        alignment.recollect(booksname = experiment_log.loc[(spaceagg, timeagg),'booksname'])
        test = Comparison(alignedobject= alignment.alignedobject, climatology = climatology.clim)
        scores = test.brierscore(groupers=['latitude','longitude'])
        
        # rename the columns
        scores.columns = scores.columns.droplevel(1)
        
# books_tx_JJA_41r1_2D_max_0.25_degrees.csv
#alignment.recollect(booksname = 'books_tx_JJA_41r1_2D_max_0.25_degrees.csv')
#quantiles = [0.1,0.5,0.9] # quantile loop comes after the file writing. Just constructing different climatologies.

"""
4 years summer temperatures 1995-1998. In Forecast domain. 
match forecasts on .38 with obs on .25. Climatology of +- 5days of 30 years.
"""
#quantile = 0.9
#obs1day = SurfaceObservations('tg')
#obs1day.load(tmin = '1970-05-30', tmax = '2000-05-30', llcrnr = (25,-30), rucrnr = (75,75))
#windowed = EventClassification(obs=obs1day)
#windowed.localclim(daysbefore=5, daysafter=5, mean = False, quant=quantile)

#alignment = ForecastToObsAlignment(season = 'JJA', observations=obs1day)
# alignment.force_resolution(time = True, space = False)
#alignment.recollect(booksname='books_tg_JJA.csv')
#subset = dd.read_hdf('/nobackup/users/straaten/match/tg_JJA_badf363636004a808a701f250175131d.h5', key = 'intermediate')
#temp = xr.open_dataarray('/nobackup/users/straaten/E-OBS/climatologyQ09.nc')
#self = Comparison(alignedobject=alignment.alignedobject, climatology= temp)
#test = self.brierscore(exceedquantile=True, groupers=['leadtime', 'latitude', 'longitude'])

# self.frame.groupby(self.frame['time'].dt.dayofyear)
# subset.compute()['time'].dt.dayofyear
# self.frame['time'].dt.dayofyear
#tg_JJA_badf363636004a808a701f250175131d.h5


"""
Probability of precipitation matching.
"""
#obsrr1day = SurfaceObservations('rr')
#obsrr1day.load(tmin = '1995-01-01', tmax = '2000-12-31', llcrnr = (25,-30), rucrnr = (75,75))
#classpop1day = EventClassification(obs = obsrr1day)
#classpop1day.pop()

# Make a separate climatology
#classpop1day.localclim(daysbefore=5, daysafter = 5)
#classpop1day.clim
#classpop1day.obs.array

# Now match this to forecasts
#alignment = ForecastToObsAlignment(season = 'DJF', observations = classpop1day.obs, cycle='41r1')
#alignment.find_forecasts()
#alignment.load_forecasts(n_members=11) # With the 'rr' basevariable
#alignment.match_and_write(newvariable = True)
#alignment.recollect() # booksname = books_pop_DJF_41r1_1D_0.25_degrees.csv

# Now to scoring
#final = Comparison(alignedobject=alignment.alignedobject, climatology=classpop1day.clim) # in this case climatology is a probability
#test = final.brierscore()
