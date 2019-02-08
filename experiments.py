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
from observations import SurfaceObservations, EventClassification
#from forecasts import Forecast
from comparison import ForecastToObsAlignment, Comparison
import itertools

"""
Mean temperature benchmarks.
"""
experiment = 'test1' # not used yet for directory based writing.
season = 'JJA'
cycle = '41r1'
basevar = 'tx'
method = 'max'
timeaggregations = ['1D', '2D', '3D', '4D', '1w'] # Make this smoother? daily steps?
spaceaggregations = [0.25, 0.75, 1.5, 3] # In degrees, in case of None, we take the raw res 0.25 of obs and raw res 0.38 of forecasts. Space aggregation cannot deal with minimum number of cells yet.
experiment_log = pd.DataFrame(data = '', index = pd.MultiIndex.from_product([spaceaggregations, timeaggregations], names = ['spaceagg','timeagg']), columns = ['booksname'])

for spaceagg, timeagg in itertools.product(spaceaggregations, timeaggregations):
    
    obs = SurfaceObservations(basevar)
    obs.load(tmin = '1995-05-30', tmax = '2000-08-31', llcrnr = (25,-30), rucrnr = (75,75))
    if timeagg != '1D':
        obs.aggregatetime(freq = timeagg, method = method)
    if spaceagg != 0.25:
        obs.aggregatespace(step = spaceagg, method = method, by_degree = True)
    obs.savechanges()
    
    alignment = ForecastToObsAlignment(season = season, observations=obs, cycle=cycle)
    alignment.find_forecasts()
    alignment.load_forecasts(n_members = 11)
    alignment.force_resolution(time = (timeagg != '1D'), space = (spaceagg != 0.25))
    alignment.match_and_write()
    experiment_log.loc[(spaceagg, timeagg),'booksname'] = alignment.books_name

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
