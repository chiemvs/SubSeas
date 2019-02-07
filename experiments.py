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
obsrr1day = SurfaceObservations('rr')
obsrr1day.load(tmin = '1995-01-01', tmax = '2000-12-31', llcrnr = (25,-30), rucrnr = (75,75))
classpop1day = EventClassification(obs = obsrr1day)
classpop1day.pop()

# Make a separate climatology
classpop1day.localclim(daysbefore=5, daysafter = 5)
#classpop1day.clim
#classpop1day.obs.array

# Now match this to forecasts
alignment = ForecastToObsAlignment(season = 'DJF', observations = classpop1day.obs, cycle='41r1')
alignment.find_forecasts()
alignment.load_forecasts(n_members=11) # With the 'rr' basevariable
alignment.match_and_write(newvariable = True)
alignment.recollect() # booksname = books_pop_DJF_41r1_1D_0.25_degrees.csv

# Now to scoring
final = Comparison(alignedobject=alignment.alignedobject, climatology=classpop1day.clim) # in this case climatology is a probability
test = final.brierscore()
