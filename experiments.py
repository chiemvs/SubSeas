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


# 4 years summer temperatures 1995-1998. In Forecast domain. Climatology of +- 5days of 30 years.
#quantile = 0.9
obs1day = SurfaceObservations('tg')
obs1day.load(tmin = '1970-05-30', tmax = '2000-05-30', llcrnr = (25,-30), rucrnr = (75,75))
#windowed = EventClassification(obs=obs1day)
#windowed.localclim(daysbefore=5, daysafter=5, mean = False, quant=quantile)

alignment = ForecastToObsAlignment(season = 'JJA', observations=obs1day)
alignment.recollect(booksname='books_tg_JJA.csv')
#subset = dd.read_hdf('/nobackup/users/straaten/match/tg_JJA_badf363636004a808a701f250175131d.h5', key = 'intermediate')
temp = xr.open_dataarray('/nobackup/users/straaten/E-OBS/climatologyQ09.nc')
self = Comparison(alignedobject=alignment.alignedobject, climatology= temp)
test = self.brierscore(exceedquantile=True, groupers=['leadtime', 'latitude', 'longitude'])

# self.frame.groupby(self.frame['time'].dt.dayofyear)
# subset.compute()['time'].dt.dayofyear
# self.frame['time'].dt.dayofyear
#tg_JJA_badf363636004a808a701f250175131d.h5
