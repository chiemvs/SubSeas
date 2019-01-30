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
from forecasts import Forecast
from comparison import Comparison

#quantile = 0.9
obs1day = SurfaceObservations('tg')
obs1day.load(tmin = '1970-05-06', tmax = '1995-05-06')
windowed = EventClassification(obs=obs1day)
windowed.localclim(daysbefore=2, daysafter=2, mean = True)

subset = dd.read_hdf('/usr/people/straaten/Documents/python_tests/subset.h5', key = 'intermediate')
self = Comparison(alignedobject=subset, climatology= windowed.clim)
test = self.brierscore(exceedquantile=True, groupers=['leadtime'])

# self.frame.groupby(self.frame['time'].dt.dayofyear)
# subset.compute()['time'].dt.dayofyear
# self.frame['time'].dt.dayofyear