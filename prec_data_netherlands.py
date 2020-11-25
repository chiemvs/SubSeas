#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to prepare daily precipitation forecasts from the ECMWF extended system cycle 45r1
- First attempt is a quick extraction of point forecasts at the 0.38x0.38 gridcell closest to Amsterdam
- Second attempt is extraction of the average spatial precipitation in cluster denoting the netherlands and matching that to the average E-OBS measurement in that same cluster
"""

import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
from forecasts import Forecast

"""
1
"""
amsterdam_location = (52.379778,4.903522) # lat, lon in decimal degrees
cycle = '45r1'
datadir = Path('/nobackup/users/straaten/EXT') / cycle 

hindcasts = datadir.glob('hin*_processed.nc') # Leaving out forecasts with 51 members for now  
available_hindcasts = [Forecast(indate = path.parts[-1][4:14], prefix = path.parts[-1][:4], cycle = cycle) for path in hindcasts] # dates and prefx are captured by strings slicing of the paths

starting_dates = pd.DatetimeIndex([hindcast.indate for hindcast in available_hindcasts], name = 'time') 
starting_dates = starting_dates.sort_values()

collected_data = [None] * len(starting_dates) # Correct length so we can insert at the correct place
for hindcast in available_hindcasts:
    hindcast.load(variable = 'rr', llcrnr = tuple(np.array(amsterdam_location) - 0.5), rucrnr = tuple(np.array(amsterdam_location) + 0.5)) # Loading a small square of the data, all 46 days
    array = hindcast.array.sel(latitude = amsterdam_location[0], longitude = amsterdam_location[1], method = 'nearest').swap_dims({'time':'leadtime'}).drop('time') # Discarding the time coordinate because we want to concatenate over starting date, not the validation time that runs along with leadtime
    collected_data[starting_dates.get_loc(hindcast.indate)] = array.load() # place at the correct sorted location

allforecasts = xr.concat(collected_data, dim = starting_dates)

"""
2
"""
