#!/usr/bin/env python
import ecmwfapi 
#from ecmwfapi import ECMWFDataServer
#from datetime import date,timedelta
import os
#import gzip
#import numpy as np
#import sys
import pandas as pd

# Extended range has 50 perturbed members. and 1 control forecast. Steps: "0/to/1104/by/6"
# Reforecasts have 11 members (stream = "enfh", number "1/to/10")
# Params: 228.128 : total precipitation (or 228172). 167.128 : T2m
# Interpolated resolution 0.4 or 0.5 degree. Native grid O320 (but subsetting is not available for native)

# Global variables
server = ecmwfapi.ECMWFService("mars") # Find a way to let requests run in parallel?
basedir = '/nobackup/users/straaten/EXT/'


def start_batch(tmin = '2015-05-14', tmax = '2015-05-14'):
    """
    Every monday and thursday an operational run is initialized, and associated hindcasts are saved for that date minus 20 years.
    For a batch of initialization days the operational members and the hindcast members are downloaded.
    """
    # Initialization dates.
    mondays = pd.date_range(tmin,tmax, freq='W-MON')
    thursdays = pd.date_range(tmin,tmax, freq='W-THU')
    dr = mondays.append(thursdays).sort_values()
    
    for indate in dr:
        download_forecast(indate = indate.strftime('%Y-%m-%d'))
        download_hindcast(indate = indate.strftime('%Y-%m-%d'))

def download_forecast(indate):
    """
    Separate download of perturbed (50 members) and control members. Then both are joined into one final file. Saves under indate.
    """
    pfname = 'for'+indate+'_ens.nc'
    cfname = 'for'+indate+'_contr.nc'
    
    # perturbed.
    if os.path.isfile(basedir+pfname):
        print('perturbed forecast already downloaded..')
    else:
        server.execute(mars_dict(indate, contr = False), basedir+pfname)
    # control
    if os.path.isfile(basedir + cfname):
        print('control forecast already downloaded..')
    else:
        server.execute(mars_dict(indate, contr = True), basedir+cfname)
    
def download_hindcast(indate):
    """
    Constructs hdates belonging to indate. For retrieval efficiency all Hdates are
    initially downloaded together and saved under indate but recognizable by 'hind'. 
    Then the files are separated. Separate download of perturbed (11 members) and control. 
    Both are joined into one final file. 
    """

    end = pd.to_datetime(indate, format='%Y-%m-%d')
    # easier arithmetics when in pandas format, then to string for the mars format
    hdates = [(end - pd.DateOffset(years = x)).strftime('%Y-%m-%d') for x in range(1,21)]
    hdates.sort()
    marshdate = '/'.join(hdates)
    
    pfname = 'hind'+indate+'_ens.nc'
    cfname = 'hind'+indate+'_contr.nc'
    
    # perturbed.
    if os.path.isfile(basedir+pfname):
        print('perturbed hindcast already downloaded..')
    else:
        server.execute(mars_dict(indate, hdate = marshdate, contr = False), basedir+pfname)
    # control
    if os.path.isfile(basedir + cfname):
        print('control hindcast already downloaded..')
    else:
        server.execute(mars_dict(indate, hdate = marshdate, contr = True), basedir+cfname)
        
    # here loop to pull hdates in one file out of each other.

def mars_dict(date, hdate = None, contr = False):
    """
    Generates the appropriate mars request dictionary. This is the place to set parameters. Called from within the download functions
    """
    req = {
    'stream'    : "enfo" if hdate is None else "enfh",
    'number'    : "0" if contr else "1/to/50" if hdate is None else "1/to/11",
    'class'     : "od",
    'expver'    : "0001",
    'date'      : date,
    'time'      : "00",
    'type'      : "cf" if contr else "pf",
    'levtype'   : "sfc",
    'param'     : "167.128/121.128/228.128", # T2M (Kelvin), Tmax in last 6 hrs. and Tot prec. Tot prec needs de-accumulation
    'step'      : "0/to/1104/by/6",
    'area'      : "75/-30/25/75", #E-OBS 75.375/-40.375/25.375/75.375/ Limits 75/-40.5/25/75.5
    'grid'      : ".5/.5", # Octahedral grid does not support sub-areas
    'format'    : "netcdf",
    'expect'    : "any",
    }
    if hdate is not None:
        req['hdate'] = hdate
    return(req)
        

def join_members(pfname, cfname):
    """
    Also seperate variables?
    """
    return()

# TODO: Function for combining control and perturbed? De-accumulation of precipitation?
# De-accumulation 
# np.r_[cum[0], np.diff(cum)]
# xr.diff()
    
start_batch()

