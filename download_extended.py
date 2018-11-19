#!/usr/bin/env python
import ecmwfapi 
#from ecmwfapi import ECMWFDataServer
#from datetime import date,timedelta
import os
#import gzip
import numpy as np
#import sys
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import xarray as xr
import pygrib

# Extended range has 50 perturbed members. and 1 control forecast. Steps: "0/to/1104/by/6"
# Reforecasts have 11 members (stream = "enfh", number "1/to/10"). Probably some duplicates will arise.
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
    pfname = 'for_'+indate+'_ens.nc'
    cfname = 'for_'+indate+'_contr.nc'
    
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
    initially downloaded together, which is only possible in a GRIB file. 
    The single file is saved under indate but recognizable by 'hin'. 
    Separate download of perturbed (11 members) and control. 
    Then the hdates within a file are extracted and perturbed and control are joined.
    The final files are saved per hdate as netcdf.
    """

    end = pd.to_datetime(indate, format='%Y-%m-%d')
    # easier arithmetics when in pandas format, then to string for the mars format
    hdates = [(end - pd.DateOffset(years = x)).strftime('%Y-%m-%d') for x in range(1,21)]
    hdates.sort()
    marshdate = '/'.join(hdates)
    
    pfname = 'hin_'+indate+'_ens.grib'
    cfname = 'hin_'+indate+'_contr.grib'
    #combname = 'hin_'+hadate+'_comb.nc' # needs to be defined within the loop
    
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
        
    # here loop to pull hdates in one file out of each other. Conversion to netcdf with 3 parameters.
    # Within grib file: first per variable, than per member
    pf = pygrib.open(basedir + pfname)
    cf = pygrib.open(basedir + cfname)
    params = list(set([x.cfVarName for x in cf.select()])) # unique parameter values ["167.128","121.128","228.128"] # Hardcoded for marsParam
    steprange = np.arange(0,1110,6) # Hardcoded
    for hd in hdates:
        collectparams = list()
        for param in params:
            collectsteps = list()
            for step in steprange: 
                # Get control values and lats, lons, units and longname of control
                control = cf.select(cfVarName = param, validDate = pd.to_datetime(hd), stepRange = str(step))[0] # for hours use .stepRange I think.
                controlval = control.values
                lats,lons = control.latlons()
                # get members values
                members = pf.select(cfVarName = param, validDate = pd.to_datetime(hd), stepRange = str(step)) #
                membersnum = [member.perturbationNumber for member in members] # 1 to 10, membernumers
                membersval = [member.values for member in members]
                # join both along the 'number' dimension by prepending the membernumbers (control = 0) and members values. Make xarray
                membersnum.insert(0,0)
                membersval.insert(0, controlval)
                result = xr.DataArray(data = np.stack(membersval), 
                                      coords=[membersnum, lats[:,0], lons[0,:]], 
                                      dims=['number', 'latitude', 'longitude'])
                collectsteps.append(result)
            
            # Combine along a time dimension, give parameter name 'test.cfVarName' and longname. Give units.
            oneparam = xr.concat(collectsteps, 'time') 
            collectparams.append(oneparam)

        xr.Dataset(collectparams)
        # Save the dataset to netcdf
    pf.close()
    cf.close()
        

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
    'expect'    : "any",
    }
    if hdate is not None:
        req['hdate'] = hdate
    else:
        req['format'] = "netcdf"
    return(req)
        

def join_members(pfname, cfname):
    """
    Also seperate variables?
    """
    pf = xr.open_dataset(basedir + pfname)
    cf = xr.open_dataset(basedir + cfname)
    cf.expand_dims('number')
    return()

# TODO: Function for combining control and perturbed? De-accumulation of precipitation?
# De-accumulation 
# np.r_[cum[0], np.diff(cum)]
# xr.diff()
    
#start_batch(tmin = "2015-05-15", tmax = '2015-05-24')
#start_batch(tmin = '2015-05-24', tmax = '2015-05-29')
