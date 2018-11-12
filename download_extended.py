#!/usr/bin/env python
import ecmwfapi 
#from ecmwfapi import ECMWFDataServer
#from datetime import date,timedelta
import os
#import gzip
#import numpy as np
#import sys
import pandas as pd

server = ecmwfapi.ECMWFService("mars")

# Extended range has 50 perturbed members. and 1 control forecast. Steps: "0/to/1104/by/6"
# Reforecasts have 11 members (stream = "enfh", number "1/to/10")
# Params: 228.128 : total precipitation (or 228172). 167.128 : T2m
# Interpolated resolution 0.4 degree. Native grid O320


# Initialization dates.
#weekdays = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
#mondays = pd.date_range('2000-09-01','2018-10-01', freq='W-MON')
#thursdays = pd.date_range('2018-09-01','2018-10-01', freq='W-THU')
#dr = mondays.append(thursdays).sort_values()

indate = '2015-05-14'
basedir = '/nobackup/users/straaten/EXT/'
#filename1 = 'ecmwf_extfc_'+date+'_ens.nc'
#filename2 = 'ecmwf_extfc_'+date+'_control.nc'

#for dt in dr:
#    filena = 'jatnje_' + date + '.nc'
#    if filename exists
#    date = dt.strftime('%Y-%m-%d')

def mars_dict(date, hdate = None, contr = False):
    """
    Generates the appropriate mars request dictionary. This is the place to set parameters.
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
    'grid'      : "O320", #".5/.5"
    'format'    : "netcdf",
    }
    if hdate is not None:
        req['hdate'] = hdate
    return(req)
    
    
def download_forecast(indate):
    """
    Separate perturbed (50 members) and control. Saves under indate
    """
    pfname = 'for'+indate+'_ens.nc'
    cfname = 'for'+indate+'_contr.nc'
    
    if os.path.isfile(pfname):
        print('file already downloaded..')
    # perturbed.
    server.execute(mars_dict(indate, contr = False), basedir+pfname)
    # control
    server.execute(mars_dict(indate, contr = True), basedir+cfname)
    
def download_hindcast(indate):
    """
    Separate perturbed (11 members) and control. Constructs hdates belonging to indate. Saves under hdate.
    """

    end = pd.to_datetime(indate, format='%Y-%m-%d')
    hdates = [end - pd.DateOffset(years = x) for x in range(1,21)]
    
    for hstamp in hdates:
        hdate = hstamp.strftime('%Y-%m-%d')
        pfname = 'hind'+hdate+'_ens.nc'
        cfname = 'hind'+hdate+'_contr.nc'
        
        #print(mars_dict(indate, hdate = hdate, contr = False))
        #print(mars_dict(indate, hdate = hdate, contr = True))
        server.execute(mars_dict(indate, hdate = hdate, contr = False), basedir+pfname)
        # control
        server.execute(mars_dict(indate, hdate = hdate, contr = True), basedir+cfname)

# Small overnight test of the O320 grid
download_forecast(indate = indate)

# TODO: Function for combining control and perturbed? De-accumulation of precipitation?
# De-accumulation 
# np.r_[cum[0], np.diff(cum)]

# TODO: File existence check. No overwriting.


