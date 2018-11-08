#!/usr/bin/env python
from ecmwfapi import *
#from ecmwfapi import ECMWFDataServer
from datetime import date,timedelta
import os
import gzip
import numpy as np
import sys
#import pandas as pd

server = ECMWFService("mars")

weekdays = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
#dr = pd.date_range('2018-09-01','2018-10-01'.freq='D')

date = (date.today()- timedelta(1)).isoformat()     # get yesterday

#date = '2018-08-13'
bd = '/nobackup/users/krikken/neerslagtekort_s5/ecmwf_data/'
filename1 = 'ecmwf_extfc_'+date+'_ens.nc'
filename2 = 'ecmwf_extfc_'+date+'_control.nc'

#for dt in dr:
#    filena = 'jatnje_' + date + '.nc'
#    if filename exists
#    date = dt.strftime('%Y-%m-%d')

if os.path.isfile(filename1):
    print 'file already downloaden..'
else:
    print date
    #date = '2018-08-09'
    server.execute({
        'stream'    : "enfo",
        'number'    : "1/to/50",
        'class'     : "od",
        'expver'    : "0001",
        'date'      : date,
        'time'      : "00",
        'type'      : "pf",
        'levtype'   : "sfc",
        'param'     : "167.128/169.128/228.128", # T2M and SSR
        #'param'     : "182.128/251.228",        # EVAP and POT_EVAP
        'step'      : "0/to/1104/by/6",
        'area'      : "54/3/46/12",
        'grid'      : ".5/.5",
        'format'    : "netcdf",
        },
        bd+filename1)

    server.execute({
        'stream'    : "enfo",
        'number'    : "0",
        'class'     : "od",
        'expver'    : "0001",
        'date'      : date,
        'time'      : "00",
        'type'      : "cf",
        'levtype'   : "sfc",
        'param'     : "167.128/169.128/228.128", # T2M and SSR
        #'param'     : "182.128/251.228",    # EVAP and POT_EVAP
        'step'      : "0/to/1104/by/6",
        'area'      : "54/3/46/12",
        'grid'      : ".5/.5",
        'format'    : "netcdf",
        },
        bd+filename2)
   
#if os.path.isfile(filename1):
#    os.system('python3 /nobackup/users/krikken/neerslagtekort_s5/#calc_plot_neerslagtekort.py')

    
