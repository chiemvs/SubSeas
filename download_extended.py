#!/usr/bin/env python
import ecmwfapi 
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
server = ecmwfapi.ECMWFService("mars") # Setup parallel requests by splitting the batches in multiple consoles. (total: max 3 active and 20 queued requests allowed)
basedir = '/nobackup/users/straaten/EXT/'
netcdf_encoding = {'t2m': {'dtype': 'int16', 'scale_factor': 0.0015, 'add_offset': 283, '_FillValue': -32767},
                   'mx2t6': {'dtype': 'int16', 'scale_factor': 0.0015, 'add_offset': 283, '_FillValue': -32767},
                   'tp': {'dtype': 'int16', 'scale_factor': 0.00005, '_FillValue': -32767},
                   'time': {'dtype': 'int64'},
                   'latitude': {'dtype': 'float32'},
                   'longitude': {'dtype': 'float32'},
                   'number': {'dtype': 'int16'}}

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
    
    svname = 'for_' + indate + '_comb.nc'
    if os.path.isfile(basedir+svname):
        print('forecasts already joined')
    else:
        join_members(pfname = pfname, cfname = cfname, svname = svname)
    
def download_hindcast(indate):
    """
    Constructs hdates belonging to indate. For retrieval efficiency all Hdates are
    initially downloaded together, which is only possible in a GRIB file. 
    The single file is saved under indate but recognizable by 'hin'. 
    Separate download of perturbed (11 members) and control. 
    """

    end = pd.to_datetime(indate, format='%Y-%m-%d')
    # easier arithmetics when in pandas format, then to string for the mars format
    hdates = [(end - pd.DateOffset(years = x)).strftime('%Y-%m-%d') for x in range(1,21)]
    hdates.sort()
    marshdate = '/'.join(hdates)
    
    pfname = 'hin_'+indate+'_ens.grib'
    cfname = 'hin_'+indate+'_contr.grib'

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
    
    # Do seperate hdate files exist? If not: start grib file extraction
    svnames = ['hin_' + hd + '_comb.nc' for hd in hdates]
    if all([os.path.isfile(basedir + svname) for svname in svnames]):
        print('hdate files already exist')
    else:
        crunch_gribfiles(pfname = pfname, cfname = cfname, hdates = hdates, svnames = svnames)

def mars_dict(date, hdate = None, contr = False):
    """
    Generates the appropriate mars request dictionary. This is the place to set parameters. Called from within the download functions
    """
    req = {
    'stream'    : "enfo" if hdate is None else "enfh",
    'number'    : "0" if contr else "1/to/50" if hdate is None else "1/to/10",
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
        
def crunch_gribfiles(pfname, cfname, hdates, svnames):
    """
    hdates within a file are extracted and perturbed and control are joined.
    The final files are saved per hdate as netcdf with three variables.
    svnames should be supplied in a list and have the same length as hdates.
    """
    pf = pygrib.open(basedir + pfname)
    cf = pygrib.open(basedir + cfname)
    params = list(set([x.cfVarName for x in cf.read(100)])) # Enough to get the variables. ["167.128","121.128","228.128"] # Hardcoded for marsParam

    steprange = np.arange(0,1110,6) # Hardcoded as this is too slow: steps = list(set([x.stepRange for x in cf.select()]))
    beginning = steprange[1:-1].tolist()
    beginning[0:0] = [0,0]
    tmaxrange = [ str(b) + '-' + str(e) for b,e in zip(beginning, steprange)] # special hardcoded range for the tmax, whose stepRanges are stored differently
    
    for hd in hdates:
        collectparams = dict()
        for param in params:
            collectsteps = list()
            # search the grib files, special search for tmax
            if param == 'mx2t6':
                steps = tmaxrange # The 0-0 stepRange is missing, replace field with _FillValue?
            else:
                steps = [str(step) for step in steprange]
            
            control = cf.select(validDate = pd.to_datetime(hd), stepRange = steps, cfVarName = param)
            members = pf.select(validDate = pd.to_datetime(hd), stepRange = steps, cfVarName = param) #1.5 min more or less

            lats,lons = control[0].latlons()
            units = control[0].units
            gribmissval = control[0].missingValue
        
            for i in range(0,len(steps)): # use of index because later on the original steprange is needed for timestamps
                cthisstep = [c for c in control if c.stepRange == steps[i]]
                mthisstep = [m for m in members if m.stepRange == steps[i]]
                # If empty lists (tmax analysis) the field in nonexisting and we want fillvalues
                if not cthisstep:
                    print('missing field')
                    controlval = np.full(shape = lats.shape, fill_value = float(gribmissval))
                    membersnum = [m.perturbationNumber for m in members if m.stepRange == steps[i+1]]
                    membersval = [controlval] * len(membersnum)
                else:
                    controlval = cthisstep[0].values
                    membersnum = [m.perturbationNumber for m in mthisstep] # 1 to 10, membernumers
                    membersval = [m.values for m in mthisstep]
                
                # join both members and control along the 'number' dimension by prepending the membernumbers (control = 0) and members values. 
                # Then add timestamp as extra dimension and make xarray
                membersnum.insert(0,0)
                membersval.insert(0, controlval)
                timestamp = [pd.to_datetime(hd) + pd.DateOffset(hours = int(steprange[i]))]
                data = np.expand_dims(np.stack(membersval, axis = -1), axis = 0)
                data[data == float(gribmissval)] = np.nan # Replace grib missing value with numpy missing value
                result = xr.DataArray(data = data, 
                                      coords=[timestamp, lats[:,0], lons[0,:], membersnum], 
                                      dims=['time', 'latitude', 'longitude', 'number'])
                collectsteps.append(result)
            
            # Combine along the time dimension, give parameter name. Give longname and units from control.
            oneparam = xr.concat(collectsteps, 'time')
            oneparam.name = param
            oneparam.attrs.update({'long_name':control[0].name, 'units':units})
            oneparam.longitude.attrs.update({'long_name':'longitude', 'units':'degrees_east'})
            oneparam.latitude.attrs.update({'long_name':'latitude', 'units':'degrees_north'})
            #oneparam.where()
            collectparams.update({param : oneparam})
        # Save the dataset to netcdf under hdate.
        onehdate = xr.Dataset(collectparams)
        svname = svnames[hdates.index(hd)]
        particular_encoding = {key : netcdf_encoding[key] for key in onehdate.keys()} # get only encoding of present variables
        onehdate.to_netcdf(path = basedir + svname, encoding= particular_encoding)
    pf.close()
    cf.close()


def join_members(pfname, cfname, svname):
    """
    Join members save the dataset. Control member gets the number 0
    """
    pf = xr.open_dataset(basedir + pfname)
    cf = xr.open_dataset(basedir + cfname)
    cf.coords['number'] = np.array(0, dtype='int16')
    cf = cf.expand_dims('number',-1)
    particular_encoding = {key : netcdf_encoding[key] for key in cf.keys()} 
    xr.concat([cf,pf], dim = 'number').to_netcdf(path = basedir + svname, encoding= particular_encoding)

# TODO: De-accumulation of precipitation? Or only when extracting 24hr values, because E-OBS has 24hr accumulations.
# De-accumulation 
# np.r_[cum[0], np.diff(cum)]
# xr.diff()
    

start_batch(tmin = "2015-06-02", tmax = '2015-06-23')
#start_batch(tmin = '2015-06-23', tmax = '2015-07-05')
#download_forecast(indate = '2015-06-11')
