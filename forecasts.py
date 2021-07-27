#!/usr/bin/env python3
import ecmwfapi 
import os
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import xarray as xr
import eccodes as ec
from helper_functions import unitconversionfactors, agg_space, agg_time
#from observations import Clustering, EventClassification

# Extended range has 50 perturbed members. and 1 control forecast. Steps: "0/to/1104/by/6"
# Reforecasts have 11 members (stream = "enfh", number "1/to/10"). Probably some duplicates will arise.
# Params: 228.128 : total precipitation (or 228172). 167.128 : T2m
# Native grid O320 (but subsetting is not available for native)
# Grids at the ecmwf are stored with decreasing latitude but increasing longitude.

# Global variables
server = ecmwfapi.ECMWFService("mars") # Setup parallel requests by splitting the batches in multiple consoles. (total: max 3 active and 20 queued requests allowed)
for_netcdf_encoding = {'t2m': {'dtype': 'int16', 'scale_factor': 0.002, 'add_offset': 273, '_FillValue': -32767},
                   'mx2t6': {'dtype': 'int16', 'scale_factor': 0.002, 'add_offset': 273, '_FillValue': -32767},
                   'tp': {'dtype': 'int16', 'scale_factor': 0.00005, '_FillValue': -32767},
                   'tpvar': {'dtype': 'int16', 'scale_factor': 0.00005, '_FillValue': -32767},
                   'rr': {'dtype': 'int16', 'scale_factor': 0.00005, '_FillValue': -32767},
                   'tx': {'dtype': 'int16', 'scale_factor': 0.002, 'add_offset': 273, '_FillValue': -32767},
                   'tg': {'dtype': 'int16', 'scale_factor': 0.002, 'add_offset': 273, '_FillValue': -32767},
                   'tg-anom': {'dtype': 'int16', 'scale_factor': 0.002, '_FillValue': -32767},
                   'sst': {'dtype': 'int16', 'scale_factor': 0.002, 'add_offset': 273, '_FillValue': -32767},
                   'swvl1': {'dtype': 'int16', 'scale_factor': 0.01, '_FillValue': -32767},
                   'swvl13': {'dtype': 'int16', 'scale_factor': 0.01, '_FillValue': -32767},
                   'swvl2': {'dtype': 'int16', 'scale_factor': 0.01, '_FillValue': -32767},
                   'swvl3': {'dtype': 'int16', 'scale_factor': 0.01, '_FillValue': -32767},
                   'swvl4': {'dtype': 'int16', 'scale_factor': 0.01, '_FillValue': -32767},
                   'msl': {'dtype': 'int16', 'scale_factor': 0.3, 'add_offset': 99000, '_FillValue': -32767},
                   'z': {'dtype': 'int16', 'scale_factor': 0.6, 'add_offset': 90000, '_FillValue': -32767},
                   'u': {'dtype': 'int16', 'scale_factor': 0.004, '_FillValue': -32767},
                   'v': {'dtype': 'int16', 'scale_factor': 0.004, '_FillValue': -32767},
                   'time': {'dtype': 'int64'},
                   'latitude': {'dtype': 'float32'},
                   'longitude': {'dtype': 'float32'},
                   'number': {'dtype': 'int16'},
                   'clustid': {'dtype': 'int16', '_FillValue': -32767},
                   'clustidfield': {'dtype': 'int16', '_FillValue': -32767},
                   'doy': {'dtype': 'int16'},
                   'leadtime': {'dtype': 'int16'},
                   'dissim_threshold': {'dtype':'float32'}}

model_cycles = pd.DataFrame(data = {'firstday':pd.to_datetime(['2015-05-12','2016-03-08','2016-11-22','2017-07-11','2018-06-05','2019-06-11']),
                             'lastday':pd.to_datetime(['2016-03-07','2016-11-21','2017-07-10','2018-06-04','2019-06-10','']),
                             'cycle':['41r1','41r2','43r1','43r3','45r1','46r1'],
                             'stepbeforeresswitch':[240,360,360,360,360,360]})

def mars_dict_extra_surface(date, hdate = None, contr = False, varres = False, stepbeforeresswitch = None):
    """
    Generates the appropriate mars request dictionary. This is the place to set parameters. Called from within the classes when ensemble or control files do not yet exist and need to be downloaded
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
    'param'     : "34.128/39.128/40.128/41.128/42.128/151.128", # sst, volumetric soil water 1-4, mslp 
    'step'      : "0/to/1104/by/6",
    'ppengine'  : "mir",
    'area'      : "80/-90/20/30", # big domain from Cassou 2005. swvl shrunken later to "75/-30/25/75"
    'grid'      : "1.5/1.5", # Octahedral grid does not support sub-areas
    'expect'    : "any",
    }
    
    if hdate is not None:
        req['hdate'] = hdate
    else:
        req['format'] = "netcdf"
        
    return(req)

def mars_dict_extra_pressure(date, hdate = None, contr = False, varres = False, stepbeforeresswitch = None):
    """
    Generates the appropriate mars request dictionary. This is the place to set parameters. Called from within the classes when ensemble or control files do not yet exist and need to be downloaded
    """

    req = {
    'stream'    : "enfo" if hdate is None else "enfh",
    'number'    : "0" if contr else "1/to/50" if hdate is None else "1/to/10",
    'class'     : "od",
    'expver'    : "0001",
    'date'      : date,
    'time'      : "00",
    'type'      : "cf" if contr else "pf",
    'levtype'   : "pl",
    'levelist'  : "300",
    'param'     : "129.128/131/132", # geopotential, u and v 
    'step'      : "0/to/1104/by/6",
    'ppengine'  : "mir",
    'area'      : "80/-90/20/30", # big domain from Cassou 2005. swvl shrunken later to "75/-30/25/75"
    'grid'      : "1.5/1.5", # Octahedral grid does not support sub-areas
    'expect'    : "any",
    }
    
    if hdate is not None:
        req['hdate'] = hdate
    else:
        req['format'] = "netcdf"
        
    return(req)

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
        cycle = model_cycles.loc[np.logical_and(model_cycles.firstday <= indate, indate <= model_cycles.lastday),'cycle'].values[0]
        forecast = Forecast(indate.strftime('%Y-%m-%d'), prefix = 'for_', cycle = cycle)
        forecast.create_processed()
        forecast.cleanup()
        hindcast = Hindcast(indate.strftime('%Y-%m-%d'), prefix = 'hin_', cycle = cycle)
        hindcast.invoke_processed_creation()
        hindcast.cleanup()

class CascadeError(Exception):
    pass

class CascadePressureError(Exception):
    pass
                       
class Forecast(object):

    def __init__(self, indate = '2015-05-14', prefix = 'for_', cycle = '41r1', basedir = '/nobackup/users/straaten/EXT_extra/'):
        """
        Defines all the intermediate and raw forecast files that are needed for the processing.
        The final usable daily product is 'processedfile'
        _pl files are for the (single) pressure level
        """
        self.cycle = cycle
        self.stepbeforeresswitch = model_cycles.loc[model_cycles['cycle'] == self.cycle, 'stepbeforeresswitch'].values[0]
        self.basedir = basedir + cycle + '/'
        self.prefix = prefix
        self.indate = indate
        self.processedfile = self.prefix + self.indate + '_processed.nc'
        self.interfile = self.prefix + self.indate + '_comb_sfc.nc'
        self.interfile_pl = self.prefix + self.indate + '_comb_pl.nc'
        self.pffile = self.prefix + self.indate + '_ens_sfc.nc'
        self.pffile_pl = self.prefix + self.indate + '_ens_pl.nc'
        self.cffile = self.prefix + self.indate + '_contr_sfc.nc'
        self.cffile_pl = self.prefix + self.indate + '_contr_pl.nc'
    
    def create_processed(self, prevent_cascade = False):
        """
        Tries to load the 6h forecasts with all members and variables in it, and to load the variable res precipitation forecast. 
        If not available these are create by 'join_members' calls.
        Joining of members for the combined file is not needed for (re)forecasts 
        that are extracted form hindcast Grib-files, therefore CascadeError is raised before function call.
        The actual processing does a daily temporal resampling, and left-labels timestamps to match E-OBS variables:
        - tg: Mean of 6h temperatures in the 24 hrs belonging to yyyy-mm-dd (including 00UTC, excluding 00 UTC of next day)
        - tx: Maximum temperature in the 24 hrs belonging to yyyy-mm-dd
        - rr: Precipitation that will accumulate in the 24 hrs belonging to yyyy-mm-dd
        """
        if os.path.isfile(self.basedir+self.processedfile):
            print('Processed forecast already exits. Do nothing')
        else:
            try:
                comb = xr.open_dataset(self.basedir + self.interfile)
                print('Combined file successfully loaded')
            except OSError:
                print('Combined file needs creation')
                if prevent_cascade:
                    raise CascadeError
                self.join_members(pf_in = self.pffile, 
                                  cf_in = self.cffile, 
                                  comb_out = self.interfile) # creates the combined interfile
                comb = xr.open_dataset(self.basedir + self.interfile)
            
            try:
                comb_pl = xr.open_dataset(self.basedir + self.interfile_pl)
                print('Combined pressure file successfully loaded')
            except OSError:
                print('Combined pressure file needs creation')
                if prevent_cascade:
                    raise CascadePressureError
                self.join_members(pf_in = self.pffile_pl,
                                  cf_in = self.cffile_pl,
                                  comb_out = self.interfile_pl) # creates the combined pressure level interfile
                comb_pl = xr.open_dataset(self.basedir + self.interfile_pl)
                
            comb.load()
            comb_pl.load() # Also read as a dataset, can later both be joined. Perhaps pressure level coordinate needs some attention?
             
            # Surface variables resample to daily means
            # Last one can be dropped, only a 00 UTC, should not become an average
            surface = comb.resample(time = 'D').mean().isel(time = slice(0,-1))
            for var in surface.data_vars:
                surface[var].attrs.update({'resample':'1D_mean','units':comb[var].attrs['units']})

            # Pressure variables select 12 UTC as the daily snapshot
            # Also replace axis with the same as surface (for merging)
            pressure = comb_pl.sel(time = comb_pl.time.dt.hour == 12)
            for var in pressure.data_vars:
                pressure[var].attrs.update({'resample':'12UTC','units':comb_pl[var].attrs['units']})
            pressure.coords['time'] = surface.coords['time']
            
            # Join and add leadtime dimension (days) for clarity
            result = surface.merge(pressure)
            result['leadtime'] = ('time', np.arange(1, len(result.coords['time'])+1, dtype = 'int16'))
            result.leadtime.attrs.update({'long_name':'leadtime', 'units':'days'})
            result = result.set_coords('leadtime') # selection by leadtime requires a quick swap: result.swap_dims({'time':'leadtime'})
            
            particular_encoding = {key : for_netcdf_encoding[key] for key in result.keys()} 
            result.to_netcdf(path = self.basedir + self.processedfile, encoding = particular_encoding)
            comb.close()
            comb_pl.close()
            print('Processed forecast successfully created')
            

    def join_members(self, pf_in, cf_in, comb_out):
        """
        Join members of perturbed and control. save the dataset. Control member gets the number 0.
        Only for non-hindcast forecasts.
        Comb_out determines whether surface or pressure level request is made
        """
        if comb_out == self.interfile: # need to download surface params
            call_func = mars_dict_extra_surface
        else: # would equal self.interfile_pl
            call_func = mars_dict_extra_pressure

        try:
            pf = xr.open_dataset(self.basedir + pf_in)
            print('Ensemble file successfully loaded')
        except OSError:
            print('Ensemble file need to be downloaded')
            server.execute(call_func(self.indate, contr = False,
                                     varres = False, stepbeforeresswitch = self.stepbeforeresswitch), self.basedir + pf_in)
            pf = xr.open_dataset(self.basedir + pf_in)
        
        try:
            cf = xr.open_dataset(self.basedir + cf_in)
            print('Control file successfully loaded')
        except OSError:
            print('Control file need to be downloaded')
            server.execute(call_func(self.indate, contr = True,
                                     varres = False, stepbeforeresswitch = self.stepbeforeresswitch), self.basedir + cf_in)
            cf = xr.open_dataset(self.basedir + cf_in)
        
        cf.coords['number'] = np.array(0, dtype='int16')
        cf = cf.expand_dims('number',-1)
        particular_encoding = {key : for_netcdf_encoding[key] for key in cf.keys()} 
        xr.concat([cf,pf], dim = 'number').to_netcdf(path = self.basedir + comb_out, encoding= particular_encoding)
    
    def cleanup(self):
        """
        Remove all files except the processed one and the raw ones.
        """
        for filename in [self.interfile, self.interfile_pl]:
            try:
                os.remove(self.basedir + filename)
            except OSError:
                pass
    
    def load(self, variable = None, tmin = None, tmax = None, n_members = None, llcrnr = (None, None), rucrnr = (None,None)):
        """
        Loading of processedfile. Similar behaviour to observation class
        Default behaviour is to load the full spatial field and whole forecast range.
        Field is stored with decreasing latitude but increasing longitude.
        Variable needs to be supplied. Otherwise no array can be loaded.
        If n_members is set to one it selects only the control member.
        Then the numbers are set to a default range because they are not unique across initialization times
        over which they are later pooled. Within one forecast instance they do of course resemble a physical pathway.
        """
        if variable in ['tg','tx','rr']:
            self.basedir = '/nobackup/users/straaten/EXT/' + self.cycle + '/' # Comptible with loading old forecast from non EXT_extra variables
        full = xr.open_dataset(self.basedir + self.processedfile)[variable]
        
        # Full range if no timelimits were given
        if tmin is None:
            tmin = pd.Series(full.coords['time'].values).min()
        if tmax is None:
            tmax = pd.Series(full.coords['time'].values).max()
            
        # control + random  members when n_members is given. Otherwise all members are loaded.
        numbers = full.coords['number'].values
        if n_members is not None:
            numbers = np.concatenate((numbers[[0]], 
                                      np.random.choice(numbers[1:], size = n_members - 1, replace = False)))
        
        self.array = full.sel(time = pd.date_range(tmin, tmax, freq = 'D'), number = numbers, longitude = slice(llcrnr[1], rucrnr[1]), latitude = slice(rucrnr[0], llcrnr[0]))
        # reset the index
        if n_members is not None:
            self.array.coords['number'] = np.arange(0,n_members, dtype = 'int16')
        # Standard methods of the processed files.
        self.timemethod = '1D'
        self.spacemethod = '1.5-degrees'
        self.basevar = variable
        
    def aggregatetime(self, freq = '7D' , method = 'mean', ndayagg = None, rolling = False, keep_leadtime = False):
        """
        Uses the pandas frequency indicators. Method can be mean, min, max, std
        Completely lazy when loading is lazy. Array needs to be already loaded because of variable choice.
        """      
        if keep_leadtime:
            leadtimes = self.array.coords['leadtime']
            if ndayagg is None:
                self.array, self.timemethod, ndayagg = agg_time(array = self.array, freq = freq, method = method, rolling = rolling, returnndayagg = True)
            else:
                self.array, self.timemethod = agg_time(array = self.array, freq = freq, method = method, rolling = rolling, ndayagg = ndayagg)
            if rolling:
                self.array.coords.update({'leadtime':leadtimes[slice(0,-(ndayagg - 1),None)]}) # Inclusive slicing. For 7day aggregation and max leadtime 46 the last one will be 40.
            else:
                self.array.coords.update({'leadtime':leadtimes[slice(0,None,ndayagg)]})
        else:
            self.array, self.timemethod = agg_time(array = self.array, freq = freq, method = method, rolling = rolling, ndayagg = ndayagg)
            try:
                self.array = self.array.drop('leadtime')
            except ValueError:
                pass
    
    def aggregatespace(self, level, clustername, clusterarray = None,  method = 'mean', skipna = True):
        """
        Aggregation by means of irregular clusters which can be supplied with the clusterarray for quickness.
        Otherwise the desired level is sought from the clustering dataset. In that case we cannot count on the fact that forecasts were already re-indexed in the matching and need to shrink by re-indexing.
        Completely lazy when supplied array is lazy.
        NOTE: this actually changes the dimension order of the array.
        """
        
        if clusterarray is None: # Shrinking too
            print('caution: forecast and the observed clusterarray are probably not yet on the same grid')
            clusterobject = Clustering(**{'name':clustername})
            clusterarray = clusterobject.get_clusters_at(level = level)
            clusterarray = clusterarray.sel(latitude = slice(self.array.latitude.min(),self.array.latitude.max()), longitude = slice(self.array.longitude.min(),self.array.longitude.max()))
            self.array = self.array.reindex_like(clusterarray, method = 'nearest')
        
        self.array, self.spacemethod = agg_space(array = self.array, level = level, clusterarray = clusterarray, clustername = clustername, method = method, skipna = skipna)
    
        
class Hindcast(object):
    """
    More difficult class because 20 reforecasts are contained in one file and need to be split to 20 separate processed files
    """
    def __init__(self, hdate = '2015-05-14', prefix = 'hin_', cycle = '41r1'):
        self.cycle = cycle
        self.stepbeforeresswitch = model_cycles.loc[model_cycles['cycle'] == self.cycle, 'stepbeforeresswitch'].values[0]
        #self.basedir = '/nobackup/users/straaten/EXT_extra/' + cycle + '/'
        self.basedir = '/nobackup/users/straaten/EXT_extra/' + cycle + '/'
        self.prefix = prefix
        self.hdate = hdate
        self.pffile = self.prefix + self.hdate + '_ens_sfc.grib'
        self.pffile_pl = self.prefix + self.hdate + '_ens_pl.grib'
        self.cffile = self.prefix + self.hdate + '_contr_sfc.grib'
        self.cffile_pl = self.prefix + self.hdate + '_contr_pl.grib'
        # Initialize one forecast class for each of the 20 runs (one year interval) contained in one reforecast
        # easier arithmetics when in pandas format, then to string for the mars format
        end = pd.to_datetime(self.hdate, format='%Y-%m-%d')
        self.hdates = [(end - pd.DateOffset(years = x)).strftime('%Y-%m-%d') for x in range(1,21)]
        self.hdates = [hd for hd in self.hdates if '02-29' not in hd] # Filter out the leap years.
        self.hdates.sort()
        self.marshdates = '/'.join(self.hdates)
        self.hindcasts = [Forecast(indate, self.prefix, self.cycle) for indate in self.hdates]
    
    def invoke_processed_creation(self):
        if all([os.path.isfile(self.basedir + hindcast.processedfile) for hindcast in self.hindcasts]):
            print('Processed hindcasts already exits. Do nothing')
        else:
            try:
                for hindcast in self.hindcasts:
                    hindcast.create_processed(prevent_cascade = True)
            except CascadeError:
                print('Combined files need creation (surface and pressure). Do this from the single grib files')
                self.crunch_gribfiles(pf_in = self.pffile, cf_in = self.cffile, comb_extension = '_comb_sfc.nc')  
                self.crunch_gribfiles(pf_in = self.pffile_pl, cf_in = self.cffile_pl, comb_extension = '_comb_pl.nc') # This won't re-download when they were actually downloaded first
                for hindcast in self.hindcasts:
                    hindcast.create_processed(prevent_cascade = True)
            except CascadePressureError:
                print('Combined pressure files need creation. Do this from the single grib files')
                self.crunch_gribfiles(pf_in = self.pffile_pl, cf_in = self.cffile_pl, comb_extension = '_comb_pl.nc')
                for hindcast in self.hindcasts:
                    hindcast.create_processed(prevent_cascade = True)
        
    def crunch_gribfiles(self, pf_in, cf_in, comb_extension = '_comb_sfc.nc'):
        """
        hdates within a file are extracted and perturbed and control are joined.
        The final files are saved per hdate as netcdf with three variables, 
        getting the name "_comb.nc" of "_comb_varres.nc" which can afterwards be read by the Forecast class
        """
        if comb_extension == '_comb_sfc.nc': # need to download surface params
            call_func = mars_dict_extra_surface
        else: 
            call_func = mars_dict_extra_pressure

        if os.path.isfile(self.basedir + pf_in):
            print('Ensemble file successfully loaded')
        else:
            print('Ensemble file needs to be downloaded')
            server.execute(call_func(self.hdate, hdate = self.marshdates, contr = False,
                                     varres = False, stepbeforeresswitch = self.stepbeforeresswitch), self.basedir + pf_in)
        
        if os.path.isfile(self.basedir + cf_in):
            print('Control file successfully loaded')
        else:
            print('Control file needs to be downloaded')
            server.execute(call_func(self.hdate, hdate = self.marshdates, contr = True,
                                     varres = False, stepbeforeresswitch = self.stepbeforeresswitch), self.basedir + cf_in)
       
        """
        Pre-allocate one dataset per hdate, (multiple variables)
        dimensions: time, latitude, longitude, number
        exact variables not known, so short discovery on first 100 messages, also for the grid
        """
        pf = open(self.basedir + pf_in, mode = 'rb')
        params = [] 
        units = {} 
        gribmissvals = {}
        for i in range(100):
            gid_pf = ec.codes_grib_new_from_file(pf)
            param = ec.codes_get(gid_pf,'cfVarName')
            params.append(param)
            units.update({param:ec.codes_get(gid_pf,'units')})
            gribmissvals.update({param:ec.codes_get(gid_pf,'missingValue')})
            ec.codes_release(gid_pf)
        params = list(set(params))

        gid_pf = ec.codes_grib_new_from_file(pf)
        lats = np.linspace(ec.codes_get(gid_pf, 'latitudeOfFirstGridPointInDegrees'), ec.codes_get(gid_pf,'latitudeOfLastGridPointInDegrees'), num = ec.codes_get(gid_pf, 'Nj'))
        lons = np.linspace(ec.codes_get(gid_pf, 'longitudeOfFirstGridPointInDegrees'), ec.codes_get(gid_pf,'longitudeOfLastGridPointInDegrees'), num = ec.codes_get(gid_pf, 'Ni'))
        lat_fastest_changing = (ec.codes_get(gid_pf, 'jPointsAreConsecutive') == 1)
        ec.codes_release(gid_pf)
        pf.close()

        steprange = list(range(0,1110,6))
        # key of the datasets will be the historical data filename
        # need to treat each hd seperately, otherwise this won't fit into memory
        def extract_field(messageid):
            values = ec.codes_get_values(messageid) # One dimensional array
            values = values.reshape((len(lats),len(lons)), order = 'F' if lat_fastest_changing else 'C') # order C means last index fastest changing
            return values

        for hd in self.hdates:
            validtimes = [pd.to_datetime(hd) + pd.DateOffset(hours = step) for step in steprange] 
            hd_as_int = int(''.join(hd.split('-'))) # For comparison against grib key dataDate (unchanging)
            ds = xr.Dataset({param:xr.DataArray(np.nan, dims = ('time','latitude','longitude','number'),coords = {'time':validtimes,'latitude':lats,'longitude':lons,'number':np.arange(11)}, attrs = {'units':units[param]}).astype(np.float32) for param in params})
     
            for filename in [pf_in, cf_in]:
                f = open(self.basedir + filename, mode = 'rb')
                while True:
                    gid = ec.codes_grib_new_from_file(f)
                    if gid is None:
                        break
                    if hd_as_int == ec.codes_get(gid, 'dataDate'):
                        step = ec.codes_get(gid, 'stepRange') # mx2t6 will be slighty weird, it has stepRanges like 0-6 or 12-18, whereas others are just 1104 or something
                        step = int(step.split('-')[-1])
                        name = ec.codes_get(gid, 'cfVarName')
                        number = ec.codes_get(gid, 'perturbationNumber') # for control this is just zero, so perfectly applicable
                        ds[name][steprange.index(step),:,:,number] = extract_field(gid) 
                    ec.codes_release(gid)
                        
                f.close() 
            
            # Saving stuff for this single hd
            svname = self.prefix + hd + comb_extension
            particular_encoding = {key : for_netcdf_encoding[key] for key in ds.variables.keys()} # get only encoding of present variables
            ds.to_netcdf(path = self.basedir + svname, encoding= particular_encoding)
        
    def cleanup(self):
        """
        Remove all files except the processed one. GRIB files are currently kept.
        """
        for hindcast in self.hindcasts:
            hindcast.cleanup()

class ModelClimatology(object):
    """
    Class to estimate model climatological mean per day of the year and per leadtime.
    So it pools over time and over members. But now has the possibility to compute quantiles, pooled over time,leadtime and members.
    When a new spacemethod is given, the newvarkwargs should contain an observations clusterarray (involves regridding)
    Otherwise, the climatology remains on the original forecast grid '0.38-degrees'
    """
    def __init__(self, cycle, variable, **kwds):
        """
        Var can be a combination like basevar-newvar, defining the operation to be done like tg-anom
        basear is the base variable that will be extracted from the model netcdfs. 
        """
        self.basedir = "/nobackup/users/straaten/modelclimatology/"
        self.cycle = cycle
        self.var = variable
        self.basevar = self.var.split('-')[0]
        self.maxleadtime = 46 #days
        self.maxdoy = 366
        self.changedunits = False
        for key in kwds.keys():
            setattr(self, key, kwds[key])
    
    def construct_name(self, force = False):
        """
        Name and filepath are based on the base variable and the relevant attributes (if present).
        Var can be a combination like basevar-newvar
        """
        keys = ['var','cycle','tmin','tmax', 'timemethod', 'spacemethod', 'daysbefore', 'daysafter', 'climmethod']
        if hasattr(self, 'name') and (not force):
            values = self.name.split(sep = '_')
            for key in keys:
                setattr(self, key, values[keys.index(key)])
        else:
            values = [ str(getattr(self,key)) for key in keys if hasattr(self, key)]
            self.name = '_'.join(values)
        
        self.filepath = ''.join([self.basedir, self.name, ".nc"])
    
    def local_clim(self, tmin = None, tmax = None, timemethod = '1D', spacemethod = '0.38-degrees', daysbefore = 5, daysafter = 5,  mean = True, quant = None, clusterarray = None, loadkwargs = {}, newvarkwargs = {}):
        """
        Method to construct the climatology based on forecasts within a desired timewindow.
        Should I add a spacemethod and spatial aggregation option? spacemethod = '0.38-degrees'
        Climatology dependend on doy and on leadtime. Takes the average over all ensemble members and times within a window.
        This window is determined by daysbefore and daysafter and the time aggregation of the desired variable.
        Newvarkwargs should contain a highresmodelclim for the conversion to anomalies. Observed clusterarray to do a spacemethod conversion is supplied seperately (also saved later)
        """
                
        keys = ['tmin','tmax','timemethod','spacemethod','daysbefore', 'daysafter']
        for key in keys:
            setattr(self, key, locals()[key])
            
        if mean:
            self.climmethod = 'mean'
        elif quant is not None:
            self.climmethod = 'q' + str(quant)
            from helper_functions import nanquantile
        
        # Overwrites possible nonsense attributes if name was supplied at initialization
        self.construct_name(force = False)
        
        # Extract a clusterarray for later saving.
        if not clusterarray is None:
            self.clusterarray = clusterarray
        
        try:
            self.clim = xr.open_dataarray(self.filepath)
            print('climatology directly loaded')
        except ValueError:
            self.clim = xr.open_dataarray(self.filepath, drop_variables = 'clustidfield').drop('dissim_threshold') # Also load the clustidfield if present??
            print('climatology directly loaded')
        except OSError:
            self.time_agg = int(pd.date_range('2000-01-01','2000-12-31', freq = timemethod.split('-')[0]).to_series().diff().dt.days.mode())
            
            eval_time_axis = pd.date_range(self.tmin, self.tmax, freq = 'D')
            
            climate = []
            
            for doy in range(1,self.maxdoy + 1):
                window = np.arange(doy - daysbefore, doy + daysafter + self.time_agg, dtype = 'int64')
                # small corrections for overshooting into previous or next year.
                window[ window < 1 ] += self.maxdoy
                window[ window > self.maxdoy ] -= self.maxdoy
                
                eval_indices = np.nonzero(np.isin(eval_time_axis.dayofyear, window))[0] # Can lead to empty windows if no entries in the chosen evaluation time axis belong to this doy and its window.
                eval_windows = np.split(eval_indices, np.nonzero(np.diff(eval_indices) > 1)[0] + 1)
                
                eval_time_windows = [ eval_time_axis[ind] for ind in eval_windows ]
                total = self.load_forecasts(eval_time_windows, loadkwargs = loadkwargs, newvarkwargs = newvarkwargs)
                
                if total is not None:
                    spatialdims = tuple(total.drop(['time','leadtime','number']).coords._names)
                    spatialcoords = {key:total.coords[key].values for key in spatialdims}
                    reduction_dims = ('number','time')
                    if mean:
                        doy_climate = total.groupby('leadtime').mean(reduction_dims, keep_attrs = True)
                        print('computed mean model climate of', doy, 'for', len(doy_climate['leadtime']), 'leadtimes.')
                    else: # In this case the quantile, meaning we don't do things by leadtime. Would be complicated
                        total = total.stack({'reducethis':reduction_dims}) # Nanquantile takes the quantile over the first axis, so we need to merge the reduction dimensions. Stacked into the last dimension
                        total = total.transpose(*(('reducethis',) + spatialdims))# Makes sure that the resulting 2D or 3D array has the reduces axis first.
                        doy_climate = xr.DataArray(data = nanquantile(array = total.values, q = quant), coords = spatialcoords, dims = spatialdims, name = self.var)
                        doy_climate.attrs = total.attrs
                        doy_climate.attrs['quantile'] = quant
                        print('computed quantile model climate of', doy, 'pooling leadtime.')
                    doy_climate.coords['doy'] = doy
                    climate.append(doy_climate)
                else:
                    print('no available forecasts for', doy, 'in chosen evaluation time axis')
            
            self.clim = xr.concat(climate, dim='doy') # Let it stretch/reindex itself to include the doys for which nothing was found
            self.clim = self.clim.reindex({'doy':range(1,self.maxdoy + 1)})
            
    def load_forecasts(self, evaluation_windows, n_members = 11, loadkwargs = {}, newvarkwargs = {}):
        """
        Per initialization date either the hindcast or the forecast can exist.
        Teste for possibility of empty supplied windows.
        Returns the whole block of (potentially time averaged) forecasts belonging to a set of windows around a doy, at different leadtimes.
        If no forecasts existed for these doy-specific evaluation_windos then it returns none.
        All conversion to newvars, and potential time and space-aggregation happens here.
        Time aggregation can both be rolling and non-rolling.
        """
        # Work per blocked window of a certain consecutive amount of days.
        forecast_collection = [] # Perhaps just a regular list?
        for window in evaluation_windows:
            
            if len(window) >= self.time_agg:
                # Determine forecast initialization times that fully contain the evaluation date (including its window for time aggregation)
                containstart = window.min() + pd.Timedelta(str(self.time_agg) + 'D') - pd.Timedelta(str(self.maxleadtime) + 'D') # Also plus 1 for the 1day aggregation?
                containend = window.max() - pd.Timedelta(str(self.time_agg - 1) + 'D') # Initialized at day x means it also already contains day x as leadtime = 1 day
                contain = pd.date_range(start = containstart, end = containend, freq = 'D')
                
                for indate in contain:
                    stringdate = indate.strftime('%Y-%m-%d')
                    forecasts = [Forecast(stringdate, prefix = 'for_', cycle = self.cycle), Forecast(stringdate, prefix = 'hin_', cycle = self.cycle)]
                    forecasts = [ f for f in forecasts if os.path.isfile(f.basedir + f.processedfile) ]
                    # Load the overlapping parts for the forecast that exist
                    if forecasts:
                        forecastrange = pd.date_range(indate, indate + pd.Timedelta(str(self.maxleadtime - 1) + 'D')) # Leadtime one plus 45 extra leadtimes.
                        overlap = np.intersect1d(forecastrange, window) # Overlap cannot be less that the aggregation period eventually required, because we selected for containend.
                        forecasts[0].load(self.basevar, tmin = overlap.min(), tmax = overlap.max(), n_members = n_members, **loadkwargs)
                        # Do conversion to newvar at highest resolution if applicable.
                        if (self.var != self.basevar) and (self.var.split('-')[-1] == 'anom'):
                            method = getattr(EventClassification(obs = forecasts[0], **newvarkwargs), 'anom')
                            method(inplace = True)
                            print('made into anomalies')
                        # Aggregate. What to do with leadtime? Assigned to first day as this is also done in the matching.
                        if forecasts[0].timemethod != self.timemethod:
                            freq, rolling, method = self.timemethod.split('-')
                            forecasts[0].aggregatetime(freq = freq, method = method, ndayagg = self.time_agg, rolling = True if (rolling == 'roll') else False, keep_leadtime = True)
                            print('aligned time')
                        if forecasts[0].spacemethod != self.spacemethod:
                            breakdown = self.spacemethod.split('-')
                            level, method = breakdown[::len(breakdown)-1] # first and last entry
                            clustername = '-'.join(breakdown[1:-1])
                            forecasts[0].array = forecasts[0].array.reindex_like(self.clusterarray, method = 'nearest')  # Involves regridding first.
                            forecasts[0].aggregatespace(level = level, clustername = clustername, clusterarray = self.clusterarray, method = method, skipna = True) # newvarkwargs supplies the clusterarray, which speeds up the computation. Forecasts also have no NA values we need to skip, speeds up the compute
                            print('aligned space')
                        
                        forecast_collection.append(forecasts[0].array)
        try:
            total = xr.concat(forecast_collection, dim = 'time') 
            if total.shape[0] == 0: # Test for empty first dimension.
                raise ValueError
            else:
                return(total)
        except ValueError: # No forecasts existed, or the supplied window was empty and forecast_collection remained empty, or the overlap was too little for the desired aggregation.
            return(None)
                            
    def change_units(self, newunit):
        """
        Simple linear change of units. Beware with changing units before saving.
        This will lead to problems with for_netcdf_encoding
        """
        a,b = unitconversionfactors(xunit = self.clim.units, yunit = newunit)
        self.clim = self.clim * a + b
        self.clim.attrs = {'units':newunit}
        self.changedunits = True
    
    def savelocalclim(self):
        
        if not self.changedunits:
            self.construct_name(force = True)
            if hasattr(self, 'clusterarray'):
                dataset = xr.Dataset({'clustidfield':self.clusterarray,self.clim.name:self.clim})
            else:
                dataset = self.clim.to_dataset()
            
            particular_encoding = {key : for_netcdf_encoding[key] for key in dataset.variables.keys()} 
            dataset.to_netcdf(self.filepath, encoding = particular_encoding)
        else:
            raise TypeError('You cannot save this model climatology after changing the original model units')

if __name__ == '__main__':
    #f = Forecast('2019-06-10', cycle = '45r1')
    #f.create_processed()
    #f.join_members(pf_in = f.pffile_pl, cf_in = f.cffile_pl, comb_out = f.interfile_pl)
    #f.join_members(pf_in = f.pffile, cf_in = f.cffile, comb_out = f.interfile)

    #h = Hindcast('2018-07-05', cycle = '45r1')
    #h.invoke_processed_creation()

    #start_batch(tmin = '2019-05-01', tmax = '2019-05-02') # SEGMENTATION FAULT for hindcast
    #start_batch(tmin = '2018-08-14', tmax = '2018-08-16') # Also SEGMENTATION fault for hindcast.
    #start_batch(tmin = '2019-03-23', tmax = '2019-03-30')
    #start_batch(tmin = '2019-03-31', tmax = '2019-04-06')
    #start_batch(tmin = '2019-04-07', tmax = '2019-04-13') 
    #start_batch(tmin = '2019-05-02', tmax = '2019-05-02') # 05-02 because of failure. from 04-13 onwards done.

    #start_batch(tmin = '2018-06-07', tmax = '2018-06-07') 
    """
    Modelclims for EXT_extra. Daily anomalies. Not year round available.
    """
    #f = Forecast('1998-06-07', prefix = 'hin_', cycle = '45r1')
    #f.load('swvl13')
    #import multiprocessing
    #def make_and_save_clim(var: str):
    #    temp = ModelClimatology(cycle='45r1', variable = var)
    #    #temp.local_clim(tmin = '1998-06-07', tmax = '2000-06-16', timemethod = '1D', spacemethod = '1.5-degrees', mean = True, loadkwargs = dict(llcrnr= (36,-24), rucrnr = (None,40)))
    #    temp.local_clim(tmin = '1998-06-07', tmax = '2018-08-31', timemethod = '1D', spacemethod = '1.5-degrees', mean = True)
    #    temp.savelocalclim()

    #with multiprocessing.Pool(2) as p: # Subprocess makes sure that memory is cleared
    #    p.map(make_and_save_clim, ['swvl13','sst', 'z', 'msl', 'swvl4', 'u','v'])


    """
    Some tests for tg-anom DJF, to see if we can get a quantile 
    Highresclim based on: dict(climtmin = '1998-06-07', climtmax = '2019-05-16', llcrnr= (36,-24), rucrnr = (None,40))
    """

#    modelclimname = 'tg_45r1_1998-06-07_2019-05-16_1D_0.38-degrees_5_5_mean' #'tg_45r1_1998-06-07_2019-05-16_1D_5_5'
#    highresmodelclim = ModelClimatology(cycle='45r1', variable = 'tg', **{'name':modelclimname}) # Name for loading
#    highresmodelclim.local_clim()
#    
#    cl = Clustering(**{'name':'tg-JJA'})
#    clusterarray = cl.get_clusters_at(level = 0.025)
#    
#    self = ModelClimatology(cycle='45r1', variable = 'tg-anom')
#    self.local_clim(tmin = '1998-06-07', tmax = '2019-05-16', timemethod = '9D-roll-mean', spacemethod = '0.025-tg-JJA-mean', mean = False, quant = 0.85, clusterarray = clusterarray, loadkwargs = dict(llcrnr= (36,-24), rucrnr = (None,40)), newvarkwargs = {'climatology':highresmodelclim})
#    self.savelocalclim()
#    tmin = '1998-06-07'
#    tmax = '2019-05-16'
#    timemethod = '9D-roll-mean' # 1day and 9day are needed
#    spacemethod = '0.025-tg-DJF-mean'
#    daysbefore = 5
#    daysafter = 5
#    mean = False
#    quant = 0.7
#    loadkwargs = dict(llcrnr= (36,-24), rucrnr = (None,40))
#    newvarkwargs = {'climatology':highresmodelclim, 'clusterarray':clusterarray}
