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
for_netcdf_encoding = {'t2m': {'dtype': 'int16', 'scale_factor': 0.0015, 'add_offset': 283, '_FillValue': -32767},
                   'mx2t6': {'dtype': 'int16', 'scale_factor': 0.0015, 'add_offset': 283, '_FillValue': -32767},
                   'tp': {'dtype': 'int16', 'scale_factor': 0.00005, '_FillValue': -32767},
                   'rr': {'dtype': 'int16', 'scale_factor': 0.00005, '_FillValue': -32767},
                   'tx': {'dtype': 'int16', 'scale_factor': 0.0015, 'add_offset': 283, '_FillValue': -32767},
                   'tg': {'dtype': 'int16', 'scale_factor': 0.0015, 'add_offset': 283, '_FillValue': -32767},
                   'time': {'dtype': 'int64'},
                   'latitude': {'dtype': 'float32'},
                   'longitude': {'dtype': 'float32'},
                   'number': {'dtype': 'int16'},
                   'leadtime': {'dtype': 'int16'}}

model_cycles = pd.DataFrame(data = {'firstday':pd.to_datetime(['2015-05-12','2016-03-08','2016-11-22','2017-07-11','2018-06-05','']),
                             'lastday':pd.to_datetime(['2016-03-07','2016-11-21','2017-07-10','2018-06-04','2019-06-30','']),
                             'cycle':['41r1','41r2','43r1','43r3','45r1','46r1']})


def mars_dict(date, hdate = None, contr = False):
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
    'param'     : "167.128/121.128/228.128", # T2M (Kelvin), Tmax in last 6 hrs. and Tot prec. Tot prec needs de-accumulation
    'step'      : "0/to/1104/by/6",
    'ppengine'  : "mir",
    'area'      : "75/-30/25/75", #E-OBS 75.375/-40.375/25.375/75.375/ Limits 75/-40.5/25/75.5
    'grid'      : ".38/.38", # Octahedral grid does not support sub-areas
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
                       
class Forecast(object):

    def __init__(self, indate = '2015-05-14', prefix = 'for_', cycle = '41r1'):
        self.cycle = cycle
        self.basedir = '/nobackup/users/straaten/EXT/' + cycle + '/'
        self.prefix = prefix
        self.indate = indate
        self.processedfile = self.prefix + self.indate + '_processed.nc'
        self.interfile = self.prefix + self.indate + '_comb.nc'
        self.pffile = self.prefix + self.indate + '_ens.nc'
        self.cffile = self.prefix + self.indate + '_contr.nc'
    
    def create_processed(self, prevent_cascade = False):
        """
        Joining of members for the combined file is not needed for forecasts 
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
                self.join_members() # creates the interfile
                comb = xr.open_dataset(self.basedir + self.interfile)
            
            comb.load()
            
            # Precipitation. First resample: right limit is included because rain within a day accumulated till that 00UTC timestamp. Then de-accumulate
            tp = comb.tp.resample(time = 'D', closed = 'right').last()
            rr = tp.diff(dim = 'time', label = 'upper') # This cutoffs the first timestep.
            rr.attrs.update({'long_name':'precipitation', 'units':comb.tp.units})
            # Mean temperature. Resample. Cutoff last timestep because this is no average (only instantaneous 00UTC value)
            tg = comb.t2m.resample(time = 'D').mean('time').isel(time = slice(0,-1))
            tg.attrs.update({'long_name':'mean temperature', 'units':comb.t2m.units})
            # Maximum temperature. Right limit is included because forecast value is the maximum in previous six hours. Therefore also cutoff the first timestep
            tx = comb.mx2t6.resample(time = 'D', closed = 'right').max('time').isel(time = slice(1,None))
            tx.attrs.update({'long_name':'maximum temperature', 'units':comb.mx2t6.units})
            
            # Join and add leadtime dimension (days) for clarity
            result = xr.Dataset({'rr':rr,'tg':tg,'tx':tx})
            result['leadtime'] = ('time', np.arange(1, len(result.coords['time'])+1, dtype = 'int16'))
            result.leadtime.attrs.update({'long_name':'leadtime', 'units':'days'})
            result.set_coords('leadtime', inplace=True) # selection by leadtime requires a quick swap: result.swap_dims({'time':'leadtime'})
            
            particular_encoding = {key : for_netcdf_encoding[key] for key in result.keys()} 
            result.to_netcdf(path = self.basedir + self.processedfile, encoding = particular_encoding)
            comb.close()
            print('Processed forecast successfully created')
            

    def join_members(self):
        """
        Join members save the dataset. Control member gets the number 0.
        Only for non-hindcast forecasts.
        """
        try:
            pf = xr.open_dataset(self.basedir + self.pffile)
            print('Ensemble file successfully loaded')
        except OSError:
            print('Ensemble file need to be downloaded')
            server.execute(mars_dict(self.indate, contr = False), self.basedir+self.pffile)
            pf = xr.open_dataset(self.basedir + self.pffile)
        
        try:
            cf = xr.open_dataset(self.basedir + self.cffile)
            print('Control file successfully loaded')
        except OSError:
            print('Control file need to be downloaded')
            server.execute(mars_dict(self.indate, contr = True), self.basedir+self.cffile)
            cf = xr.open_dataset(self.basedir + self.cffile)
        
        cf.coords['number'] = np.array(0, dtype='int16')
        cf = cf.expand_dims('number',-1)
        particular_encoding = {key : for_netcdf_encoding[key] for key in cf.keys()} 
        xr.concat([cf,pf], dim = 'number').to_netcdf(path = self.basedir + self.interfile, encoding= particular_encoding)
    
    def cleanup(self):
        """
        Remove all files except the processed one.
        """
        for filename in [self.interfile, self.pffile, self.cffile]:
            try:
                os.remove(self.basedir + filename)
            except OSError:
                pass
    
    def load(self, variable = None, tmin = None, tmax = None, n_members = None):
        """
        Loading of processedfile. Similar behaviour to observation class
        If n_members is set to one it selects only the control member.
        Then the numbers are set to a default range because they are not unique across initialization times
        over which they are later pooled. Within one forecast instance they do of course resemble a physical pathway.
        """
        
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
        
        self.array = full.sel(time = pd.date_range(tmin, tmax, freq = 'D'), number = numbers)
        # reset the index
        if n_members is not None:
            self.array.coords['number'] = np.arange(0,n_members, dtype = 'int16')
        # Standard methods of the processed files.
        self.timemethod = '1D'
        self.spacemethod = '0.38_degrees'
        
    def aggregatetime(self, freq = 'w' , method = 'mean', keep_leadtime = False):
        """
        Uses the pandas frequency indicators. Method can be mean, min, max, std
        Completely lazy when loading is lazy. Array needs to be already loaded because of variable choice.
        """
        from helper_functions import agg_time
        
        if keep_leadtime:
            lead0 = self.array.coords['leadtime'].isel(time = slice(0,1))
            self.array, self.timemethod = agg_time(array = self.array, freq = freq, method = method)
            self.array.coords.update({'leadtime':lead0})
        else:
            self.array, self.timemethod = agg_time(array = self.array, freq = freq, method = method)
    
    def aggregatespace(self, step, method = 'mean', by_degree = False, skipna = True):
        """
        Regular lat lon or gridbox aggregation by creating new single coordinate which is used for grouping.
        In the case of degree grouping the groups might not contain an equal number of cells.
        Completely lazy when supplied array is lazy.
        NOTE: this actually changes the dimension order of the array.
        """
        from helper_functions import agg_space
        
        self.array, self.spacemethod = agg_space(array = self.array, 
                                                 orlats = self.array.latitude.load(),
                                                 orlons = self.array.longitude.load(),
                                                 step = step, method = method, by_degree = by_degree, skipna = skipna)
    
        
class Hindcast(object):
    """
    More difficult class because 20 reforecasts are contained in one file and need to be split to 20 separate processed files
    """
    def __init__(self, hdate = '2015-05-14', prefix = 'hin_', cycle = '41r1'):
        self.cycle = cycle
        self.basedir = '/nobackup/users/straaten/EXT/' + cycle + '/'
        self.prefix = prefix
        self.hdate = hdate
        self.pffile = self.prefix + self.hdate + '_ens.grib'
        self.cffile = self.prefix + self.hdate + '_contr.grib'
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
                print('Combined files need creation. Do this from the single grib files')
                self.crunch_gribfiles()
                for hindcast in self.hindcasts:
                    hindcast.create_processed(prevent_cascade = True)
        
    def crunch_gribfiles(self):
        """
        hdates within a file are extracted and perturbed and control are joined.
        The final files are saved per hdate as netcdf with three variables, 
        getting the name "_comb.nc" which can afterwards be read by the Forecast class
        """
        try:
            pf = pygrib.open(self.basedir + self.pffile)
            print('Ensemble file successfully loaded')
        except:
            print('Ensemble file needs to be downloaded')
            server.execute(mars_dict(self.hdate, hdate = self.marshdates, contr = False), self.basedir+self.pffile)
            pf = pygrib.open(self.basedir + self.pffile)
        
        try:
            cf = pygrib.open(self.basedir + self.cffile)
            print('Control file successfully loaded')
        except:
            print('Control file needs to be downloaded')
            server.execute(mars_dict(self.hdate, hdate = self.marshdates, contr = True), self.basedir+self.cffile)
            cf = pygrib.open(self.basedir + self.cffile)
        
        params = list(set([x.cfVarName for x in cf.read(100)])) # Enough to get the variables. ["167.128","121.128","228.128"] # Hardcoded for marsParam

        steprange = np.arange(0,1110,6) # Hardcoded as this is too slow: steps = list(set([x.stepRange for x in cf.select()]))
        beginning = steprange[1:-1].tolist()
        beginning[0:0] = [0,0]
        tmaxrange = [ str(b) + '-' + str(e) for b,e in zip(beginning, steprange)] # special hardcoded range for the tmax, whose stepRanges are stored differently

        for hd in self.hdates:
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
            svname = self.prefix + hd + '_comb.nc'
            particular_encoding = {key : for_netcdf_encoding[key] for key in onehdate.keys()} # get only encoding of present variables
            onehdate.to_netcdf(path = self.basedir + svname, encoding= particular_encoding)
        pf.close()
        cf.close()
        
    def cleanup(self):
        """
        Remove all files except the processed one. GRIB files are currently kept.
        """
        for hindcast in self.hindcasts:
            hindcast.cleanup()

#start_batch(tmin = '2018-06-05', tmax = '2018-07-12')
