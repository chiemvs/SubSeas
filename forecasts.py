#!/usr/bin/env python3
import ecmwfapi 
import os
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import xarray as xr
import pygrib
from helper_functions import unitconversionfactors

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
                   'time': {'dtype': 'int64'},
                   'latitude': {'dtype': 'float32'},
                   'longitude': {'dtype': 'float32'},
                   'number': {'dtype': 'int16'},
                   'doy': {'dtype': 'int16'},
                   'leadtime': {'dtype': 'int16'}}

model_cycles = pd.DataFrame(data = {'firstday':pd.to_datetime(['2015-05-12','2016-03-08','2016-11-22','2017-07-11','2018-06-05','2019-06-11']),
                             'lastday':pd.to_datetime(['2016-03-07','2016-11-21','2017-07-10','2018-06-04','2019-06-10','']),
                             'cycle':['41r1','41r2','43r1','43r3','45r1','46r1'],
                             'stepbeforeresswitch':[240,360,360,360,360,360]})


def mars_dict(date, hdate = None, contr = False, varres = False, stepbeforeresswitch = None):
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
    if varres:
        req['stream'] = "efov" if hdate is None else "efho" # Some modification
        req['param'] = "228.230" # Only the variable resolution prec needed for de-accumulation
        req['step'] = str(stepbeforeresswitch)
        
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
        """
        Defines all the intermediate and raw forecast files that are needed for the processing.
        The final usable daily product is 'processedfile'
        """
        self.cycle = cycle
        self.stepbeforeresswitch = model_cycles.loc[model_cycles['cycle'] == self.cycle, 'stepbeforeresswitch'].values[0]
        self.basedir = '/nobackup/users/straaten/EXT/' + cycle + '/'
        self.prefix = prefix
        self.indate = indate
        self.processedfile = self.prefix + self.indate + '_processed.nc'
        self.interfile = self.prefix + self.indate + '_comb.nc'
        self.interfile_varres = self.prefix + self.indate + '_comb_varres.nc'
        self.pffile = self.prefix + self.indate + '_ens.nc'
        self.pffile_varres = self.prefix + self.indate + '_ens_varres.nc'
        self.cffile = self.prefix + self.indate + '_contr.nc'
        self.cffile_varres = self.prefix + self.indate + '_contr_varres.nc'
    
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
                comb_varres = xr.open_dataset(self.basedir + self.interfile_varres)
                print('Combined varres file successfully loaded')
            except OSError:
                print('Combined varres file needs creation')
                if prevent_cascade:
                    raise CascadeError
                self.join_members(pf_in = self.pffile_varres,
                                  cf_in = self.cffile_varres,
                                  comb_out = self.interfile_varres) # creates the combined varres interfile
                comb_varres = xr.open_dataset(self.basedir + self.interfile_varres)
                
            comb.load()
            comb_varres.load() # Also read as a dataset.
            
            # Precipitation. First resample: right limit is included because rain within a day accumulated till that 00UTC timestamp. Then de-accumulate
            tp = comb.tp.resample(time = 'D', closed = 'right').last()
            rr = tp.diff(dim = 'time', label = 'upper') # This cutoffs the first timestep.
            # Correction of de-accumulation. By subtracting coarse varres 00 UTC field before switch from 00UTC after switch (in tp, but labeled left)
            coarse_coarse = tp.sel(time = comb_varres.tpvar.time) - comb_varres.tpvar
            coarse_coarse = coarse_coarse.where(coarse_coarse >= 0, 0.0) # removing some very spurious and small negative values
            rr.loc[comb_varres.tpvar.time,...] = coarse_coarse
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
            comb_varres.close()
            print('Processed forecast successfully created')
            

    def join_members(self, pf_in, cf_in, comb_out):
        """
        Join members save the dataset. Control member gets the number 0.
        Only for non-hindcast forecasts.
        """
        try:
            pf = xr.open_dataset(self.basedir + pf_in)
            print('Ensemble file successfully loaded')
        except OSError:
            print('Ensemble file need to be downloaded')
            server.execute(mars_dict(self.indate, contr = False,
                                     varres = (comb_out == self.interfile_varres),
                                     stepbeforeresswitch = self.stepbeforeresswitch), self.basedir + pf_in)
            pf = xr.open_dataset(self.basedir + pf_in)
        
        try:
            cf = xr.open_dataset(self.basedir + cf_in)
            print('Control file successfully loaded')
        except OSError:
            print('Control file need to be downloaded')
            server.execute(mars_dict(self.indate, contr = True,
                                     varres = (comb_out == self.interfile_varres),
                                     stepbeforeresswitch = self.stepbeforeresswitch), self.basedir + cf_in)
            cf = xr.open_dataset(self.basedir + cf_in)
        
        cf.coords['number'] = np.array(0, dtype='int16')
        cf = cf.expand_dims('number',-1)
        particular_encoding = {key : for_netcdf_encoding[key] for key in cf.keys()} 
        xr.concat([cf,pf], dim = 'number').to_netcdf(path = self.basedir + comb_out, encoding= particular_encoding)
    
    def cleanup(self):
        """
        Remove all files except the processed one and the raw ones.
        """
        for filename in [self.interfile, self.interfile_varres]:
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
        self.spacemethod = '0.38-degrees'
        self.basevar = variable
        
    def aggregatetime(self, freq = '7D' , method = 'mean', ndayagg = None, rolling = False, keep_leadtime = False):
        """
        Uses the pandas frequency indicators. Method can be mean, min, max, std
        Completely lazy when loading is lazy. Array needs to be already loaded because of variable choice.
        """
        from helper_functions import agg_time
        
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
    
    def aggregatespace(self, step, method = 'mean', by_degree = False, rolling = False, skipna = True):
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
                                                 step = step, method = method, by_degree = by_degree, rolling = rolling, skipna = skipna)
    
        
class Hindcast(object):
    """
    More difficult class because 20 reforecasts are contained in one file and need to be split to 20 separate processed files
    """
    def __init__(self, hdate = '2015-05-14', prefix = 'hin_', cycle = '41r1'):
        self.cycle = cycle
        self.stepbeforeresswitch = model_cycles.loc[model_cycles['cycle'] == self.cycle, 'stepbeforeresswitch'].values[0]
        self.basedir = '/nobackup/users/straaten/EXT/' + cycle + '/'
        self.prefix = prefix
        self.hdate = hdate
        self.pffile = self.prefix + self.hdate + '_ens.grib'
        self.pffile_varres = self.prefix + self.hdate + '_ens_varres.grib'
        self.cffile = self.prefix + self.hdate + '_contr.grib'
        self.cffile_varres = self.prefix + self.hdate + '_contr_varres.grib'
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
                self.crunch_gribfiles(pf_in = self.pffile, cf_in = self.cffile, comb_extension = '_comb.nc')
                self.crunch_gribfiles(pf_in = self.pffile_varres, cf_in = self.cffile_varres, comb_extension = '_comb_varres.nc')
                for hindcast in self.hindcasts:
                    hindcast.create_processed(prevent_cascade = True)
        
    def crunch_gribfiles(self, pf_in, cf_in, comb_extension = '_comb.nc'):
        """
        hdates within a file are extracted and perturbed and control are joined.
        The final files are saved per hdate as netcdf with three variables, 
        getting the name "_comb.nc" of "_comb_varres.nc" which can afterwards be read by the Forecast class
        """
        try:
            pf = pygrib.open(self.basedir + pf_in)
            print('Ensemble file successfully loaded')
        except:
            print('Ensemble file needs to be downloaded')
            server.execute(mars_dict(self.hdate, hdate = self.marshdates, contr = False,
                                     varres = (pf_in == self.pffile_varres),
                                     stepbeforeresswitch = self.stepbeforeresswitch), self.basedir + pf_in)
            pf = pygrib.open(self.basedir + pf_in)
        
        try:
            cf = pygrib.open(self.basedir + cf_in)
            print('Control file successfully loaded')
        except:
            print('Control file needs to be downloaded')
            server.execute(mars_dict(self.hdate, hdate = self.marshdates, contr = True,
                                     varres = (cf_in == self.cffile_varres),
                                     stepbeforeresswitch = self.stepbeforeresswitch), self.basedir + cf_in)
            cf = pygrib.open(self.basedir + cf_in)
        
        params = list(set([x.cfVarName for x in pf.read(100)])) # Enough to get the variables. ["167.128","121.128","228.128"] or "228.230"
        
        if ('tpvar' in params):
            steprange = [self.stepbeforeresswitch] # The variable resolution stream is only downloaded with one step.
        else:
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
                
                lats = np.linspace(control[0]['latitudeOfFirstGridPointInDegrees'], control[0]['latitudeOfLastGridPointInDegrees'], num = control[0]['Nj'])
                lons = np.linspace(control[0]['longitudeOfFirstGridPointInDegrees'], control[0]['longitudeOfLastGridPointInDegrees'], num = control[0]['Ni'])
                #lats,lons = control[0].latlons()
                units = control[0].units
                gribmissval = control[0].missingValue
            
                for i in range(0,len(steps)): # use of index because later on the original steprange is needed for timestamps
                    cthisstep = [c for c in control if c.stepRange == steps[i]]
                    mthisstep = [m for m in members if m.stepRange == steps[i]]
                    # If empty lists (tmax analysis) the field in nonexisting and we want fillvalues
                    if not cthisstep:
                        print('missing field')
                        controlval = np.full(shape = (len(lats),len(lons)), fill_value = float(gribmissval)) #np.full(shape = lats.shape, fill_value = float(gribmissval))
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
                                          coords=[timestamp, lats, lons, membersnum], #coords=[timestamp, lats[:,0], lons[0,:], membersnum], 
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
            svname = self.prefix + hd + comb_extension
            particular_encoding = {key : for_netcdf_encoding[key] for key in onehdate.variables.keys()} # get only encoding of present variables
            onehdate.to_netcdf(path = self.basedir + svname, encoding= particular_encoding)
        pf.close()
        cf.close()
        
    def cleanup(self):
        """
        Remove all files except the processed one. GRIB files are currently kept.
        """
        for hindcast in self.hindcasts:
            hindcast.cleanup()

class ModelClimatology(object):
    """
    Class to estimate model climatology per day of the year and per leadtime.
    Only means, over time and over members.
    """
    def __init__(self, cycle, variable, **kwds):
        """
        Var is the base variable that will be extracted from the model netcdfs. 
        """
        self.basedir = "/nobackup/users/straaten/modelclimatology/"
        self.cycle = cycle
        self.var = variable
        self.maxleadtime = 46 #days
        self.maxdoy = 366
        self.changedunits = False
        for key in kwds.keys():
            setattr(self, key, kwds[key])
    
    def construct_name(self, force = False):
        """
        Name and filepath are based on the base variable and the relevant attributes (if present).
        """
        keys = ['var','tmin','tmax', 'timemethod', 'daysbefore', 'daysafter']
        if hasattr(self, 'name') and (not force):
            values = self.name.split(sep = '_')
            for key in keys:
                setattr(self, key, values[keys.index(key)])
        else:
            values = [ str(getattr(self,key)) for key in keys if hasattr(self, key)]
            self.name = '_'.join(values)
        
        self.filepath = ''.join([self.basedir, self.name, ".nc"])
    
    def local_clim(self, tmin = None, tmax = None, timemethod = '1D', daysbefore = 5, daysafter = 5, loadkwargs = {}):
        """
        Method to construct the climatology based on forecasts within a desired timewindow.
        Should I add a spacemethod and spatial aggregation option? spacemethod = '0.38-degrees'
        Climatology dependend on doy and on leadtime. Takes the average over all ensemble members and times within a window.
        This window is determined by daysbefore and daysafter and the time aggregation of the desired variable.
        """
                
        keys = ['tmin','tmax','timemethod','daysbefore', 'daysafter']
        for key in keys:
            setattr(self, key, locals()[key])
        
        # Overwrites possible nonsense attributes if name was supplied at initialization
        self.construct_name(force = False)
                
        try:
            self.clim = xr.open_dataarray(self.filepath)
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
                total = self.load_forecasts(eval_time_windows, loadkwargs = loadkwargs)
                
                if total is not None:
                    doy_climate = total.groupby('leadtime').mean(['number','time'], keep_attrs = True)
                    doy_climate.coords['doy'] = doy
                    climate.append(doy_climate)
                    print('computed model climate of', doy, 'for', len(doy_climate['leadtime']), 'leadtimes.')
                else:
                    print('no available forecasts for', doy, 'in chosen evaluation time axis')
                    f = Forecast()
                    f.load(variable=self.var, n_members = 1, **loadkwargs)
                    doy_climate = f.array.swap_dims({'time':'leadtime'}).isel(leadtime = slice(None), latitude = slice(None), longitude = slice(None), number = 0).drop(['number','time'])
                    doy_climate[:] = np.nan
                    doy_climate.coords['doy'] = doy
                    climate.append(doy_climate)
            
            self.clim = xr.concat(climate, dim='doy')
            
    def load_forecasts(self, evaluation_windows, n_members = 11, loadkwargs = {}):
        """
        Per initialization date either the hindcast or the forecast can exist.
        Teste for possibility of empty supplied windows.
        Returns the whole block of (potentially time averaged) forecasts belonging to a set of windows around a doy, at different leadtimes.
        If no forecasts existed for these doy-specific evaluation_windos then it returns none.
        """
        # Work per blocked window of a certain consecutive amount of days.
        forecast_collection = [] # Perhaps just a regular list?
        for window in evaluation_windows:
            
            if len(window) > 0:
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
                        overlap = np.intersect1d(forecastrange, window) # Some sort of inner join? numpy perhaps?
                        forecasts[0].load(self.var, tmin = overlap.min(), tmax = overlap.max(), n_members = n_members, **loadkwargs)
                        # Aggregate. What to do with leadtime? Assigned to first day as this is also done in the matching.
                        if forecasts[0].timemethod != self.timemethod:
                            freq, rolling, method = self.timemethod.split('-')
                            forecasts[0].aggregatetime(freq = freq, method = method, ndayagg = self.time_agg, rolling = False, keep_leadtime = True) # Array becomes empty when overlap loaded is less than the time aggregation.
                            print('aligned time')
                            
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
            particular_encoding = {key : for_netcdf_encoding[key] for key in self.clim.to_dataset().variables.keys()} 
            self.clim.to_netcdf(self.filepath, encoding = particular_encoding)
        else:
            raise TypeError('You cannot save this model climatology after changing the original model units')

#self = ModelClimatology('41r1', 'tg')
#self.local_clim(tmin = '2000-01-01',tmax = '2001-01-21', timemethod = '1D', daysbefore = 3, daysafter = 3)

#start_batch(tmin = '2018-06-07', tmax = '2018-07-10')
#start_batch(tmin = '2018-07-11', tmax = '2018-08-31')
#start_batch(tmin = '2018-11-01', tmax = '2018-11-30')
#start_batch(tmin = '2019-04-15', tmax = '2019-04-22')
#start_batch(tmin = '2019-04-23', tmax = '2019-04-30')
