#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 09:31:16 2018

@author: straaten
"""
import os
import sys
import numpy as np
import xarray as xr
import pandas as pd
import dask.dataframe as dd
import itertools
import properscoring as ps
import uuid
import multiprocessing
from datetime import datetime
from observations import SurfaceObservations, Climatology, EventClassification, Clustering
from forecasts import Forecast, ModelClimatology
from helper_functions import monthtoseasonlookup, unitconversionfactors, lastconsecutiveabove, assignmidpointleadtime
from fitting import NGR, Logistic, ExponentialQuantile

def matchforecaststoobs(obs, datesubset, outfilepath, books_path, time_agg, n_members, cycle, maxleadtime, loadkwargs, newvarkwargs, newvariable = False, matchtime = True, matchspace = True):
    """
    Function to act as a child process, matching forecasts to the obs until a certain size dataframe is reached. Then that file is written, and the booksfile is updated.
    Neirest neighbouring to match the forecast grid to the observed grid in the form of the clusterarray belonging to the observations. Be careful when domain of the (clustered) observations is larger.
    Determines the order in which space-time aggregation and classification is done based on the potential newvar.
    Also calls unit conversion for the forecasts. (the observed units of a newvariable are actually its old units)
    Loadkwargs can carry the delimiting spatial corners if only a part of the domain is desired.
    """
    def find_forecasts(date):
        """
        Here the forecasts corresponding to the observations are determined, by testing their existence.
        Hindcasts and forecasts might be mixed. But they are in the same class.
        Leadtimes may differ.
        Returns an empty list if non were found
        """       
        # Find forecast initialization times that fully contain the observation date (including its window for time aggregation)
        containstart = date + pd.Timedelta(str(time_agg) + 'D') - pd.Timedelta(str(maxleadtime) + 'D')
        containend = date
        contain = pd.date_range(start = containstart, end = containend, freq = 'D').strftime('%Y-%m-%d')
        if obs.basevar in ['tg','tx','rr']:
            forbasedir = '/nobackup/users/straaten/EXT/'
        else:
            forbasedir = '/nobackup/users/straaten/EXT_extra/'
        forecasts = [Forecast(indate, prefix = 'for_', cycle = cycle, basedir = forbasedir) for indate in contain]
        hindcasts = [Forecast(indate, prefix = 'hin_', cycle = cycle, basedir = forbasedir) for indate in contain]
        # select from potential forecasts only those that exist.
        forecasts = [f for f in forecasts if os.path.isfile(f.basedir + f.processedfile)]
        hindcasts = [h for h in hindcasts if os.path.isfile(h.basedir + h.processedfile)]
        return(forecasts + hindcasts)
    
    def load_forecasts(date, listofforecasts):
        """
        Gets the daily processed forecasts into memory. Delimited by the left timestamp and the aggregation time.
        This is done by using the load method of each Forecast class in the dictionary. They are stored in a list.
        """ 
        tmin = date
        tmax = date + pd.Timedelta(str(time_agg - 1) + 'D') # -1 because date itself also counts for one day in the aggregation.
       
        for forecast in listofforecasts:
            forecast.load(variable = obs.basevar, tmin = tmin, tmax = tmax, n_members = n_members, **loadkwargs)
     
    def force_resolution(forecast, time = True, space = True):
        """
        Force the observed resolution onto the supplied Forecast. Checks if the same resolution and force spatial/temporal aggregation if that is not the case. Checks will fail on the roll-norm difference, e.g. 2D-roll-mean observations and .
        Makes use of the methods of each Forecast class. Checks can be switched on and off. Time-rolling is never applied to the forecasts as for each date already the precise window was loaded, but space-rolling is.
        """                
        if time:
            # Check time aggregation
            obstimemethod = getattr(obs, 'timemethod')
                        
            try:
                fortimemethod = getattr(forecast, 'timemethod')
                if not fortimemethod == obstimemethod:
                    raise AttributeError
                else:
                    print('Time already aligned')
            except AttributeError:
                print('Aligning time aggregation')
                freq, rolling, method = obstimemethod.split('-')
                forecast.aggregatetime(freq = freq, method = method, keep_leadtime = True, ndayagg = time_agg, rolling = False)
        
        if space:
            # Check space aggregation
            obsspacemethod = getattr(obs, 'spacemethod')
                
            try:
                forspacemethod = getattr(forecast, 'spacemethod')
                if not forspacemethod == obsspacemethod:
                    raise AttributeError
                else:
                    print('Space already aligned')
            except AttributeError:
                print('Aligning space aggregation')
                breakdown = obsspacemethod.split('-')
                level, method = breakdown[::len(breakdown)-1] # first and last entry
                clustername = '-'.join(breakdown[1:-1])
                forecast.aggregatespace(level = level, clustername = clustername, clusterarray = obs.clusterarray, method = method, skipna = True) # Forecasts will not have NA values anywhere and this speeds up the computation.
    
    def force_units(forecast):
        """
        Linear conversion of forecast object units to match the observed units.
        """
        a,b = unitconversionfactors(xunit = forecast.array.units, yunit = obs.array.units)
        forecast.array = forecast.array * a + b
        forecast.array.attrs = {'units': obs.array.units}
                      
    def force_new_variable(forecast, newvarkwargs, inplace = True, newvariable: str = None):
        """
        Call upon event classification on the forecast object to get the on-the-grid conversion of the base variable.
        This is classification method is the same as applied to obs and is determined by the similar name.
        Possibly returns xarray object if inplace is False. If newvar is anom and model climatology is supplied through newvarkwargs, it should already have undergone the change in units.
        """
        if newvariable is None:
            newvariable = obs.newvar
        method = getattr(EventClassification(obs = forecast, **newvarkwargs), newvariable)
        return(method(inplace = inplace))
    
    p = multiprocessing.current_process()
    print('Starting:', p.pid)
    
    aligned_basket = []
    aligned_basket_size = 0 # Size of the content in the basket. Used to determine when to write
    indexdtypes = {'clustid':'integer','latitude':'float','longitude':'float'}
    
    while (aligned_basket_size < 5*10**8) and (not datesubset.empty):
        
        date = datesubset.iloc[0]
        listofforecasts = find_forecasts(date) # Results in empty list if none were found
        
        if listofforecasts:
            
            load_forecasts(date = date, listofforecasts = listofforecasts) # Does this modify listofforecasts inplace?
            
            for forecast in listofforecasts:
                force_units(forecast) # Get units correct.
                if newvariable:
                    if obs.newvar == 'anom':
                        force_new_variable(forecast, newvarkwargs = newvarkwargs, inplace = True) # If newvar is anomaly then first new variable and then aggregation. If e.g. newvar is pop then first aggregation then transformation.
                        forecast.array = forecast.array.reindex_like(obs.clusterarray, method = 'nearest')
                        force_resolution(forecast, time = matchtime, space = matchspace)
                        forecast.array = forecast.array.swap_dims({'time':'leadtime'}) # So it happens inplace
                    elif obs.newvar in ['pop','pod']: # This could even be some sort of 'binary newvariable' criterion.
                        forecast.array = forecast.array.reindex_like(obs.clusterarray, method = 'nearest')
                        force_resolution(forecast, time = matchtime, space = matchspace)
                        forecast.array = forecast.array.swap_dims({'time':'leadtime'}) # So it happens inplace
                        try:
                            listofbinaries.append(force_new_variable(forecast, newvarkwargs = newvarkwargs, inplace = False))
                        except NameError:
                            listofbinaries = [force_new_variable(forecast, newvarkwargs = newvarkwargs, inplace = False)]
                    elif obs.newvar.startswith('ex'): # hotdays newvariable
                        assert not matchtime, 'hotdays requires unaggregated daily data, do not match time'
                        force_new_variable(forecast, newvarkwargs = newvarkwargs, inplace = True, newvariable = 'anom') #First anomalies with highresclim (supplied via newvarkwargs)
                        forecast.array = forecast.array.reindex_like(obs.clusterarray, method = 'nearest')
                        force_resolution(forecast, time = matchtime, space = matchspace)
                        force_new_variable(forecast, newvarkwargs = newvarkwargs, inplace = True, newvariable = 'hotdays') # Second hotdays newvariable step. With quantileclimatology (also supplied via newvarwkargs). Because it is a rolling temporal aggregation the hotdays event classification takes care of left-stamping time and leadtime
                        forecast.array = forecast.array.swap_dims({'time':'leadtime'}) # So it happens inplace

                else:
                    forecast.array = forecast.array.reindex_like(obs.clusterarray, method = 'nearest')
                    force_resolution(forecast, time = matchtime, space = matchspace)
                    forecast.array = forecast.array.swap_dims({'time':'leadtime'}) # So it happens inplace
            
            # Get the correct observed and inplace forecast arrays.
            fieldobs = obs.array.sel(time = date).drop('time')
            allleadtimes = xr.concat(objs = [f.array for f in listofforecasts], 
                                             dim = 'leadtime') # concatenates over leadtime dimension.
            
            # When we have created a binary new_variable like pop, the forecastlist was appended with the not-inplace transformed ones 
            # of These we only want to retain the ensemble probability (i.e. the mean). For the binaries we only retain the probability of the event over the members (i.e. the mean)
            try:
                pi = xr.concat(objs = listofbinaries,dim = 'leadtime').mean(dim = 'number') # concatenates over leadtime dimension.
                listofbinaries.clear() # Empty for next iteration
                # Merging, exporting to pandas and masking by dropping on NA observations.
                combined = xr.Dataset({'forecast':allleadtimes.drop('time'),'observation':fieldobs, 'pi':pi.drop('time')}).to_dataframe().dropna(axis = 0)
                # Unstack creates duplicates. Two extra columns (obs and pi) need to be selected. Therefore the iloc
                temp = combined.unstack('number').iloc[:,np.append(np.arange(0,n_members + 1), n_members * 2)]
                temp.reset_index(inplace = True) # places spatial and time dimension in the columns
                #labels = temp.columns.labels[1].tolist()
                #labels[-2:] = np.repeat(n_members, 2)
            except NameError:
                # Merging, exporting to pandas and masking by dropping on NA observations.
                combined = xr.Dataset({'forecast':allleadtimes.drop('time'), 'observation':fieldobs}).to_dataframe().dropna(axis = 0)
                # Handle the multi-index
                # first puts number dimension into columns. observerations are duplicated so therefore selects up to n_members +1
                temp = combined.unstack('number').iloc[:,:(n_members + 1)]
                temp.reset_index(inplace = True) # places spatial and time dimension in the columns
                #labels = temp.columns.get_level_values(1).tolist() # get_level_values
                #labels[-1] = n_members
            
            #temp.columns.set_labels(labels, level = 1, inplace = True) # set_levels
            
            # Downcasting the spatial coordinates and leadtime
            spatialcoords = list(fieldobs.dims)
            for key in spatialcoords:
                temp[[key]] = temp[[key]].apply(pd.to_numeric, downcast = indexdtypes[key])
            temp['leadtime'] = np.array(temp['leadtime'].values.astype('timedelta64[D]'), dtype = 'int8')
            
            # prepend with the time index.
            temp.insert(0, 'time', date)
            aligned_basket.append(temp)
            print(date, 'matched')
            
            # If aligned takes too much system memory (> 500Mb) . Write it out
            aligned_basket_size += sys.getsizeof(temp)
        datesubset = datesubset.drop(date)
    
    if aligned_basket:
        dataset = pd.concat(aligned_basket)
        dataset.to_hdf(outfilepath, key = 'intermediate', format = 'table')
        
        books = pd.DataFrame({'file':[outfilepath],
                              'tmax':[dataset.time.max().strftime('%Y-%m-%d')],
                              'tmin':[dataset.time.min().strftime('%Y-%m-%d')],
                              'unit':[obs.array.units],
                              'write_date':[datetime.now().strftime('%Y-%m-%d_%H:%M:%S')]})
        
        # Create booksfile if it does not exist, otherwise append to it.
        try:
            with open(books_path, 'x') as f:
                books.to_csv(f, header=True, index = False)
        except FileExistsError:
            with open(books_path, 'a') as f:
                books.to_csv(f, header = False, index = False)
        print('written out', outfilepath)
    else:
        raise ValueError('empty alignment, found no further initialization dates for remaining observed dates') 
    print('Exiting:', p.pid)


class ForecastToObsAlignment(object):
    """
    Idea is: you have already prepared an observation class, with a certain variable, temporal extend and space/time aggregation.
    This searches the corresponding forecasts original, forces similar: units, aggregations and potentially a new-variable adaptation.
    """
    def __init__(self, season, cycle, n_members = 11,  observations = None, **kwds):
        """
        Temporal extend can be read from the attributes of the observation class. 
        Specify the season under inspection for subsetting.
        """
        self.basedir = '/nobackup_1/users/straaten/match/'
        self.cycle = cycle
        self.season = season
        self.n_members = n_members
        if observations is not None:
            self.obs = observations
            self.dates = self.obs.array.coords['time'].to_series() # Left stamped
            self.dates = self.dates[monthtoseasonlookup(self.dates.index.month) == self.season]
            self.obs.array = self.obs.array.sel(time = self.dates.values)
            # infer dominant time aggregation in days
            timefreq = self.obs.timemethod.split('-')[0]
            self.time_agg = int(pd.date_range('2000-01-01','2000-12-31', freq = timefreq).to_series().diff().dt.days.mode()) # Determines the amount of forecast days loaded per initialized forecast.
            self.maxleadtime = 46
        
        for key in kwds.keys():
            setattr(self, key, kwds[key])
                
    def match_and_write(self, newvariable = False, newvarkwargs = {}, loadkwargs = {}, matchtime = True, matchspace = True):
        """
        Control function to spawn the processes matching forecasts to the observation.
        Prepares the input to the top-level matching function (see parameter description there), and spawns it.
        Waits until termination to figure out by means of the booksfile how much was matched, such that the next is spawned.
        This termination is currently determined by the pressure on memory, inside the top level function.
        NOTE: perhaps in the future find a way to split the dates according to resources, and run a few matchings in parallel.
        """
        self.outfiles = []
        if newvariable:
            characteristics = ['-'.join([self.obs.basevar,self.obs.newvar]), self.season, self.cycle, self.obs.timemethod, self.obs.spacemethod]
        else:
            characteristics = [self.obs.basevar, self.season, self.cycle, self.obs.timemethod, self.obs.spacemethod]
        characteristics = [self.expname] + characteristics if hasattr(self, 'expname') else characteristics
        self.books_name = 'books_' + '_'.join(characteristics) + '.csv'
        books_path = self.basedir + self.books_name
        
        # Preparing the toplevel function kwargs.
        kwargs = {'obs': self.obs, 'newvariable':newvariable, 'newvarkwargs':newvarkwargs, 'loadkwargs':loadkwargs, 'matchtime' : matchtime, 'matchspace' :matchspace, 
                  'books_path':books_path, 'time_agg':self.time_agg, 'n_members':self.n_members, 'cycle':self.cycle, 'maxleadtime':self.maxleadtime}
        
        # Check if there was already something written. If not we start at the beginning of the current observation set.
        try:
            nextstartdate = pd.read_csv(books_path, usecols = ['tmax']).apply(pd.to_datetime).max().iloc[0] + pd.Timedelta('1D')
        except OSError:
            nextstartdate = self.dates[0]
        # Keep the sequential sub-process spwaning till all dates have been matched.
        terminate = False
        while nextstartdate <= self.dates.max() and (not terminate):
            filepath = self.basedir + '_'.join(characteristics) + '_' + uuid.uuid4().hex + '.h5'
            kwargs['outfilepath'] = filepath
            kwargs['datesubset'] = self.dates.loc[slice(nextstartdate, None, None)].copy()
            p = multiprocessing.Process(name = 'subset', target = matchforecaststoobs, kwargs = kwargs)
            p.start()
            p.join() # This forces the wait till termination.
            if p.exitcode == 0: # Correct execution
                self.outfiles.append(filepath)
                nextstartdate = pd.read_csv(books_path, usecols = ['tmax']).apply(pd.to_datetime).max().iloc[0] + pd.Timedelta('1D') # Read till where was matched and follow up.
            else:
                terminate = True
        
    def recollect(self, booksname = None):
        """
        Makes a dask dataframe object. The outfilenames can be collected by reading the timefiles. Or for instance by searching with a regular expression.
        """
        if hasattr(self, 'outfiles'):
            outfilenames = self.outfiles
        else:
            try:
                books = pd.read_csv(self.basedir + booksname)
                self.books_name = booksname
                outfilenames = books['file'].values.tolist()
            except AttributeError:
                pass
            
        self.alignedobject = dd.read_hdf(outfilenames, key = 'intermediate')
        

class Comparison(object):
    """
    All based on the dataframe format. This class at first instance requires the aligned object, posessing the raw forecasts and observations,
    and a climatology object which is used as a reference forecast. These two objects determine the naming of the scorefile to be created here
    From the raw forecasts, certain predictors can be computed and with those
    we can fit a supplied pp-model models in a cross-validation sense. This fit can be seperately exported
    and later imported again to make predictions, which once again can be exported.
    This is because the original idea was to have all members of all forecast types present as columns in the dataframe.
    However when sampling lots of members (100+) this fails when dask needs to collect all those for the scoring
    and they are discarded later. Therefore the fits and post-processed predicted values remain separate dataframes in the scorefile.
    Only their final score column is added to the original aligned object.
    For binary (event) observations, there already is a 'pi' column, inferred on-the-grid during matching from the raw members.
    For continuous variables, the quantile of the climatology is important, this will determine the event.
    """
    
    def __init__(self, alignment, climatology, modelclimatology = None):
        """
        The aligned object has Observation | raw forecast members |
        Potentially an external observed climatology object can be supplied that takes advantage of the full observed dataset. It has to have a location and dayofyear timestamp. Is it already aggregated?
        This climatology can be a quantile (used as the threshold for brier scoring) or it is a climatological probability if we have an aligned event predictor like POP. Or it is a random draw used for CRPS scoring. which means that multiple 'numbers' will be present.
        An optional model climatology can be supplied, to be used as a different exceedence threshold corresponging to a same quantile (only relevant for Brier Scoring)
        """
        self.frame = alignment.alignedobject
        self.basedir = '/nobackup_1/users/straaten/scores/'
        self.name = alignment.books_name[6:-4] + '_' + climatology.name
        self.grouperdowncasting = {'leadtime':'integer','latitude':'float','longitude':'float','clustid':'integer','doy':'integer'}
        # Used for bookkeeping on what is added to the frame:
        self.predcols = []
        
        # Casting the supplied climatology to the dataframe format
        climatology.clim.name = 'climatology'
        self.clim = climatology.clim.to_dataframe().dropna(axis = 0, how = 'any')
        # Some formatting to make merging with the two-level-columns aligned object easier
        if 'number' in self.clim.index.names: # If we are dealing with random draws. We are automatically creating a two level column index
            self.clim = self.clim.unstack('number')
        else: # We are dealing with a quantile or mean (probability) climatology
            if 'quantile' in climatology.clim.attrs: # We are dealing with a quantile climatology denoting the threshold and the actual reference forecast is the associated probability itself
                self.clim['threshold'] = self.clim['climatology']
                self.clim['climatology'] = np.array(1 - climatology.clim.attrs['quantile'], dtype = 'float32') # Making this reference forecast
            # Either way in both cases we have to manually create a two-level index.
            self.clim.columns = pd.MultiIndex.from_product([self.clim.columns, ['']], names = [None,'number'])
        self.clim.reset_index(inplace = True)
        for key in self.grouperdowncasting.keys():
            if key in self.clim.columns:
                self.clim[[key]] = self.clim[[key]].apply(pd.to_numeric, downcast = self.grouperdowncasting[key])
        
        try:
            self.quantile = climatology.clim.attrs['quantile']
        except KeyError:
            pass
        
        # Construction of the model climatology. These quantiles currently are not leadtime dependent so is going to be joined
        # with the normal climatology such that later both are joined on unique location and doy.
        # Be careful that the unit is correct (either Celsius or Kelvin amnomalies for temperature)
        if not modelclimatology is None:
            assert modelclimatology.clim.attrs['quantile'] == self.quantile
            modelclimatology.clim.name = 'modelclimatology' # Modelclimatology serves only to provide a threshold. Not as a reference forecast
            self.modelclim = modelclimatology.clim.to_dataframe().dropna(axis = 0, how = 'any')
            self.modelclim.columns = pd.MultiIndex.from_product([self.modelclim.columns, ['']], names = [None,'number'])
            # Joining does not need resetting the columns I think:
            if 'clustid' in self.clim.columns:
                self.clim = self.clim.merge(self.modelclim, how = 'outer', on = ['clustid','doy'])
            else:
                self.clim = self.clim.merge(self.modelclim, how = 'outer', on = ['clustid','latitude','longitude'], right_index = True)
            
    def compute_predictors(self, pp_model):
        """
        Computes simple predictors like ensmean and ensstd if they are not yet present as columns, and needed by the pp_model
        pp_model is an object from the fitting script. It has a fit method (returning parameters) and a predict method (requiring parameters).
        Currently adds those to the alignedobject frame. Does a bit of bookkeeping about what predcols were added
        """
        if not 'ensmean' in self.frame.columns:
            self.frame['ensmean'] = self.frame['forecast'].mean(axis = 1)
            self.predcols.append('ensmean')
        if pp_model.need_std and (not 'ensstd' in self.frame.columns):
            self.frame['ensstd']  = self.frame['forecast'].std(axis = 1)
            self.predcols.append('ensstd')
            
    def fit_pp_models(self, pp_model, groupers = ['leadtime','latitude','longitude'], nfolds = 3):
        """
        Starts the computation of simple predictors (next to the optional raw 'pi' for a binary variable)
        Groups the dask dataframe with the groupers, and then pushes these groups to an apply function. 
        In this apply a model fitting function is called which uses the predictor columns and the observation column.
        pp_model is an object from the fitting script. It has a fit method (returning parameters) and a predict method (requiring parameters).
        Computation is only invoked when exporting
        """

        def cv_fit(data, nfolds, fitfunc, modelcoefs, uniquetimes = False):
            """
            Acts per group (predictors should already be constructed). Calls a fitting function 
            in a time-cross-validation fashion. Supplies data as (nfolds - 1)/nfolds for training, 
            and writes the returned coeficients by the model to the remaining 1/nfolds part 
            (where the prediction will be made) as a dataframe. The returned Dataframe is indexed with time
            Which combines with the groupers to a multi-index.
            """
            
            nperfold = len(data) // nfolds
            foldindex = range(1,nfolds+1)
            
            # Pre-allocating a full size array structure for the eventual dataframe.
            coefs = np.zeros((len(data), len(modelcoefs)), dtype = 'float32')
                        
            # Stored by increasing time enables the cv to be done by placement index in data. (assuming equal group size). 
            # Times are ppossibly non-unique when also fitting to a spatial pool
            # Later we remove duplicate times (if present and we gave uniquetimes = False)
                        
            for fold in foldindex:
                test_ind = np.full((len(data),), False) # indexer to all data. 
                if (fold == foldindex[-1]):
                    test_ind[slice((fold - 1)*nperfold, None, None)] = True # Last fold gets all data till the end
                else:
                    test_ind[slice((fold - 1)*nperfold, (fold*nperfold), None)] = True
                
                # Calling of the fitting function on the training Should return an 1d array with the indices (same size as modelcoefs)           
                # Write into the full sized array, this converts 64-bit fitting result to float32
                # Use of inverse of the data test index to select train.
                coefs[test_ind] = fitfunc(data[~test_ind])
            
            # Returning only the unique time indices but this eases the wrapping and joining of the grouped results later on.
            if uniquetimes:
                return(pd.DataFrame(data = coefs, index = data['time'], columns = modelcoefs))
            else:
                duplicated = data['time'].duplicated(keep = 'first')
                return(pd.DataFrame(data = coefs[~duplicated], index = data['time'][~duplicated], columns = modelcoefs))
        
        # Computation of predictands for the models.
        self.compute_predictors(pp_model = pp_model)
        
        fitfunc = getattr(pp_model, 'fit')
        fitreturns = dict(itertools.product(pp_model.model_coefs, ['float32']))
        # Store some information. We have unique times when all possible groupers are provided, and there is no pooling.
        self.fitgroupers = groupers
        uniquetimes = all([(g in ['leadtime','latitude','longitude']) for g in groupers]) or all([(g in ['leadtime','clustid']) for g in groupers])
        
        # Actual (lazy) computation. Passing information to cv_fit.
        grouped = self.frame.groupby(groupers)        
        self.fits = grouped.apply(cv_fit, meta = fitreturns, **{'nfolds':nfolds, 
                                                                'fitfunc':fitfunc, 
                                                                'modelcoefs':pp_model.model_coefs,
                                                                'uniquetimes':uniquetimes})

    def merge_to_clim(self):
        """
        Merging ased on day-of-year and the spatial coordinate, returns a dataframe in which the (potentially numerous) columns
        from climatology are added (either a numeric quantile, the climatological probability or the random draws)
        """
        self.frame['doy'] = self.frame['time'].dt.dayofyear.astype('int16')
        if 'clustid' in self.frame.columns:
            return(self.frame.merge(self.clim, on = ['doy', 'clustid'], how = 'left'))
        else:
            return(self.frame.merge(self.clim, on = ['doy','latitude','longitude'], how = 'left'))
    
    def merge_to_fits(self):
        """
        Merges the fits (which when exported have a two-level column index) based on the variables used in fitting
        plus time because of the 3-fold cross validation.
        """
        joincolumns = ['time'] + self.fitgroupers
        return(self.frame.merge(self.fits, on = joincolumns, how = 'left'))

    def make_pp_forecast(self, pp_model, n_members = None, random = False):
        """
        Makes a probabilistic forecast based on already fitted models (either exported and imported, or externally supplied).
        From that fit it gets the coefficients (indexed by time and fitgroupers), from the aligned frame the compited predictors, and 
        with predict method of the pp_model a new dataframe is created. (computation invoked upon exporting)
        If n_members is not None, then we are forecasting n members from the fitted distribution, either randomly or with the weibull equidistant estimator
        If n_members is None then there are two options: 
        1) The event for which we forecast is already present as an observed binary variable and we use the Logistic model for the probability of 1. 
        2) The event is the exceedence of the quantile beloning to the climatology in this object and we employ scipy implementation of the normal model.
        """
        self.compute_predictors(pp_model = pp_model) # computation if not present yet (fit was loaded, fitted somewhere else)
        frameplusfits = self.merge_to_fits() # frameplusfits contains in each row coefficients and predictor values,
        
        predfunc = getattr(pp_model, 'predict')
        
        # Generation of predictions gives a frame of equal length (axis = 0) and equal ordering as the aligned frame.
        if n_members is not None:
            self.preds = frameplusfits.map_partitions(predfunc, **{'n_draws':n_members, 'random':random}) # choice for random or equidistant is made in the method of the pp_model
            self.preds.columns = pd.MultiIndex.from_product([['corrected'],self.preds.columns])
        else:
            if isinstance(pp_model, Logistic):
                # In this case we have an event variable and fitted a logistic model. And predict with that parametric model.
                self.preds = frameplusfits.map_partitions(predfunc) #  meta = {'corrected':'float32'}
                
            else:
                # in this case continuous and we predict exceedence of the climatological threshold column. So we need an extra merge and redo the merge to fits
                self.frame = self.merge_to_clim()
                frameplusfits = self.merge_to_fits()
                self.preds = frameplusfits.map_partitions(predfunc, **{'quant_col':'threshold'}) # meta = {'corrected':'float32'}
            self.preds.columns = pd.MultiIndex.from_product([['corrected'],['']])
            
        self.preds[['observation','time'] + self.fitgroupers] = frameplusfits[['observation','time'] + self.fitgroupers] # Also include the observation. This way we can directly score on the to-be exported frame
        
    def brierscore(self):
        """
        Computes the climatological and raw scores. The raw scores uses a different threshold when a 'modelclimatology' column is present
        Also computes the score of the predictions from the post-processed model this attribute was created beforehand
        """
        # Merging with climatological file. This contains either the quantile to be exceeded and the probability, or the mean probability of the binary event. When NGR predictions were made for the first case, climatology is already merged to the frame.
        if not ('climatology' in self.frame.columns):
            self.frame = self.merge_to_clim() # Potentially also adds the 'modelclimatology' if that object was supplied at __init__
                    
        # Creation of the raw exceedence probability, by computing boolean exceedences of the threshold
        # And creation of the binary observation. Only invoked when there has not been an eventclassification previously, like pop. In that case 
        if not ('pi' in self.frame.columns):
            print('lazy pi and bool observation construction')
            def boolwrap(frame, quant_col):
                bools = frame['forecast'].values > frame[quant_col].values[:,np.newaxis]
                return(bools.mean(axis=1).astype('float32')) # will probably fail when NA is present
            self.frame['pi'] = self.frame.map_partitions(boolwrap, **{'quant_col':'modelclimatology' if hasattr(self, 'modelclim') else 'threshold'}, meta = ('pi','float32')) # The threshold that potentially comes from modelclimatology, otherwise observed climatology
            def obswrap(frame):
                exceedence = frame['observation'].values.squeeze() > frame['threshold'].values.squeeze() # to make sure that comparison is 1D
                return exceedence.astype('float32')
            self.frame['observation'] = self.frame.map_partitions(obswrap, meta = ('observation','float32'))
            #self.frame['observation'] = self.frame['observation'] > self.frame['threshold'] # The threshold from observed climatology

        # Merge the predictions if available. Only one column 'corrected', we don't want to get the 'observation' column into the merge
        if hasattr(self, 'preds'):
            self.frame = self.frame.merge(self.preds[['time','corrected'] + self.fitgroupers], on = ['time'] + self.fitgroupers, how = 'left')

        def scorewrap(frame, forecasttype):
            score = (frame[forecasttype].values.squeeze() - frame['observation'].values.squeeze())**2
            return score.astype('float32')
        for forecasttype in ['pi','climatology','corrected']:
            if forecasttype in self.frame.columns:
                scorename = forecasttype + '_bs'
                self.frame[scorename] = self.frame.map_partitions(scorewrap, **{'forecasttype':forecasttype}, meta = (scorename,'float32'))
            
    def crpsscore(self):
        """
        This method goes over the respective frames for the raw, the climatological (and potentially the post-processed) forecasts.
        The score resulting from all of those frames are written as a column to the original aligned frame
        Discrete crps scoring by use of the properscoring package. For a small number of members (11),
        the overestimation compared to the gaussian analytical form is about 20%. 
        For good comparison the climatology, the forecast, and corrected should be scored with the same number of members.
        """
        # Preparation of the frames
        scoreframe = self.merge_to_clim()
        frames = {'climatology': scoreframe,
                  'forecast':scoreframe}
        if hasattr(self, 'preds'):
            frames.update({'corrected':self.preds})
        
        def crps_wrap(frame, forecasttype):
            """
            Wrapper for discrete ensemble crps scoring. Finds observations and forecasts to supply to the properscoring function
            """
            return(ps.crps_ensemble(observations = frame['observation'], forecasts = frame[forecasttype]).astype('float32')) # Not sure if this will provide the needed array type to ps
        
        for forecasttype in frames.keys(): 
            scorename = forecasttype + '_crps' 
            frames[forecasttype][scorename] = frames[forecasttype].map_partitions(crps_wrap, **{'forecasttype':forecasttype}, meta = (scorename,'float32'))
        
        # Joining the computed results. Hopefully this does not join all the columns.
        joincolumns = ['time'] + self.fitgroupers 
        self.frame = self.frame.merge(scoreframe[['climatology_crps','forecast_crps']+joincolumns], on = joincolumns, how = 'left')
        if hasattr(self, 'preds'):
            self.frame = self.frame.merge(self.preds[['corrected_crps'] + joincolumns], on = joincolumns, how = 'left')
    
    def export(self, fits = False, frame = False, preds = False, store_minimum = False):
        """
        Put both in the same hdf file, but different key. So append mode. If frame than writes one dataframe for self.frame
        Columns have already been downcased in the previous steps. If store_minimum the function tries to drop forecast members, model coefficients, climatology column. Boolean columns are always dropped.
        """

        self.filepath = self.basedir + self.name + '.h5'
        if fits:
            # Temporary store to file, to write the grouping multi-index. Then reset the indices, downcast where possible and write the final one
            tempfile = self.basedir + uuid.uuid4().hex + '.h5'
            self.fits.to_hdf(tempfile, key = 'fits', scheduler = 'threads')
            print('all models have been fitted')
            self.fits = dd.read_hdf(tempfile, key = 'fits').reset_index()
            for group in list(self.grouperdowncasting.keys()):
                try:
                    self.fits[group] = self.fits[group].map_partitions(pd.to_numeric, **{'downcast':self.grouperdowncasting[group]}) # pd.to_numeric(self.fits[group], downcast = self.grouperdowncasting[group])
                except KeyError:
                    pass
            self.fits.columns = pd.MultiIndex.from_product([self.fits.columns, ['']]) # To be able to merge with self.frame
            self.fits.to_hdf(self.filepath, key = 'fits', format = 'table', **{'mode':'a'})
            self.fits = dd.read_hdf(self.filepath, key = 'fits')
            print('all models downcasted, saved and reloaded')
            os.remove(tempfile)
        if frame:
            if store_minimum:
                discard = ['climatology','threshold','modelclimatology','forecast','corrected','doy','pi'] + self.predcols 
                self.frame = self.frame.drop(discard, axis = 1, errors = 'ignore')
            
            self.frame.to_hdf(self.filepath, key = 'scores', format = 'table', **{'mode':'a'})
        
        if preds:
            self.preds.to_hdf(self.filepath, key = 'preds')
            self.preds = dd.read_hdf(self.filepath, key = 'preds', **{'mode':'a'})
        
        return(self.name)
        
class ScoreAnalysis(object):
    """
    Contains several ways to analyse an exported file with scores. 
    Its main function is to calculate skill-scores. 
    It can export bootstrapped skillscore samples and later analyze these
    For spatial mean scores and forecast horizons.
    """
    def __init__(self, scorefile, timeagg, rolling = False):
        """
        Provide the name of the exported file with the scores.
        Change here the quantiles that are exported by bootstrapping procedures.
        """
        self.basedir = '/nobackup_1/users/straaten/scores/'
        self.scorefile = scorefile
        self.filepath = self.basedir + self.scorefile + '.h5'
        self.timeagg = timeagg
        self.rolling = rolling
        self.quantiles = [0.025,0.5,0.975]
        
    def load(self):
        """
        Searches for relevant columns to load. Should find either _bs in certain columns or _crps
        Also searches for what the spatial coordinates are, either ['latitude','longitude'] or ['clustid']
        """
        with pd.HDFStore(path=self.filepath, mode='r') as hdf:
            allcolslevelzero = pd.Index([ tup[0] for tup in  hdf.get_storer('scores').non_index_axes[0][1] ])
        if allcolslevelzero.str.contains('_bs').any():
            self.scorecols = allcolslevelzero[allcolslevelzero.str.contains('_bs')].tolist()
            self.output = '_bss'
        elif allcolslevelzero.str.contains('_crps').any():
            self.scorecols = allcolslevelzero[allcolslevelzero.str.contains('_crps')].tolist()
            self.output = '_crpss'
        if 'clustid' in allcolslevelzero:
            self.spatcoords = ['clustid']
        else:
            self.spatcoords = ['latitude','longitude']
        self.frame = dd.read_hdf(self.filepath, key = 'scores', columns = self.scorecols + self.spatcoords + ['leadtime'])
        self.frame.columns = self.frame.columns.droplevel(1)
        
        # Some preparation for computing and returning scores in the methods below.
        self.returncols = [ col.split('_')[0] + self.output for col in self.scorecols]
        self.climcol = [col for col in self.scorecols if col.startswith('climatology')][0]
    
    def eval_skillscore(self, data):
        """
        Is supplied an isolated data group on which to evaluate skill of mean brierscores.
        These relevant columns are given as the scorecols.
        Returns a series with same length as scorecols. Climatology obviously has skill 1
        """
        meanscore = data[self.scorecols].mean(axis = 0)
        returns = np.zeros((len(self.returncols),), dtype = 'float32')

        for scorecol, returncol in zip(self.scorecols, self.returncols):
            returns[self.returncols.index(returncol)] = np.array(1, dtype = 'float32') - meanscore[scorecol]/meanscore[self.climcol] 

        return(pd.Series(returns, index = self.returncols, name = self.output))
        
    def mean_skill_score(self, groupers = ['leadtime']):
        """
        Grouping. Average and compute skill score.
        Always a mid-point leadtime correction is performed.
        """
        grouped =  self.frame.groupby(groupers)
        
        scores = grouped.apply(self.eval_skillscore, 
                               meta = pd.DataFrame(dtype='float32', columns = self.returncols, index=[self.output])).compute()
        
        scores = assignmidpointleadtime(scores, timeagg = self.timeagg)
        
        return(scores)
    
    def characteristiclength(self):
        """
        Computes a characteristic length per location to use as bootstrapping block length.
        According the formula of Leith 1973 which is reference in Feng (2011)
        T_0 = 1 + 2 * \sum_{\tau= 1}^D (1 - \tau/D) * \rho_\tau
        The number is in the unit of the time-aggregation of the variable. (nr days if daily timeseries, which rolling aggregated ones are)
        Note: should the use of the charlengths have to be adapted for the small datasets? Cannot be 7 weeks or something.
        """
        def auto_cor(df, freq, cutofflag = 20, return_char_length = True):
            """
            Computes the lagged autocorrelation in df['observation'] starting at lag one till cutofflag.
            It is assumed that the rows in the dataframe are unique timesteps
            For the scoresets written in experiment 2 this is a correct assumption.
            Uses the time column to make the correct shifts, with the native freq of the variable (difficult to infer)
            """
            res = np.zeros((cutofflag,), dtype = 'float32')
            for i in range(cutofflag):
                series = df[['observation','time']].drop_duplicates().set_index('time')['observation']
                # Use pd.Series capabilities.
                res[i] = series.corr(series.shift(periods = i + 1, freq = '1D' if self.rolling else freq).reindex(series.index))
                    
            if return_char_length: # Formula by Leith 1973, referenced in Feng (2011)
                return(np.nansum(((1 - np.arange(1, cutofflag + 1)/cutofflag) * res * 2)) + 1)
            else:
                return(res)
                
        # Read an extra column. Namely the observed column on which we want to compute the length.
        tempframe = dd.read_hdf(self.filepath, key = 'scores', columns = ['observation','time'] + self.spatcoords)
        tempframe.columns = tempframe.columns.droplevel(1)
        # Do stuff per location
        self.charlengths = tempframe.groupby(self.spatcoords).apply(auto_cor, meta = ('charlength','float32'), **{'freq':self.timeagg,'cutofflag':20}).compute()
        
        # Quality control needed. Values below 1 are set to 1 (reduces later to normal bootstrapping)
        self.charlengths[self.charlengths < 1] = 1
    
    def block_bootstrap_local_skills(self, n_samples = 200, fixsize = True):
        """
        At each random instance picks a local time-block-bootstraped sample and computes local skill for each location.
        Full grouping. returns 200 samples per group. Either does a full fraction = 1 sampling
        But also has the possibility to fix the sample size: either with providing an integer to fixsize or according to the minimum availabilities in the loaded dataset, thereby under-sampling the smaller groups.
        Exports the bootstrap samples to the scorefile. Such that later quantiles can be computed.
        Returns a true if completed
        """        
        if (not hasattr(self, 'charlengths')):
            self.characteristiclength()
                
        # Get the maximum count. The goal is to equalize this accross leadtimes.
        if fixsize:
            maxcount = fixsize
            if not isinstance(fixsize, int):
                maxcount = self.frame.groupby(['leadtime'] + self.spatcoords)[self.climcol].count().max().compute()
            print(maxcount)
        
        def fix_sample_score(df, n_samples, fixed_size = None):
            """
            Acts per location/leadtime group. 
            Assumes that all rows in the supplied data are unique timesteps.
            Returns a dataframe of 3 cols and n_samples rows.
            Block bootstrapping splits the sets of rows in a non-overlapping sense in blocks with the local blocklength.
            These splits currently might jump into the new season. Otherwise it will be more expensive to track consequtive months
            with local records of uneven length.
            The local blocklength is read from the self.charlengths attribute.
            """
            blocklength = self.charlengths.loc[tuple(df[self.spatcoords].iloc[0])]
            #print(blocklength)
            blocklength = int(np.ceil(blocklength))
            setlength = len(df)
            if fixed_size is None:
                size = setlength
            else:
                size = fixed_size
            rowsets = pd.Series(np.array_split(np.arange(setlength), np.arange(blocklength,setlength,blocklength))) # produces a list of arrays
            
            local_result = [None] * n_samples            
            for i in range(n_samples):
                if blocklength == 1:
                    sample = df[self.scorecols].sample(n = size, replace = True)
                else:
                    choice = rowsets.sample(n = size // blocklength, replace = True)
                    choice = np.concatenate(choice.values) # Glue the chosen sets back to a listchoice = np.concatenate(choice.values)
                    sample = df.iloc[choice,:] # Duplicate indices are allowed in iloc.
                
                local_result[i] = self.eval_skillscore(sample)
            
            return(pd.DataFrame(local_result, index = pd.RangeIndex(n_samples, name = 'samples')))
        
        grouped =  self.frame.groupby(['leadtime'] + self.spatcoords) # Dask grouping
        sample_scores = grouped.apply(fix_sample_score,
                               meta = pd.DataFrame(dtype='float32', columns = self.returncols, index = pd.RangeIndex(n_samples)),
                               **{'n_samples':n_samples, 'fixed_size':maxcount if fixsize else None })
        
        sample_scores.to_hdf(self.filepath, key = 'bootstrap',  format = 'table', **{'mode':'a'})
        return(True)
    
    def process_bootstrapped_skills(self, local = True, fitquantiles = False, forecast_horizon = True, skillthreshold = 0, average_afterwards = False):
        """
        Rework all sampled local skills to a spatial mean for each random instance. Or let the scores remain local.
        Compute quantiles on these skills. Standard empirical, option is quantile fitted exponential model
        Option to rework this to forecast horizon. Option to average these spatially afterwards when dealing with local scores.
        The options can be seen as a sequential procedure, for the forcast horizon quantiles are needed.
        Always a mid-point leadtime correction is performed.
        """
        skills = pd.read_hdf(self.filepath, key = 'bootstrap') # DaskDataframe would give issues related to this https://github.com/dask/dask/issues/4643
        groupers = list(skills.index.names)
        if local:
            groupers.remove('samples')
        else:
            groupers = ['leadtime']
            skills = skills.groupby(['leadtime','samples']).mean()
        
        # Extract quantiles from the samples. Removes samples as an index. Eventually I want to go to dask. But the  grouped.apply combination is giving trouble
        #meta = dict(itertools.product(skills.columns.to_list(), ['float32']))
        if fitquantiles:
            # Not grouping by leadtime here, because we will fit against leadtime
            groupers.remove('leadtime')
            def fit_quantiles(df, fullindex = True):
                model1 = ExponentialQuantile(obscol=skills.columns[0])
                model1.fit(train = df, quantiles=self.quantiles, startx=None, endx=None)
                pred1 = model1.predict(test= df, quantiles = self.quantiles, startx=None, endx=None, restoreindex = fullindex)
                model2 = ExponentialQuantile(obscol=skills.columns[-1])
                model2.fit(train = df, quantiles=self.quantiles, startx=None, endx=None)
                pred2 = model2.predict(test= df, quantiles = self.quantiles, startx=None, endx=None, restoreindex = fullindex)
                return(pred1.merge(pred2, left_index = True, right_index = True))
            if not groupers:
                quants = fit_quantiles(skills)
            else:
                grouped = skills.groupby(groupers)
                quants = grouped.apply(fit_quantiles, **{'fullindex':False})
        else:
            grouped = skills.groupby(groupers)
            quants = grouped.quantile(self.quantiles) # Quantile is not a dask dataframe grouping method. Also the apply seems to give problems
            quants.index.rename('quantile',level = -1, inplace = True)
            
        #quants.compute()
        # Midpoint correction using self.timeagg. (Converted to integer value in the helper function)
        quants = assignmidpointleadtime(quants, timeagg = self.timeagg)
        
        if not forecast_horizon:
            return(quants)
        else:
            data = quants.unstack(level = 'quantile').loc(axis = 1)[slice(None), np.min(self.quantiles)] # We want the lower bound
            try:
                groupers.remove('leadtime') # Still no grouping with leadtime, try to remove it if it has not been removed in the quantile fitting
            except ValueError:
                pass
            def get_f_hor(df, skillthreshold):
                horizons = pd.Series(np.nan, index = df.columns.get_level_values(0), name = 'horizon')
                for col in df.columns:
                    horizons.loc[col[0]] = lastconsecutiveabove(df.loc[:,col], threshold=skillthreshold)
                return(horizons)
            if not groupers:
                result = get_f_hor(data, skillthreshold=skillthreshold)
            else:
                result = data.groupby(groupers).apply(get_f_hor, **{'skillthreshold':skillthreshold})
                if average_afterwards:
                    result = result.mean(axis = 0)
            return(result)

if __name__ == '__main__':
    mc = ModelClimatology('45r1','tg-anom', **{'name':'tg-anom_45r1_1998-06-07_1999-05-16_1D_0.3-tg-DJF-mean_5_5_q0.15'}) #
    mc.local_clim()
    al = ForecastToObsAlignment('DJF','45r1')
    al.recollect(booksname = 'books_clustga25_tg-anom_DJF_45r1_1D_0.3-tg-DJF-mean.csv')
    cl = Climatology('tg-anom', **{'name':'tg-anom_clim_1998-01-01_2018-12-31_1D_0.3-tg-DJF-mean_5_5_q0.15'})
    cl.localclim()
    self = Comparison(alignment=al, climatology=cl, modelclimatology=mc) # Add possibility for a model_climatology that is processed upon initialization?
#    self.basedir = '/nobackup/users/straaten/'
#    self.fit_pp_models(NGR(), groupers = ['clustid','leadtime'])
#    self.export(fits = True, frame=False)
#    self.make_pp_forecast(NGR())
#    self.brierscore()
#    filename = self.export(fits = False, frame = True, store_minimum = False)
#    
#    sc = ScoreAnalysis(scorefile = filename, timeagg = '1D', rolling=True)
#    sc.basedir = '/nobackup/users/straaten/'
#    sc.load()
#highresobs = SurfaceObservations('tg')
#highresobs.load(tmin = '2000-01-01', tmax = '2005-12-31',  llcrnr= (64,40))
#highresobs.minfilter(season = 'DJF', n_min_per_seas = 80)
#highresobs.aggregatespace(level = 0.01, clustername = 'tg-JJA', method = 'mean') 
#        
#highresclim = Climatology(highresobs.basevar)
#highresclim.localclim(obs = highresobs, mean = False, n_draws=11, daysbefore = 5, daysafter = 5)
#       
#align = ForecastToObsAlignment('JJA', '45r1')
#align.recollect(booksname = 'books_tg-anom_JJA_45r1_1D_0.01-tg-JJA-mean.csv')
#
#comp = Comparison(align, climatology=highresclim)
#comp.fit_pp_models(NGR(), groupers = ['clustid'])
#comp.make_pp_forecast(NGR(), n_members=11)
#comp.crpsscore()
#comp.export(fits = False, frame = True)
#clustga25_tg-anom_DJF_45r1_7D-roll-mean_0.2-tg-DJF-mean_tg-anom_clim_1998-01-01_2018-12-31_7D-roll-mean_0.2-tg-DJF-mean_5_5_q0.75.h5            
#clim = Climatology('tg-anom',**{'name':'tg-anom_clim_1998-01-01_2018-12-31_7D-roll-mean_0.2-tg-DJF-mean_5_5_q0.75'})
#clim.localclim()

#align = ForecastToObsAlignment('DJF', '45r1')
#align.recollect(booksname = 'books_clustga25_tg-anom_DJF_45r1_7D-roll-mean_0.2-tg-DJF-mean.csv')

#comp = Comparison(align, climatology = clim)
#comp.name = 'meanclimtest'
#comp.fits = dd.read_hdf('/nobackup_1/users/straaten/scores/meanclimtest.h5', key = 'fits')
#comp.fitgroupers = ['leadtime','clustid']
#comp.fit_pp_models(NGR(), groupers = ['leadtime','clustid'])
#comp.export(fits = True)
#comp.make_pp_forecast(NGR())
#comp.preds = dd.read_hdf('/nobackup_1/users/straaten/scores/meanclimtest.h5', key = 'preds')
#dat = pd.read_hdf('/nobackup_1/users/straaten/scores/meanclimtest.h5', key = 'scores')
#sc = ScoreAnalysis(scorefile = 'clustga33_tg-anom_DJF_45r1_11D-roll-mean_0.05-tg-DJF-mean_tg-anom_clim_1998-01-01_2018-12-31_11D-roll-mean_0.05-tg-DJF-mean_5_5_equi100', timeagg = '11D', rolling = True)
#msc = sc.process_bootstrapped_skills(local = True, fitquantiles = False, forecast_horizon = True, skillthreshold= 0, average_afterwards=False)
#
#obs = SurfaceObservations('tg', **{'name':'tg-anom_1998-06-07_2018-12-31_11D-roll-mean_0.05-tg-DJF-mean'})
#obs.load()
#mscgeo = georeference(msc, obs.clusterarray)
