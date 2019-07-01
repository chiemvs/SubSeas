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
from observations import SurfaceObservations, Climatology, EventClassification
from forecasts import Forecast
from helper_functions import monthtoseasonlookup, unitconversionfactors, lastconsecutiveabove, assignmidpointleadtime
from fitting import NGR, Logistic, ExponentialQuantile

class ForecastToObsAlignment(object):
    """
    Idea is: you have already prepared an observation class, with a certain variable, temporal extend and space/time aggregation.
    This searches the corresponding forecasts, and possibly forces the same aggregations.
    TODO: not sure how to handle groups yet.
    """
    def __init__(self, season, cycle, observations = None, **kwds):
        """
        Temporal extend can be read from the attributes of the observation class. 
        Specify the season under inspection for subsetting.
        """
        self.basedir = '/nobackup/users/straaten/match/'
        self.cycle = cycle
        self.season = season
        if observations is not None:
            self.obs = observations
            self.dates = self.obs.array.coords['time'].to_series() # Left stamped
            self.dates = self.dates[monthtoseasonlookup(self.dates.index.month) == self.season]
            # infer dominant time aggregation in days
            timefreq = self.obs.timemethod.split('-')[0]
            self.time_agg = int(pd.date_range('2000-01-01','2000-12-31', freq = timefreq).to_series().diff().dt.days.mode())
            self.maxleadtime = 46
        
        for key in kwds.keys():
            setattr(self, key, kwds[key])
    
    def find_forecasts(self):
        """
        Here the forecasts corresponding to the observations are determined, by testing their existence.
        Hindcasts and forecasts might be mixed. But they are in the same class.
        Leadtimes may differ.
        """
        self.forecasts = {}
        for date in self.dates.tolist():
            # Find forecast initialization times that fully contain the observation date (including its window for time aggregation)
            containstart = date + pd.Timedelta(str(self.time_agg) + 'D') - pd.Timedelta(str(self.maxleadtime) + 'D')
            containend = date
            contain = pd.date_range(start = containstart, end = containend, freq = 'D').strftime('%Y-%m-%d')
            forecasts = [Forecast(indate, prefix = 'for_', cycle = self.cycle) for indate in contain]
            hindcasts = [Forecast(indate, prefix = 'hin_', cycle = self.cycle) for indate in contain]
            # select from potential forecasts only those that exist.
            forecasts = [f for f in forecasts if os.path.isfile(f.basedir + f.processedfile)]
            hindcasts = [h for h in hindcasts if os.path.isfile(h.basedir + h.processedfile)]
            self.forecasts.update({date : forecasts + hindcasts})   
        
    def load_forecasts(self, n_members, loadkwargs = {}):
        """
        Gets the daily processed forecasts into memory. Delimited by the left timestamp and the aggregation time.
        This is done by using the load method of each Forecast class in the dictionary. They are stored in a list.
        Loadkwargs can carry the delimiting spatial corners if only a part of the domain is desired.
        """
        for date, listofforecasts in self.forecasts.items():
            
            if listofforecasts: # empty lists are skipped
                tmin = date
                tmax = date + pd.Timedelta(str(self.time_agg - 1) + 'D') # -1 because date itself also counts for one day in the aggregation.
                
                for forecast in listofforecasts:
                    forecast.load(variable = self.obs.basevar, tmin = tmin, tmax = tmax, n_members = n_members, **loadkwargs)
        
        self.n_members = n_members
     
    def force_resolution(self, forecast, time = True, space = True):
        """
        Force the observed resolution onto the supplied Forecast. Checks if the same resolution and force spatial/temporal aggregation if that is not the case. Checks will fail on the roll-norm difference, e.g. 2D-roll-mean observations and .
        Makes use of the methods of each Forecast class. Checks can be switched on and off. Time-rolling is never applied to the forecasts as for each date already the precise window was loaded, but space-rolling is.
        """                
        if time:
            # Check time aggregation
            obstimemethod = getattr(self.obs, 'timemethod')
                        
            try:
                fortimemethod = getattr(forecast, 'timemethod')
                if not fortimemethod == obstimemethod:
                    raise AttributeError
                else:
                    print('Time already aligned')
            except AttributeError:
                print('Aligning time aggregation')
                freq, rolling, method = obstimemethod.split('-')
                forecast.aggregatetime(freq = freq, method = method, keep_leadtime = True, ndayagg = self.time_agg, rolling = False)
        
        if space:
            # Check space aggregation
            obsspacemethod = getattr(self.obs, 'spacemethod')
                
            try:
                forspacemethod = getattr(forecast, 'spacemethod')
                if not forspacemethod == obsspacemethod:
                    raise AttributeError
                else:
                    print('Space already aligned')
            except AttributeError:
                print('Aligning space aggregation')
                step, what, rolling, method = obsspacemethod.split('-')
                forecast.aggregatespace(step = float(step), method = method, by_degree = (what == 'degrees'), rolling = (rolling == 'roll'))
    
    def force_units(self, forecast):
        """
        Linear conversion of forecast object units to match the observed units.
        """
        a,b = unitconversionfactors(xunit = forecast.array.units, yunit = self.obs.array.units)
        forecast.array = forecast.array * a + b
        forecast.array.attrs = {'units':self.obs.array.units}
                      
    def force_new_variable(self, forecast, newvarkwargs = {}, inplace = True):
        """
        Call upon event classification on the forecast object to get the on-the-grid conversion of the base variable.
        This is classification method is the same as applied to obs and is determined by the similar name.
        Possibly returns xarray object if inplace is False.
        """
        newvariable = self.obs.newvar
        method = getattr(EventClassification(obs = forecast, **newvarkwargs), newvariable)
        return(method(inplace = inplace))
        
                
    def match_and_write(self, newvariable = False, newvarkwargs = {}, matchtime = True, matchspace = True):
        """
        Neirest neighbouring to match pairs. Be careful when domain of observations is larger.
        Determines the order in which space-time aggregation and classification is done based on the potential newvar.
        Also calls unit conversion for the forecasts. (the observed units of a newvariable are actually its old units)
        Creates the dataset and writes it to disk. Possibly empties basket and writes to disk 
        at intermediate steps if intermediate results press too much on memory.
        """
        import uuid
        from datetime import datetime
        
        aligned_basket = []
        self.outfiles = []
        
        # Make sure the first parts are recognizable as the time method and the space method. Make last part unique
        # Possibly I can also append in h5 files.
        def write_outfile(basket, newvariable = False):
            """
            Saves to a unique filename and creates a bookkeeping file (appends if it already exists)
            """
            if newvariable:
                characteristics = ['-'.join([self.obs.basevar,self.obs.newvar]), self.season, self.cycle, self.obs.timemethod, self.obs.spacemethod]
            else:
                characteristics = [self.obs.basevar, self.season, self.cycle, self.obs.timemethod, self.obs.spacemethod]
            filepath = self.basedir + '_'.join(characteristics) + '_' + uuid.uuid4().hex + '.h5'
            characteristics = [self.expname] + characteristics if hasattr(self, 'expname') else characteristics
            self.books_name = 'books_' + '_'.join(characteristics) + '.csv'
            books_path = self.basedir + self.books_name
            
            dataset = pd.concat(basket)
            dataset.to_hdf(filepath, key = 'intermediate', format = 'table')
            
            books = pd.DataFrame({'file':[filepath],
                                  'tmax':[dataset.time.max().strftime('%Y-%m-%d')],
                                  'tmin':[dataset.time.min().strftime('%Y-%m-%d')],
                                  'unit':[self.obs.array.units],
                                  'write_date':[datetime.now().strftime('%Y-%m-%d_%H:%M:%S')]})
            
            try:
                with open(books_path, 'x') as f:
                    books.to_csv(f, header=True, index = False)
            except FileExistsError:
                with open(books_path, 'a') as f:
                    books.to_csv(f, header = False, index = False)
            
            print('written out', filepath)
            self.outfiles.append(filepath)
        
        for date, listofforecasts in self.forecasts.items():
            
            if listofforecasts:
                
                for forecast in listofforecasts:
                    self.force_units(forecast) # Get units correct.
                    if newvariable:
                        if self.obs.newvar == 'anom':
                            self.force_new_variable(forecast, newvarkwargs = newvarkwargs, inplace = True) # If newvar is anomaly then first new variable and then aggregation. If e.g. newvar is pop then first aggregation then transformation
                            self.force_resolution(forecast, time = matchtime, space = matchspace)
                            forecast.array = forecast.array.swap_dims({'time':'leadtime'}) # So it happens inplace
                        elif self.obs.newvar == 'pop': # This could even be some sort of 'binary newvariable' criterion.
                            self.force_resolution(forecast, time = matchtime, space = matchspace)
                            forecast.array = forecast.array.swap_dims({'time':'leadtime'}) # So it happens inplace
                            try:
                                listofbinaries.append(self.force_new_variable(forecast, newvarkwargs = newvarkwargs, inplace = False))
                            except NameError:
                                listofbinaries = [self.force_new_variable(forecast, newvarkwargs = newvarkwargs, inplace = False)]
                    else:
                        self.force_resolution(forecast, time = matchtime, space = matchspace)
                        forecast.array = forecast.array.swap_dims({'time':'leadtime'}) # So it happens inplace
                
                # Get the correct observed and inplace forecast arrays.
                fieldobs = self.obs.array.sel(time = date).drop('time')
                allleadtimes = xr.concat(objs = [f.array for f in listofforecasts], 
                                                 dim = 'leadtime') # concatenates over leadtime dimension.
                exp = allleadtimes.reindex_like(fieldobs, method='nearest') # Select nearest neighbour forecast for each observed point in space.
                
                # When we have created a binary new_variable like pop, the forecastlist was appended with the not-inplace transformed ones 
                # of These we only want to retain the ensemble probability (i.e. the mean).
                try:
                    binary = xr.concat(objs = listofbinaries,
                                       dim = 'leadtime') # concatenates over leadtime dimension.
                    listofbinaries.clear() # Empty for next iteration
                    pi = binary.reindex_like(fieldobs, method='nearest').mean(dim = 'number') # only the probability of the event over the members is retained
                    # Merging, exporting to pandas and masking by dropping on NA observations.
                    combined = xr.Dataset({'forecast':exp.drop('time'),'observation':fieldobs, 'pi':pi.drop('time')}).to_dataframe().dropna(axis = 0)
                    # Unstack creates duplicates. Two extra columns (obs and pi) need to be selected. Therefore the iloc
                    temp = combined.unstack('number').iloc[:,np.append(np.arange(0,self.n_members + 1), self.n_members * 2)]
                    temp.reset_index(inplace = True) # places latitude and all in 
                    labels = temp.columns.labels[1].tolist()
                    labels[-2:] = np.repeat(self.n_members, 2)
                except NameError:
                    # Merging, exporting to pandas and masking by dropping on NA observations.
                    combined = xr.Dataset({'forecast':exp.drop('time'), 'observation':fieldobs}).to_dataframe().dropna(axis = 0)
                    # Handle the multi-index
                    # first puts number dimension into columns. observerations are duplicated so therefore selects up to n_members +1
                    temp = combined.unstack('number').iloc[:,:(self.n_members + 1)]
                    temp.reset_index(inplace = True) # places latitude and all in 
                    labels = temp.columns.labels[1].tolist()
                    labels[-1] = self.n_members
                
                temp.columns.set_labels(labels, level = 1, inplace = True)
                
                # Downcasting latitude and longitude
                temp[['latitude', 'longitude']] = temp[['latitude', 'longitude']].apply(pd.to_numeric, downcast = 'float')
                # Downcasting the leadtime column
                temp['leadtime'] = np.array(temp['leadtime'].values.astype('timedelta64[D]'), dtype = 'int8')
                
                # prepend with the time index.
                temp.insert(0, 'time', date)
                aligned_basket.append(temp)
                self.forecasts[date].clear()
                print(date, 'matched')
                
                # If aligned takes too much system memory (> 1Gb) . Write it out
                if sys.getsizeof(aligned_basket[0]) * len(aligned_basket) > 1*10**9:
                    write_outfile(aligned_basket, newvariable=newvariable)
                    aligned_basket = []
        
        # After last loop also write out 
        if aligned_basket:
            write_outfile(aligned_basket, newvariable=newvariable)
        
        
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
    All based on the dataframe format. No arrays anymore. In this class the aligned object is grouped in multiple ways to fit models in a cross-validation sense. 
    To make predictions and to score, raw, post-processed and climatological forecasts
    The type of scoring and the type of prediction is inferred from whether certain columns are present
    For binary (event) observations, there already is a 'pi' column, inferred on-the-grid during matching from the raw members.
    For continuous variables, the quantile of the climatology is important, this will determine the event.
    """
    
    def __init__(self, alignment, climatology):
        """
        The aligned object has Observation | members |
        Potentially an external observed climatology object can be supplied that takes advantage of the full observed dataset. It has to have a location and dayofyear timestamp. Is it already aggregated?
        This climatology can be a quantile (used as the threshold for brier scoring) or it is a climatological probability if we have an aligned event predictor like POP. Or it is a random draw used for CRPS scoring. which means that multiple 'numbers' will be present.
        """
        self.frame = alignment.alignedobject
        self.basedir = '/nobackup/users/straaten/scores/'
        self.name = alignment.books_name[6:-4] + '_' + climatology.name
        self.grouperdowncasting = {'leadtime':'integer','latitude':'float','longitude':'float'}
        
        # Construction of climatology
        climatology.clim.name = 'climatology'
        self.clim = climatology.clim.to_dataframe().dropna(axis = 0, how = 'any')
        # Some formatting to make merging with the two-level aligned object easier
        if 'number' in self.clim.index.names: # If we are dealing with random draws. We are automatically creating a two level column index
            self.clim = self.clim.unstack('number')
        else: # Otherwise we have to create one manually
            self.clim.columns = pd.MultiIndex.from_product([self.clim.columns, ['']], names = [None,'number'])
        self.clim.reset_index(inplace = True)
        self.clim.loc[:,['latitude','longitude','climatology']] = self.clim[['latitude','longitude','climatology']].apply(pd.to_numeric, downcast = 'float')
        self.clim['doy'] = pd.to_numeric(self.clim['doy'], downcast = 'integer')
        
        try:
            self.quantile = climatology.clim.attrs['quantile']
        except KeyError:
            pass
            
            
    def fit_pp_models(self, pp_model, groupers = ['leadtime','latitude','longitude'], nfolds = 3):
        """
        Computes simple predictors like ensmean and ensstd (next to the optional raw 'pi' for an event variable). 
        Groups the dask dataframe with the groupers, and then pushes these groups to an apply function. 
        In this apply a model fitting function is called which uses the predictor columns and the observation column.
        pp_model is an object from the fitting script. It has a fit method, returning parameters and a predict method.
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
                        
            # Need inverse of the data test index to select train.
            # Stored by increasing time enables the cv to be done by location. (assuming equal group size). 
            # Times are ppossibly non-unique when also fitting to a spatial pool
            # Later we remove duplicate times (if present and we gave uniquetimes = False)
                        
            for fold in foldindex:
                test_ind = np.full((len(data),), False) # indexer to all data
                if (fold == foldindex[-1]):
                    test_ind[slice((fold - 1)*nperfold, None, None)] = True
                else:
                    test_ind[slice((fold - 1)*nperfold, (fold*nperfold), None)] = True
                
                # Calling of the fitting function on the training Should return an 1d array with the indices (same size as modelcoefs)           
                # Write into the full sized array, this converts 64-bit fitting result to float32
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
        uniquetimes = all([(group in groupers) for group in list(self.grouperdowncasting.keys())])
        self.coefcols = pp_model.model_coefs
        
        # Actual computation. Passing information to cv_fit.
        grouped = self.frame.groupby(groupers)        
        self.fits = grouped.apply(cv_fit, meta = fitreturns, **{'nfolds':nfolds, 
                                                                'fitfunc':fitfunc, 
                                                                'modelcoefs':pp_model.model_coefs,
                                                                'uniquetimes':uniquetimes}).compute()
        
        # Reset the multiindex created by the grouping to columns so that the fits can be stored in hdf table format.
        # The created index has 64 bits. we want to downcast leadtime to int8, latitude and longitude to float32
        self.fits.reset_index(inplace = True)
        for group in list(self.grouperdowncasting.keys()):
            try:
                self.fits[group] = pd.to_numeric(self.fits[group], downcast = self.grouperdowncasting[group])
            except KeyError:
                pass
        self.fits.columns = pd.MultiIndex.from_product([self.fits.columns, ['']]) # To be able to merge with self.frame
        print('models fitted for all groups')

    
    def merge_to_clim(self):
        """
        Based on day-of-year, writes an extra column to the dataframe with 'climatology', either a numeric quantile, or a climatological probability of occurrence.
        """
        self.frame['doy'] = self.frame['time'].dt.dayofyear.astype('int16')
        self.frame = self.frame.merge(self.clim, on = ['doy','latitude','longitude'], how = 'left')
        print('climatology lazily merged with frame')
    
    def compute_predictors(self, pp_model):
        """
        Currently adds ensmean and ensstd to the frame if they are not yet present as columns
        """
        if not 'ensmean' in self.frame.columns:
            self.frame['ensmean'] = self.frame['forecast'].mean(axis = 1)
        if pp_model.need_std and (not 'ensstd' in self.frame.columns):
            self.frame['ensstd']  = self.frame['forecast'].std(axis = 1)
        
    def make_pp_forecast(self, pp_model, n_members = None):
        """
        Makes a probabilistic forecast based on already fitted models. 
        Works by joining the fitted parameters (indexed by the fitting groupers and time) to the dask frame.
        If n_members is not None, then we are forecasting n random members from the fitted distribution.
        If n_members is None then there are two options: 1) The event for which we forecast is already present as an observed binary variable and we use the Logistic model for the probability of 1. 2) The event is the exceedence of the quantile beloning to the climatology in this object and we employ scipy implementation of the normal model.
        The prediction function is contained in the pp_model classes themselves.
        All corrected forecasts appear as extra columns in the self.frame
        """
        
        joincolumns = ['time'] + self.fitgroupers
        
        self.compute_predictors(pp_model = pp_model) # computation if not present yet (fit was loaded, fitted somewhere else)
        
        self.frame = self.frame.merge(self.fits, on = joincolumns, how = 'left')
        
        predfunc = getattr(pp_model, 'predict')
        
        if n_members is not None:
            """
            This is going to be ugly because of two limitations: I cannot assign multiple columns at once, so there is no way
            to get an array with all draws from the predfunc. So we have to go draw by draw, and here we cannot assign to a multi-index
            So this needs to be created afterwards.
            """
            # Creates multiple columns. Supply n_members to the predfunc, given to here in the experiment class.
            #returnmeta = pd.DataFrame(np.zeros((1,n_members), dtype = 'float32'), columns = pd.MultiIndex.from_product([['corrected'],np.arange(n_members)], names = ['','number'])) # Needs a meta dataframe.
            #returndict =  dict(itertools.product(np.arange(n_members), ['float32']))
            
            # Dask array way. #Accessing multiple members self.frame[[('forecast',1),('forecast',2)]]
            #self.frame.map_partitions(predfunc, **{'n_draws':n_members}) # Multiple columns can be generated. But not assigned.
            levelzero = self.frame.columns.get_level_values(0).tolist()
            levelone = self.frame.columns.get_level_values(1).tolist()
            for m in range(n_members):
                corcol = 'corrected' + str(m)
                self.frame[corcol] = self.frame.map_partitions(predfunc, **{'n_draws':1})
                levelzero.append('corrected')
                levelone.append(m)
            self.frame.columns = pd.MultiIndex.from_arrays([levelzero,levelone], names = self.frame.columns.names)
            
        else:
            if isinstance(pp_model, Logistic):
                # In this case we have an event variable and fitted a logistic model. And predict with that parametric model.
                self.frame['corrected'] = self.frame.map_partitions(predfunc, meta = ('corrected','float32'))
            else:
                # in this case continuous and we predict exceedence, of the climatological quantile. So we need an extra merge.
                self.merge_to_clim()
                self.frame['corrected'] = self.frame.map_partitions(predfunc, **{'quant_col':'climatology'}, meta = ('corrected','float32'))

    def add_custom_groups(self):
        """
        Add groups that are for instance chosen with clustering in observations. Based on lat-lon coding.
        """
    
    def brierscore(self):
        """
        Computes the climatological and raw scores.
        Also computes the score of the predictions from the post-processed model if this column ('corrected') is present in the dataframe
        """
        # Merging with climatological file. In case of quantile scoring for the exceeding quantiles. In case of an already e.g. pop for the climatological probability.
        if not ('climatology' in self.frame.columns):
            self.merge_to_clim()
                    
        if 'pi' in self.frame.columns: # This column comes from the match-and-write, when it is present we are dealing with an already binary observation.
            obscolname = 'observation'
            self.frame['climatology_bs'] = (self.frame['climatology'] - self.frame[obscolname])**2
        else: # In this case quantile scoring. and pi needs creation
            # Boolean comparison cannot be made in one go. because it does not now how to compare N*members agains the climatology of N
            print('lazy boolcol (pi) construction')
            self.boolcols = list()
            for member in self.frame['forecast'].columns:
                name = '_'.join(['bool',str(member)])
                self.frame[name] = self.frame[('forecast',member)] > self.frame['climatology']
                self.boolcols.append(name)
            self.frame['pi'] = (self.frame[self.boolcols].sum(axis = 1) / len(self.frame['forecast'].columns)).astype('float32') # or use .count(axis = 1) if members might be NA
            self.frame['oi'] = self.frame['observation'] > self.frame['climatology']
            obscolname = 'oi'
            self.frame['climatology_bs'] = (np.array(1 - self.quantile, dtype = 'float32') - self.frame[obscolname])**2 # Add the climatological score. use self.quantile.
            
        self.frame['forecast_bs'] = (self.frame['pi'] - self.frame[obscolname])**2
        
        if 'corrected' in self.frame.columns:
            self.frame['corrected_bs'] = (self.frame['corrected'] - self.frame[obscolname])**2
        
        # No grouping and averaging here. This is done in ScoreAnalysis objects that also allows bootstrapping.
    
    def crpsscore(self):
        """
        Discrete crps scoring by use of the properscoring package. For a small number of members (11),
        the overestimation compared to the gaussian analytical form is about 20%. 
        Therefore both climatology, the forecast, and corrected should be scored with the same number of members.
        """
        if not ('climatology' in self.frame.columns):
            self.merge_to_clim()
        
        def crps_wrap(frame, forecasttype):
            """
            Wrapper for discrete ensemble crps scoring. Finds observations and forecasts to supply to the properscoring function
            """
            return(ps.crps_ensemble(observations = frame['observation'], forecasts = frame[forecasttype])) # Not sure if this will provide the needed array type to ps
            
        for forecasttype in ['forecast','climatology','corrected']: 
            if forecasttype in self.frame.columns: # Only score the corrected when present.
                scorename = forecasttype + '_crps' 
                self.frame[scorename] = self.frame.map_partitions(crps_wrap, **{'forecasttype':forecasttype}, meta = (scorename,'float32'))
        
    
    def export(self, fits = True, frame = False):
        """
        Put both in the same hdf file, but different key. So append mode. If frame than writes one dataframe for self.frame
        float 64 columns from the brierscoring are then downcasted and duplicated model coefficients are dropped if these are present. 
        """

        self.filepath = self.basedir + self.name + '.h5'
        if fits:
            self.fits.to_hdf(self.filepath, key = 'fits', format = 'table', **{'mode':'a'})
        if frame:
            try:
                self.frame.drop(self.boolcols, axis = 1).to_hdf(self.filepath, key = 'scores', format = 'table', **{'mode':'a'})
            except AttributeError:
                self.frame.to_hdf(self.filepath, key = 'scores', format = 'table', **{'mode':'a'})
        
        return(self.name)
        

class ScoreAnalysis(object):
    """
    Contains several ways to analyse an exported file with scores. 
    Its main function is to calculate skill-scores. 
    It can export bootstrapped skillscore samples and later analyze these
    For spatial mean scores and forecast horizons.
    """
    def __init__(self, scorefile, timeagg):
        """
        Provide the name of the exported file with the scores.
        Change here the quantiles that are exported by bootstrapping procedures.
        """
        self.basedir = '/nobackup/users/straaten/scores/'
        self.scorefile = scorefile
        self.filepath = self.basedir + self.scorefile + '.h5'
        self.timeagg = timeagg
        self.quantiles = [0.025,0.5,0.975]
        
    def load(self):
        """
        Searches for relevant columns to load. Should find either _bs in certain columns or _crps
        """
        with pd.HDFStore(path=self.filepath, mode='r') as hdf:
            allcolslevelzero = pd.Index([ tup[0] for tup in  hdf.get_storer('scores').non_index_axes[0][1] ])
        if allcolslevelzero.str.contains('_bs').any():
            self.scorecols = allcolslevelzero[allcolslevelzero.str.contains('_bs')].tolist()
            self.output = '_bss'
        elif allcolslevelzero.str.contains('_crps').any():
            self.scorecols = allcolslevelzero[allcolslevelzero.str.contains('_crps')].tolist()
            self.output = '_crpss'
        self.frame = dd.read_hdf(self.filepath, key = 'scores', columns = self.scorecols + ['leadtime','latitude','longitude'])
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
        The number is in the unit of the time-aggregation of the variable. (nr days if daily values)
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
                res[i] = series.corr(series.shift(periods = i + 1, freq = freq).reindex(series.index))
                    
            if return_char_length: # Formula by Leith 1973, referenced in Feng (2011)
                return(np.nansum(((1 - np.arange(1, cutofflag + 1)/cutofflag) * res * 2)) + 1)
            else:
                return(res)
                
        # Read an extra column. Namely the observed column on which we want to compute the length.
        tempframe = dd.read_hdf(self.filepath, key = 'scores', columns = ['observation','time','latitude','longitude'])
        tempframe.columns = tempframe.columns.droplevel(1)
        # Do stuff per location
        self.charlengths = tempframe.groupby(['latitude','longitude']).apply(auto_cor, meta = ('charlength','float32'), **{'freq':self.timeagg,'cutofflag':20}).compute()
        
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
                maxcount = self.frame.groupby(['leadtime','latitude','longitude'])[self.climcol].count().max().compute()
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
            blocklength = self.charlengths.loc[(df['latitude'].iloc[0],df['longitude'].iloc[0])]
            blocklength = int(np.ceil(blocklength))
            #print(blocklength)
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
        
        grouped =  self.frame.groupby(['leadtime','latitude','longitude']) # Dask grouping
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
        
        if local:
            groupers = ['leadtime','latitude','longitude']
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
