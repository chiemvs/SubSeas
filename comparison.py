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
from helper_functions import monthtoseasonlookup, unitconversionfactors
from fitting import NGR, Logistic

class ForecastToObsAlignment(object):
    """
    Idea is: you have already prepared an observation class, with a certain variable, temporal extend and space/time aggregation.
    This searches the corresponding forecasts, and possibly forces the same aggregations.
    TODO: not sure how to handle groups yet.
    """
    def __init__(self, season, cycle, observations = None):
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
            self.time_agg = int(self.dates.diff().dt.days.mode())
            self.maxleadtime = 46
    
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
        
    def load_forecasts(self, n_members):
        """
        Gets the daily processed forecasts into memory. Delimited by the left timestamp and the aggregation time.
        This is done by using the load method of each Forecast class in the dictionary. They are stored in a list.
        """
        for date, listofforecasts in self.forecasts.items():
            
            if listofforecasts: # empty lists are skipped
                tmin = date
                tmax = date + pd.Timedelta(str(self.time_agg - 1) + 'D') # -1 because date itself also counts for one day in the aggregation.
                
                for forecast in listofforecasts:
                    forecast.load(variable = self.obs.basevar, tmin = tmin, tmax = tmax, n_members = n_members)
        
        self.n_members = n_members
     
    def force_resolution(self, time = True, space = True):
        """
        Force the observed resolution onto the Forecast arrays. Checks if the same resolution and force spatial/temporal aggregation if that is not the case.
        Makes use of the methods of each Forecast class
        """
            
        for date, listofforecasts in self.forecasts.items():
        
            for forecast in listofforecasts:
                
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
                        freq, method = obstimemethod.split('-')
                        forecast.aggregatetime(freq = freq, method = method, keep_leadtime = True, ndayagg = self.time_agg)
                
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
                        step, what, method = obsspacemethod.split('-')
                        forecast.aggregatespace(step = float(step), method = method, by_degree = (what == 'degrees'))
                      
    def force_new_variable(self, array):
        """
        Call upon event classification to get the on-the-grid conversion of the base variable.
        Do this by a small initization of an object with an .array attribute.
        Then call the classification method with a similar name as the newvar and execute it to let it return the newvar array
        """
        newvariable = self.obs.newvar
        temp = SurfaceObservations(alias = newvariable, **{'array':array}) # Just a convenient class, does not neccessarily mean the array is an observed one.
        method = getattr(EventClassification(obs = temp), newvariable)
        return(method(inplace = False))
        
                
    def match_and_write(self, newvariable = False):
        """
        Neirest neighbouring to match pairs. Be careful when domain of observations is larger.
        Also converts forecast units to observed units.
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
                characteristics = [self.obs.newvar, self.season, self.cycle, self.obs.timemethod, self.obs.spacemethod]
            else:
                characteristics = [self.obs.basevar, self.season, self.cycle, self.obs.timemethod, self.obs.spacemethod]
            filepath = self.basedir + '_'.join(characteristics) + '_' + uuid.uuid4().hex + '.h5'
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
                fieldobs = self.obs.array.sel(time = date).drop('time')
                
                allleadtimes = xr.concat(objs = [f.array.swap_dims({'time':'leadtime'}) for f in listofforecasts], dim = 'leadtime') # concatenates over leadtime dimension.
                a,b = unitconversionfactors(xunit = allleadtimes.units, yunit = fieldobs.units)
                exp = allleadtimes.reindex_like(fieldobs, method='nearest') * a + b
                
                # Call force_new_variable here. Which in its turn calls the same EventClassification method that was also used on the obs.
                if newvariable:
                    binary = self.force_new_variable(exp) # the binary members are discarded at some point.
                    pi = binary.mean(dim = 'number') # only the probability of the event over the members is retained.
                    # Merging, exporting to pandas and masking by dropping on NA observations.
                    combined = xr.Dataset({'forecast':exp.drop('time'),'observation':fieldobs, 'pi':pi.drop('time')}).to_dataframe().dropna(axis = 0)
                    # Unstack creates duplicates. Two extra columns (obs and pi) need to be selected. Therefore the iloc
                    temp = combined.unstack('number').iloc[:,np.append(np.arange(0,self.n_members + 1), self.n_members * 2)]
                    temp.reset_index(inplace = True) # places latitude and all in 
                    labels = [l for l in temp.columns.labels[1]]
                    labels[-2:] = np.repeat(self.n_members, 2)
                else:
                    # Merging, exporting to pandas and masking by dropping on NA observations.
                    combined = xr.Dataset({'forecast':exp.drop('time'), 'observation':fieldobs}).to_dataframe().dropna(axis = 0)
                    # Handle the multi-index
                    # first puts number dimension into columns. observerations are duplicated so therefore selects up to n_members +1
                    temp = combined.unstack('number').iloc[:,:(self.n_members + 1)]
                    temp.reset_index(inplace = True) # places latitude and all in 
                    labels = [l for l in temp.columns.labels[1]]
                    labels[-1] = self.n_members
                
                temp.columns.set_labels(labels, level = 1, inplace = True)
                
                # Downcasting latitude and longitude
                temp[['latitude', 'longitude']] = temp[['latitude', 'longitude']].apply(pd.to_numeric, downcast = 'float')
                # Downcasting the leadtime column
                temp['leadtime'] = np.array(temp['leadtime'].values.astype('timedelta64[D]'), dtype = 'int8')
                
                # prepend with the time index.
                temp.insert(0, 'time', date)
                aligned_basket.append(temp) # temp.swaplevel(1,2, axis = 0)
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
        This climatology can be a quantile (used as the threshold for brier scoring) of it is a climatological probability if we have an aligned event predictor like POP. Or it is a random draw used for CRPS scoring. which means that multiple 'numbers' will be present.
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
    """
    def __init__(self, scorefile, timeagg):
        """
        Provide the name of the exported file with the scores.
        Searches for relevant columns to load. Should find either _bs in certain columns or _crps
        """
        self.basedir = '/nobackup/users/straaten/scores/'
        self.filepath = self.basedir + scorefile + '.h5'
        with pd.HDFStore(path=self.filepath, mode='r') as hdf:
            allcolslevelzero = pd.Index([ tup[0] for tup in  hdf.get_storer('scores').non_index_axes[0][1] ])
        if allcolslevelzero.str.contains('_bs').any():
            self.scorecols = allcolslevelzero[allcolslevelzero.str.contains('_bs')].tolist()
            self.output = '_bss'
        elif allcolslevelzero.str.contains('_crps').any():
            self.scorecols = allcolslevelzero[allcolslevelzero.str.contains('_crps')].tolist()
            self.output = '_crpss'
        self.frame = dd.read_hdf(self.filepath, key = 'scores', columns = self.scorecols + ['leadtime','latitude','longitude'])
        # self.frame = dd.read_hdf(self.filepath, key = 'scores', columns = self.scorecols + ['leadtime','latitude','longitude', 'observation','time'])
        self.frame.columns = self.frame.columns.droplevel(1)
        self.timeagg = timeagg
    
    def eval_skillscore(self, data, scorecols, returncols, climcol):
        """
        Is supplied an isolated data group on which to evaluate skill of mean brierscores.
        These relevant columns are given as the scorecols.
        Returns a series with same length as scorecols. Climatology obviously has skill 1
        """
        meanscore = data[scorecols].mean(axis = 0)
        returns = np.zeros((len(returncols),), dtype = 'float32')

        for scorecol, returncol in zip(scorecols, returncols):
            returns[returncols.index(returncol)] = np.array(1, dtype = 'float32') - meanscore[scorecol]/meanscore[climcol] 

        return(pd.Series(returns, index = returncols, name = self.output))
        
    def mean_skill_score(self, groupers = ['leadtime']):
        """
        Grouping. Average and compute skill score.
        """
        returncols = [ col.split('_')[0] + self.output for col in self.scorecols]
        climcol = [col for col in self.scorecols if col.startswith('climatology')][0]
        grouped =  self.frame.groupby(groupers)
        
        scores = grouped.apply(self.eval_skillscore, 
                               meta = pd.DataFrame(dtype='float32', columns = returncols, index=[self.output]), 
                               **{'scorecols':self.scorecols, 'returncols': returncols, 'climcol':climcol}).compute()
        return(scores)
    
    def bootstrap_skill_score(self, groupers = ['leadtime']):
        """
        Samples the score entries. First grouping and then per (local) group multiple 
        random samples are taken and their quantiles are outputted per group.
        Quantiles for the confidence intervals (also for plots) are defined in this function.
        NOTE: decorrelation time and block-bootstrapping cannot be incorporated
        here as it is only easy for local in time.
        """
        returncols = [ col.split('_')[0] + self.output for col in self.scorecols]
        quantiles = [0.025,0.5,0.975]
        climcol = [col for col in self.scorecols if col.startswith('climatology')][0]
        grouped =  self.frame.groupby(groupers)
        
        def bootstrap_quantiles(df, returncols, climcol, n_samples, quantiles):
            """
            Acts per group. Creates n samples (with replacement) of the dataframe.
            For each of these it calls the evaluate skill-scores method, whose output is collected.
            From the sized n collection it distills and returns the desired quantiles.
            
            """
            collect = pd.DataFrame(dtype = 'float32', columns = returncols, index = pd.RangeIndex(stop = n_samples))
            
            for i in range(n_samples):
                collect.iloc[i,:] = self.eval_skillscore(df[self.scorecols].sample(frac = 1, replace = True), scorecols = self.scorecols, returncols = returncols, climcol = climcol)
                
            return(collect.quantile(q = quantiles, axis = 0).astype('float32'))
        
        bounds = grouped.apply(bootstrap_quantiles,
                               meta = pd.DataFrame(dtype='float32', columns = returncols, index = pd.Index(quantiles, name = 'quantile')),
                               **{'returncols':returncols, 'climcol':climcol, 'n_samples':200, 'quantiles':quantiles}).compute()
        return(bounds)
    
    def characteristiclength(self):
        """
        Computes a characteristic length per location to use as bootstrapping block length.
        According the formula of Leith 1973 which is reference in Feng (2011)
        T_0 = 1 + 2 * \sum_{\tau= 1}^D (1 - \tau/D) * \rho_\tau
        The number is in the unit of the time-aggregation of the variable. (nr days if daily values)
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
            
    def block_bootstrap_mean_of_local_skills(self, n_samples = 200):
        """
        At each random instance picks a local time-block-bootstraped sample and computes local skill for each location.
        These are reworked to the areal mean. This process is repeated n_samples times 
        and quantiles of the generated skillsare returned. The procedure loads the whole dataset
        per leadtime (dask based), and does local (pandas based) grouping to get local block lengths and a local random set.
        Note: should the use of the charlengths have to be adapted for the small datasets? Cannot be 7 weeks or something.
        """

        if not hasattr(self, 'charlengths'):
            self.characteristiclength()
        
        def block_sample(dflocation, fraction, kwargs):
            """
            Assumes that all rows in the supplied data are unique timesteps.
            Kwargs are for the eval_skillscore function
            It splits the sets of rows in a non-overlapping sense in blocks with the local blocklength.
            These splits currently might jump into the new season. Otherwise it will be more expensive to track consequtive months
            with local records of uneven length.
            The local blocklength is read from the self.charlengths attribute.
            """
            blocklength = self.charlengths.loc[(dflocation['latitude'].iloc[0],dflocation['longitude'].iloc[0])]
            blocklength = int(np.ceil(blocklength))
            #print(blocklength)
            setlength = len(dflocation)
            
            if blocklength == 1:
                return(self.eval_skillscore(dflocation[self.scorecols].sample(frac = fraction, replace = True), **kwargs))
            else:
                rowsets = pd.Series(np.array_split(np.arange(setlength), np.arange(blocklength,setlength,blocklength))) # produces a list of arrays
                choice = rowsets.sample(frac = fraction, replace = True)
                choice = np.concatenate(choice.values) # Glue the chosen sets back to a list
                return(self.eval_skillscore(dflocation.iloc[choice,:], **kwargs)) # Duplicate indices are allowed in iloc.
            
        def per_leadtime(df, n_samples, quantiles, kwargs):
            """
            Function so we can apply the procedure per leadtime in the dask-framework.
            Kwargs are for the eval_skillscore function. The overall score is computed n_samples times. Only quantiles are returned.
            """
            localgroups = df.groupby(['latitude','longitude']) # Pandas grouping.
            collect = pd.DataFrame(dtype = 'float32', columns = returncols, index = pd.RangeIndex(stop = n_samples))
            
            for i in range(n_samples):
                localscores = localgroups.apply(block_sample, **{'fraction':1, 'kwargs':kwargs})
                spacemean = localscores.mean(axis = 0, skipna = True) # mean of local scores.
                collect.iloc[i,:] = spacemean
                
            return(collect.quantile(q = quantiles, axis = 0).astype('float32'))
        
        returncols = [ col.split('_')[0] + self.output for col in self.scorecols]
        quantiles = [0.025,0.5,0.975]
        climcol = [col for col in self.scorecols if col.startswith('climatology')][0]
        kwargs = {'scorecols':self.scorecols, 'returncols':returncols,'climcol':climcol}
        grouped =  self.frame.groupby(['leadtime']) # Dask grouping
        bounds = grouped.apply(per_leadtime,
                               meta = pd.DataFrame(dtype='float32', columns = returncols, index = pd.Index(quantiles, name = 'quantile')),
                               **{'n_samples':n_samples, 'quantiles':quantiles, 'kwargs':kwargs}).compute()
        
        return(bounds)

#from dask.distributed import Client
#client = Client(silence_logs=False)        
#self = ScoreAnalysis('tg_DJF_41r1_1D_0.75-degrees-mean_tg_clim_1980-05-30_2015-02-28_1D_0.75-degrees-mean_5_5_rand',timeagg = '1D')
#self.filepath = self.basedir + 'tg_DJF_41r1_1D_2-degrees-mean_tg_clim_1980-05-30_2015-02-28_1D_2-degrees-mean_5_5_q0.66.h5'
#self.characteristiclength()
#test = self.block_bootstrap_mean_of_local_skills(n_samples = 5)
    
#basedir = '/nobackup/users/straaten/E-OBS/' # '/home/jsn295/Documents/climtestdir/'
#test1 = SurfaceObservations('tx', **{'basedir':basedir})
#test1.load(tmax = '1970-01-01', llcrnr = (25,-30), rucrnr = (75,75)) # llcrnr = (36.0, None))
#test2 = SurfaceObservations('tx', **{'basedir':basedir})
#test2.load(tmax = '1970-01-01', llcrnr = (25,-30), rucrnr = (75,75)) # llcrnr = (36.0, None))
#test1.aggregatetime(freq = '4D', method = 'max')
#test1.aggregatespace(step = 3, method = 'max', by_degree = True)
#test2.aggregatespace(step = 3, method = 'max', by_degree = True)
#clim1 = Climatology(test1.basevar) # '/home/jsn295/Documents/climtestdir/'
#clim1.localclim(obs = test1, daysbefore = 5, daysafter = 5, mean = False, n_draws = 11, daily_obs_array = test2.array)
#clim1.localclim(obs = test1, daysbefore = 5, daysafter = 5, mean = True, daily_obs_array = test2.array)

#clim1 = Climatology('tx', **{'name':'tx_clim_1980-05-30_2010-08-31_4D-max_3-degrees-max_5_5_q0.5'})
#clim1.localclim()
#
#matchobject = '/usr/people/straaten/Downloads/tx_JJA_41r1_4D_max_3_degrees_max.h5' # '/home/jsn295/Documents/climtestdir/tx_JJA_41r1_4D_max_3_degrees_max.h5'
#ddtx = ForecastToObsAlignment(season = 'JJA', cycle = '41r1')
#ddtx.alignedobject = dd.read_hdf(matchobject, key = 'intermediate')
#ddtx.books_name = 'blahblahblahblah'
#
#self = Comparison(ddtx, climatology = clim1)
#self.merge_to_clim()

#self = ScoreAnalysis('ahblah_tx_clim_1980-05-30_2010-08-31_4D-max_3-degrees-max_5_5_q0.5')
        
#ddtg = ForecastToObsAlignment(season = 'DJF', cycle = '41r1')
#ddtg.recollect(booksname= 'books_tg_DJF_41r1_7D-mean_3-degrees-mean.csv')
#climatology = Climatology('tg', **{'name':'tg_clim_1980-05-30_2015-02-28_7D-mean_3-degrees-mean_5_5_q0.1'})
#climatology.localclim()
#self = Comparison(alignment=ddtg, climatology = climatology)

#climatology = Climatology('tg', **{'name':'tg_clim_1980-05-30_2015-02-28_2D-mean_3-degrees-mean_5_5_q0.1'})
#climatology.localclim()
#ddtx = ForecastToObsAlignment(season = 'DJF', cycle = '41r1')
##ddtx.alignedobject = dd.read_hdf('/nobackup/users/straaten/match/tx_JJA_41r1_3D_max_1.5_degrees_max_169c5dbd7e3a4881928a9f04ca68c400.h5', key = 'intermediate')
#ddtx.recollect(booksname='books_tg_DJF_41r1_2D-mean_3-degrees-mean.csv')
#comp = Comparison(ddtx, climatology = climatology)
#comp.compute_predictors(pp_model = NGR())
#comp.fit_pp_models(pp_model= NGR())
#comp.make_pp_forecast(pp_model = NGR())
#comp.brierscore()
#case1 = comp.frame.compute()
#
#comp2 = Comparison(ddtx, climatology = climatology)
#comp2.compute_predictors(pp_model = NGR2())
#comp2.fit_pp_models(pp_model = NGR2())
#comp2.make_pp_forecast(pp_model = NGR2())
#comp2.brierscore()
#case2 = comp2.frame.compute()
#
##domain wide
#globalbs1 = case1.groupby(['leadtime']).quantile(0.95)
#globalbs1['corbss'] = 1 - globalbs1['corbrier'] / globalbs1['climbrier']
#globalbs2 = case2.groupby(['leadtime']).quantile(0.95)
#globalbs2['cor_transformbss'] = 1 - globalbs2['corbrier'] / globalbs2['climbrier']
#pd.DataFrame([globalbs1['corbss'], globalbs2['cor_transformbss']]).T.plot()
#
#(globalbs2['corbrier'] - globalbs1['corbrier']).plot()
#
##local
#globalbs1 = case1.groupby(['leadtime', 'latitude','longitude']).mean()['corbrier']
#globalbs2 = case2.groupby(['leadtime', 'latitude','longitude']).mean()['corbrier']
#globalbs2.name = 'corbrier_logtransform'
#
#bsframe = pd.DataFrame([globalbs1, globalbs2]).T
#bsfields = bsframe.to_xarray()
#
## Check the fits at location: lon 16.5, lat 50.75
#fits1 = comp.fits.loc[np.logical_and(comp.fits['latitude'] == 50.75, comp.fits['longitude'] == 16.5),comp.coefcols + ['leadtime']].drop_duplicates()
#fits2 = comp2.fits.loc[np.logical_and(comp.fits['latitude'] == 50.75, comp.fits['longitude'] == 16.5),comp.coefcols + ['leadtime']].drop_duplicates()
