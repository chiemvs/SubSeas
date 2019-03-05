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
                        forecast.aggregatetime(freq = freq, method = method, keep_leadtime = True)
                
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
            
            books = pd.DataFrame({'write_date':[datetime.now().strftime('%Y-%m-%d_%H:%M:%S')],
                                  'file':[filepath],
                                  'tmin':[dataset.time.min().strftime('%Y-%m-%d')],
                                  'tmax':[dataset.time.max().strftime('%Y-%m-%d')],
                                  'unit':[self.obs.array.units]})
            
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
        This climatology can be a quantile (used as the threshold for brier scoring) of it is a climatological probability if we have an aligned event predictor like POP.
        """
        self.frame = alignment.alignedobject
        self.basedir = '/nobackup/users/straaten/scores/'
        self.name = alignment.books_name[6:-4] + '_' + climatology.name
        
        climatology.clim.name = 'climatology'
        self.clim = climatology.clim.to_dataframe().dropna(axis = 0, how = 'any')
        # Some formatting to make merging with the two-level aligned object easier
        self.clim.reset_index(inplace = True)
        self.clim[['latitude','longitude','climatology']] = self.clim[['latitude','longitude','climatology']].apply(pd.to_numeric, downcast = 'float')
        self.clim.columns = pd.MultiIndex.from_product([self.clim.columns, ['']])
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
        
        def cv_fit(data, nfolds, fitfunc, modelcoefs):
            """
            Acts per group (predictors should already be constructed). Calls a fitting function 
            in a time-cross-validation fashion. Supplies data as (nfolds - 1)/nfolds for training, 
            and writes the returned coeficients by the model to the remaining 1/nfolds part 
            (where the prediction will be made) as a dataframe. The returned Dataframe is indexed with time
            Which combines with the groupers to a multi-index.
            """
            
            nperfold = len(data) // nfolds
            foldindex = range(1,nfolds+1)
            
            # Pre-allocating a structure for the dataframe.
            coefs = pd.DataFrame(data = np.nan, index = data['time'], columns = modelcoefs)
                        
            # Need inverse of the data test index to select train. For loc is done through np.setdiff1d
            # Stored by increasing time enables the cv to be done by integer location. (assuming equal group size). 
            # Times are ppossibly non-unique when also fitting to a spatial pool
            # Later we remove duplicate times (if present)
            full_ind = np.arange(len(coefs))
            
            for fold in foldindex:
                if (fold == foldindex[-1]):
                    test_ind = np.arange((fold - 1)*nperfold, len(coefs))
                else:
                    test_ind = np.arange(((fold - 1)*nperfold),(fold*nperfold))

                train_ind = np.setdiff1d(full_ind, test_ind)
                
                # Calling of the fitting function on the training Should return an 1d array with the indices (same size as modelcoefs)
                fitted_coef = fitfunc(data.iloc[train_ind,:])
                
                # Write to the frame with time index. Should the fold be prepended as extra level in a multi-index?
                coefs.iloc[test_ind,:] = fitted_coef
            
            # Returning only the unique time indices but this eases the wrapping and joining of the grouped results later on.
            return(coefs[~coefs.index.duplicated(keep='first')])
        
        # Computation of predictands for the models.
        self.frame['ensmean'] = self.frame['forecast'].mean(axis = 1)
        if pp_model.need_std:
            self.frame['ensstd']  = self.frame['forecast'].std(axis = 1)
        
        fitfunc = getattr(pp_model, 'fit')
        fitreturns = dict(itertools.product(pp_model.model_coefs, ['float32']))
        
        grouped = self.frame.groupby(groupers)
        
        self.fits = grouped.apply(cv_fit, meta = fitreturns, **{'nfolds':nfolds, 'fitfunc':fitfunc, 'modelcoefs':pp_model.model_coefs}).compute()
        self.fits.reset_index(inplace = True)
        self.fits.columns = pd.MultiIndex.from_product([self.fits.columns, ['']])
        # Store some information
        self.fitgroupers = groupers
        self.coefcols = pp_model.model_coefs
    
    def merge_to_clim(self):
        """
        Based on day-of-year, writes an extra column to the dataframe with 'climatology', either a numeric quantile, or a climatological probability of occurrence.
        """
        self.frame['doy'] = self.frame['time'].dt.dayofyear
        self.frame = self.frame.merge(self.clim, on = ['doy','latitude','longitude'], how = 'left')
        
        
    def make_pp_forecast(self, pp_model):
        """
        Makes a probabilistic forecast based on already fitted models. 
        Works by joining the fitted parameters (indexed by the fitting groupers and time) to the dask frame.
        The event for which we forecast is already present as an observed event variable. Or it is the exceedence of the quantile beloning to the climatology.
        The forecast is made with the parametric model for the logistic regression, or a scipy implementation of the normal model, both contained in the fitting classes
        """
        
        joincolumns = ['time'] + self.fitgroupers
        
        self.frame = self.frame.merge(self.fits, on = joincolumns, how = 'left')
        
        predfunc = getattr(pp_model, 'predict')
        
        if isinstance(pp_model, Logistic):
            # In this case we have an event variable and fitted a logistic model. And predict with that parametric model.
            self.frame['pi_cor'] = self.frame.map_partitions(predfunc, meta = ('pi_cor','float32'))
        else:
            # in this case continuous and we predict exceedence, of the climatological quantile. So we need an extra merge.
            self.merge_to_clim()
            self.frame['pi_cor'] = self.frame.map_partitions(predfunc, **{'quant_col':'climatology'}, meta = ('pi_cor','float32'))

    def add_custom_groups(self):
        """
        Add groups that are for instance chosen with clustering in observations. Based on lat-lon coding.
        """
    
    def brierscore(self):
        """
        Asks a list of groupers, could use custom groupers. Computes the climatological and raw scores.
        Also computes the score of the predictions from the post-processed model if this column (pi_cor) is present in the dataframe
        """
        # Merging with climatological file. In case of quantile scoring for the exceeding quantiles. In case of an already e.g. pop for the climatological probability.
        if not ('climatology' in self.frame.columns):
            self.merge_to_clim()
        
        if 'pi' in self.frame.columns: # This column comes from the match-and-write, when it is present we are dealing with an already binary observation.
            obscolname = 'observation'
            self.frame['climbrier'] = (self.frame['climatology'] - self.frame[obscolname])**2
        else: # In this case quantile scoring. and pi needs creation
            # Boolean comparison cannot be made in one go. because it does not now how to compare N*members agains the climatology of N
            self.boolcols = list()
            for member in self.frame['forecast'].columns:
                name = '_'.join(['bool',str(member)])
                self.frame[name] = self.frame[('forecast',member)] > self.frame['climatology']
                self.boolcols.append(name)
            self.frame['pi'] = self.frame[self.boolcols].sum(axis = 1) / len(self.frame['forecast'].columns) # or use .count(axis = 1) if members might be NA
            self.frame['oi'] = self.frame['observation'] > self.frame['climatology']
            obscolname = 'oi'
            self.frame['climbrier'] = ((1 - self.quantile) - self.frame[obscolname])**2 # Add the climatological score. use self.quantile.
            
        self.frame['rawbrier'] = (self.frame['pi'] - self.frame[obscolname])**2
        
        if 'pi_cor' in self.frame.columns:
            self.frame['corbrier'] = (self.frame['pi_cor'] - self.frame[obscolname])**2
        
        # No grouping and averaging yet.
        #grouped = delayed.groupby(groupers)
        #return(grouped.mean()[['rawbrier','climbrier']].compute())
    
    def export(self):
        """
        Writes one dataframe for self.frame, discarding only the duplicated model coefficients if these are present. Fits are written to a separate entry. 
        """
        self.filepath = self.basedir + self.name + '.h5'
        if hasattr(self, 'fits'):
            try:
                self.frame.drop(self.coefcols + self.boolcols, axis = 1).to_hdf(self.filepath, key = 'scores', format = 'table')
            except AttributeError:
                self.frame.drop(self.coefcols, axis = 1).to_hdf(self.filepath, key = 'scores', format = 'table')
            self.fits.to_hdf(self.filepath, key = 'fits', format = 'table')
        else:
            try:
                self.frame.drop(self.boolcols, axis = 1).to_hdf(self.filepath, key = 'scores', format = 'table')
            except AttributeError:
                self.frame.to_hdf(self.filepath, key = 'scores', format = 'table')
        
        return(self.name)

class ScoreAnalysis(object):
    """
    Contains several ways to analyse an exported file with scores. 
    """
    def __init__(self, scorefile):
        """
        Provide the name of the exported file with Brier scores.
        """
        self.basedir = '/nobackup/users/straaten/scores/'
        self.filepath = self.basedir + scorefile + '.h5'
        self.frame = dd.read_hdf(self.filepath, key = 'scores')
    
    def bootstrap_skillscore(self, groupers = ['leadtime']):
        """
        Samples the score entries. First grouping and then random number generation. 100% of the sample size with replacement. Average and compute skill score. Return a 1000 skill scores.
        """
        scorecols = [col for col in ['rawbrier', 'climbrier','corbrier'] if (col in self.frame.columns)]
        grouped =  self.frame.groupby(groupers)
        return(grouped.mean()[scorecols].compute())
        # DataFrame.sample(frac = 1, replace = True)
            
#ddtx = ForecastToObsAlignment(season = 'JJA', cycle = '41r1')
#ddtx.recollect(booksname='books_tx_JJA_41r1_3D_max_1.5_degrees_max.csv') #dd.read_hdf('/nobackup/users/straaten/match/tx_JJA_41r1_3D_max_1.5_degrees_max_169c5dbd7e3a4881928a9f04ca68c400.h5', key = 'intermediate')
#climatology = Climatology('tx', **{'name':'tx_clim_1980-05-30_2010-08-31_3D-max_1.5-degrees-max_5_5_q0.9'})
#climatology.localclim()
#self = Comparison(alignment=ddtx, climatology = climatology)
#self.fit_pp_models(pp_model= NGR(), groupers = ['leadtime'])
#self.make_pp_forecast(pp_model = NGR())
#self.brierscore()

#ddpop = ForecastToObsAlignment(season = 'DJF', cycle = '41r1') #dd.read_hdf('/nobackup/users/straaten/match/pop_DJF_41r1_1D_0.25_degrees_8b505d0f2d024bf086054fdf7629e8ed.h5', key = 'intermediate')
#ddpop.outfiles = ['/nobackup/users/straaten/match/pop_DJF_41r1_1D_0.25_degrees_8b505d0f2d024bf086054fdf7629e8ed.h5']
#ddpop.recollect()
#ddpop.books_name = 'books_pop_DJF_41r1_1D_0.25_degrees.csv'
#climatology = Climatology('pop', **{'name':'pop_clim_1980-01-01_2017-12-31_1D_0.25-degrees_5_5_mean'})
#climatology.localclim()
#self = Comparison(alignment=ddpop, climatology = climatology)
#self.fit_pp_models(pp_model = Logistic(), groupers = ['leadtime'])
#self.make_pp_forecast(pp_model = Logistic())
#self.brierscore()
#self.export()

#temp = ScoreAnalysis(scorefile = 'tx_JJA_41r1_3D_max_1.5_degrees_max_tx_clim_1980-05-30_2010-08-31_3D-max_1.5-degrees-max_5_5_q0.9')
#sc = temp.bootstrap_skillscore()

#temp2 = ScoreAnalysis(scorefile= 'pop_DJF_41r1_1D_0.25_degrees_pop_clim_1980-01-01_2017-12-31_1D_0.25-degrees_5_5_mean')
#sc2 = temp2.bootstrap_skillscore()
