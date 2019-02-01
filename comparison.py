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
from observations import SurfaceObservations
from forecasts import Forecast
from helper_functions import monthtoseasonlookup, unitconversionfactors

class ForecastToObsAlignment(object):
    """
    Idea is: you have already prepared an observation class, with a certain variable, temporal extend and space/time aggregation.
    This searches the corresponding forecasts, and possibly forces the same aggregations.
    TODO: not sure how to handle groups yet.
    """
    def __init__(self, season, observations):
        """
        Temporal extend can be read from the attributes of the observation class. 
        Specify the season under inspection for subsetting.
        """
        self.basedir = '/nobackup/users/straaten/match/'
        self.season = season
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
            forecasts = [Forecast(indate, prefix = 'for_') for indate in contain]
            hindcasts = [Forecast(indate, prefix = 'hin_') for indate in contain]
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
    
    def force_resolution(self):
        """
        Check if the same resolution and force spatial/temporal aggregation if that is not the case.
        Makes use of the methods of each Forecast class
        """
            
        for date, listofforecasts in self.forecasts.items():
        
            for forecast in listofforecasts:
                
                # Check time aggregation
                try:
                    obstimemethod = getattr(self.obs, 'timemethod')
                    try:
                        fortimemethod = getattr(forecast, 'timemethod')
                        if not fortimemethod == obstimemethod:
                            raise AttributeError
                    except AttributeError:
                        print('Aligning time aggregation')
                        freq, method = obstimemethod.split('_')
                        forecast.aggregatetime(freq = freq, method = method, keep_leadtime = True)
                except AttributeError:
                    print('no time aggregation in obs')
                    pass # obstimemethod has no aggregation

                # Check space aggregation
                try:
                    obsspacemethod = getattr(self.obs, 'spacemethod')
                    try:
                        forspacemethod = getattr(forecast, 'spacemethod')
                        if not forspacemethod == obsspacemethod:
                            raise AttributeError
                    except AttributeError:
                        print('Aligning space aggregation')
                        step, what, method = obsspacemethod.split('_')
                        forecast.aggregatespace(step = int(step), method = method, by_degree = (what is 'degrees'))
                except AttributeError:
                    print('no space aggregation in obs')
                    pass # obsspacemethod has no aggregation                           
        
    def match_and_write(self):
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
        def write_outfile(basket):
            """
            Saves to a unique filename and creates a bookkeeping file (appends if it already exists)
            """
            characteristics = [self.obs.basevar, self.season]
            for m in ['timemethod','spacemethod']:
                if hasattr(self.obs, m):
                    characteristics.append(getattr(self.obs, m))
            filepath = self.basedir + '_'.join(characteristics) + '_' + uuid.uuid4().hex + '.h5'
            books_path = self.basedir + 'books_' + '_'.join(characteristics) + '.csv'
            
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
                    write_outfile(aligned_basket)
                    aligned_basket = []
        
        # After last loop also write out 
        if aligned_basket:
            write_outfile(aligned_basket)
        
    def recollect(self, booksname = None):
        """
        Makes a dask dataframe object. The outfilenames can be collected by reading the timefiles. Or for instance by searching with a regular expression.
        """
        if hasattr(self, 'outfiles'):
            outfilenames = self.outfiles
        else:
            try:
                books = pd.read_csv(self.basedir + booksname)
                outfilenames = books['file'].values.tolist()
            except AttributeError:
                pass
            
        self.alignedobject = dd.read_hdf(outfilenames, key = 'intermediate')
        
#obs = SurfaceObservations(alias = 'tg')
#obs.load(tmin = '1995-05-14', tmax = '1998-09-01', llcrnr = (25,-30), rucrnr = (75,75)) # "75/-30/25/75"
#obs.aggregatetime(freq = 'w', method = 'mean')

#test = ForecastToObsAlignment(season = 'JJA', observations=obs)

#test.find_forecasts()
#test.load_forecasts(n_members=11)
#test.force_resolution()
#test.match_and_write()

# Make a counter plot:
#obs = SurfaceObservations(alias = 'rr')
#obs.load(tmin = '1995-05-14')
#test = ForecastToObsAlignment(season = 'JJA', observations=obs)
#test.find_forecasts()
#counter = np.array([len(listofforecasts) for date, listofforecasts in test.forecasts.items()])
#counter = pd.Series(data = counter, index = test.dates.index)
#counter['19950514':'19960514'].plot()
#plt.close()

        
class Comparison(object):
    """
    All based on the dataframe format. No arrays anymore.
    Potentially I can set an index like time in any operation where time becomes unique to speed up lookups.
    """
    
    def __init__(self, alignedobject, climatology = None):
        """
        The aligned object has Observation | members |
        Potentially an external observed climatology array can be supplied that takes advantage of the full observed dataset. It has to have a location and dayofyear timestamp. Is it already aggregated?
        NOTE: make the climatological object also contain the climatological forecast for that quantile. (this is the quantile)
        """
        self.frame = alignedobject
        if climatology is not None: 
            climatology.name = 'climatology'
            self.clim = climatology.to_dataframe().dropna(axis = 0, how = 'any')
            # Some formatting to make merging with the two-level aligned object easier
            self.clim.reset_index(inplace = True)
            self.clim[['latitude','longitude','climatology']] = self.clim[['latitude','longitude','climatology']].apply(pd.to_numeric, downcast = 'float')
            self.clim.columns = pd.MultiIndex.from_product([self.clim.columns, ['']])
            try:
                self.quantile = climatology.attrs['quantile']
            except KeyError:
                pass
            
        
    def setup_cv(self, nfolds = 3):
        """
        The amount of available times can differ per leadtime.
        And due to NA's the field size is not always equal. Varies between 31544 and 31549.
        This computes the amount of dates to be selected per leadtime, location and fold. Last fold should get the remainder.
        """
        self.nfolds = nfolds
        datesperloclead = self.frame.groupby(['leadtime','latitude','longitude'])['time'].count()
        self.n_per_fold = datesperloclead.floordiv(nfolds).compute()
        # accessible with multi-index as self.n_per_fold.loc[(leadtime,latitude,longitude)]
        
        #self.cvbins = pd.cut(self.frame['time'].compute(), bins = nfolds, labels = np.arange(1,nfolds+1))
        # Pd.cut cannot be done delayed and unfortunately the whole time column is too large to lead explicitly
            
    def fit_pp_models(self, groupers = ['leadtime','latitude','longitude'], nfolds = 1):
        """
        Computes simple predictors like. Groups the dask dataframe per location and leadtime, and then pushes these groups to an apply function. In this apply a model fitting function can be called which uses the predictor columns and the observation column, the function currently returns a Dataframe with parametric coefficients. Which are stored in-memory with the groupers as multi-index.
        """
        from sklearn.linear_model import LinearRegression
        
        def fitngr(data):
            reg = LinearRegression().fit(data['ensmean'].values.reshape(-1,1), data['observation'].values.reshape(-1,1))
            return(pd.DataFrame({'intercept':reg.intercept_[0], 'slope':reg.coef_[0]}))
        
        def cv_fitlinear(data, nfolds):
            """
            Fits a linear model. If nfolds = 1 it uses all the data, otherwise it is split up. This cross validation is done here because addition of a column to the full ungrouped frame is too expensive, also the amount of available times can differ per leadtime and due to NA's the field size is not always equal. Therefore, the available amount is computed here.
            TODO: needs to be expanded to NGR and Logistic regression for the PoP.
            """
            
            modelcoefs = ['intercept','slope']
            nperfold = len(data) // nfolds
            foldindex = range(1,nfolds+1)
            
            # Pre-allocating a structure.
            coefs = np.full((nfolds,len(modelcoefs)), np.nan)
            
            for fold in foldindex:
                if (fold == foldindex[-1]):
                    train = data.iloc[((fold - 1)*nperfold):,:]
                else:
                    train = data.iloc[((fold - 1)*nperfold):(fold*nperfold),:]
                
                # Fitting of the linear model
                reg = LinearRegression().fit(train['ensmean'].values.reshape(-1,1), train['observation'].values.reshape(-1,1))
                coefs[fold-1,:] = [reg.intercept_[0], reg.coef_[0]]
            
            models = pd.DataFrame(data = coefs, index = pd.Index(foldindex, name = 'fold'), columns = modelcoefs)            
            
            return(models)
        
        # Computation of predictands for the linear models.
        # There are some negative precipitation values in .iloc[27:28,:]
        self.frame['ensmean'] = self.frame['forecast'].mean(axis = 1)
        self.frame['ensstd']  = self.frame['forecast'].std(axis = 1)
                
        grouped = self.frame.groupby(groupers)
        #self.fits = grouped.apply(fitngr, meta = {'intercept':'float32', 'slope':'float32'}).compute() #
        
        self.fits = grouped.apply(cv_fitlinear, meta = {'intercept':'float32', 'slope':'float32'}, **{'nfolds':nfolds}).compute()
        
        
    def make_pp_forecast(self):
        """
        Adds either forecast members or a probability distribution
        """
    

    def add_custom_groups(self):
        """
        Add groups that are for instance chosen with clustering in observations. Based on lat-lon coding.
        """
    
    def brierscore(self, threshold = None, exceedquantile = False, groupers = ['leadtime']):
        """
        Check requirements for the forecast horizon of Buizza. Theirs is based on the crps. No turning the knob of extremity
        Asks a list of groupers, could use custom groupers. Also able to get climatological exceedence if climatology was supplied at initiazion.
        """
        
        if exceedquantile:
            self.frame['doy'] = self.frame['time'].dt.dayofyear
            delayed = self.frame.merge(self.clim, on = ['doy','latitude','longitude'], how = 'left')#.to_hdf('temp2.h5', key = 'climquant', format = 'table')
            #delayed = dd.read_hdf('temp2.h5', key = 'climquant')
            boolcols = list()
            # Boolean comparison cannot be made in one go. because it does not now how to compare N*members agains the climatology of N
            for member in delayed['forecast'].columns:
                name = '_'.join(['bool',str(member)])
                delayed[name] = delayed[('forecast',member)] > delayed['climatology']
                boolcols.append(name)
            delayed['pi'] = delayed[boolcols].sum(axis = 1) / len(delayed['forecast'].columns) # or use .count(axis = 1) if members might be NA
            delayed['oi'] = delayed['observation'] > delayed['climatology']
        else:
            delayed = self.frame
            delayed['pi'] = (delayed['forecast'] > threshold).sum(axis = 1) / len(delayed['forecast'].columns) # or use .count(axis = 1) if members might be NA
            delayed['oi'] = delayed['observation'] > threshold  
        
        delayed['dist'] = (delayed['pi'] - delayed['oi'])**2
        
        grouped = delayed.groupby(groupers)
        return(grouped['dist'].mean().compute())

#from dask.distributed import Client
#client = Client(processes = False)
        
#obs = SurfaceObservations(alias = 'tg')
#obs.load(tmin = '1995-05-14', tmax = '1998-09-01', llcrnr = (25,-30), rucrnr = (75,75))
#alignment = ForecastToObsAlignment(season = 'JJA', observations=obs)
#alignment.recollect(booksname='books_tg_JJA.csv')
#self = Comparison(alignment.alignedobject)
#brierpool = self.brierscore(threshold = 30)
#temp = self.brierscore(threshold = 1, groupers = ['leadtime','latitude', 'longitude'])
#self.setup_cv(nfolds = 3)
#testset = dd.read_hdf('/nobackup/users/straaten/match/tg_JJA_9a820b55d52b4a57974c8389f1f3176d.h5', key = 'intermediate')
#subset = testset.compute().iloc[:10000,:]
#subset.to_hdf('/usr/people/straaten/Documents/python_tests/subset.h5', key = 'intermediate', format = 'table')
#subset = dd.read_hdf('/usr/people/straaten/Documents/python_tests/subset.h5', key = 'intermediate')
#self = Comparison(subset)

