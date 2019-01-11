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
import itertools
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
                        forecast.aggregatetime(freq = freq, method = method)
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
        Neirest neighbouring to match pairs. Also converts forecast units to observed units.
        Creates the dataset and writes it to disk. Possibly empties basket and writes to disk 
        at intermediate steps if intermediate results press too much on memory.
        """
        import uuid
        
        aligned_basket = []
        self.outfiles = []
        
        # Make sure the first parts are recognizable as the time method and the space method. Make last part unique
        def write_outfile(basket, filetype = ['csv','h5']):
            characteristics = [self.season]
            for m in ['timemethod','spacemethod']:
                if hasattr(self.obs, m):
                    characteristics.append(getattr(self.obs, m))
            filepath = self.basedir + '_'.join(characteristics) + '_' + uuid.uuid4().hex + '.' + filetype
            if filetype == 'h5':
                pd.concat(basket).to_hdf(filepath, key = 'intermediate', mode = 'w') # format = 'table'
            elif filetype == 'csv':
                pd.concat(basket).to_csv(filepath)
            else:
                raise ValueError('give filetype as h5 or csv')
            self.outfiles.append(filepath)
            print('write out', filepath)
        
        for date, listofforecasts in self.forecasts.items():
            
            if listofforecasts:
                fieldobs = self.obs.array.sel(time = date).drop('time')
                
                allleadtimes = xr.concat(objs = [f.array.swap_dims({'time':'leadtime'}) for f in listofforecasts], dim = 'leadtime') # concatenates over leadtime dimension.
                a,b = unitconversionfactors(xunit = allleadtimes.units, yunit = fieldobs.units)
                exp = allleadtimes.reindex_like(fieldobs, method='nearest') * a + b
                
                # Merging, exporting to pandas and masking by dropping on NA observations.
                combined = xr.Dataset({'forecast':exp.drop('time'), 'observation':fieldobs}).to_dataframe().dropna(axis = 0, )
                
                # puts members in columns. observerations are duplicated so therefore selects up to n_members +1
                temp = combined.unstack('number').iloc[:,:(self.n_members + 1)] 
                
                # Downcasting float precision Not possible as float32 is the lowest precision.
                #converted = temp.select_dtypes(include = ['float']).apply(pd.to_numeric,downcast='float16')
                #temp[converted.columns] = converted
                
                # prepend with the time index.
                aligned_basket.append(pd.concat([temp], keys=[date], names=['time'])) # temp.swaplevel(1,2, axis = 0)
                print(date, 'matched')
                
                # If aligned takes too much system memory (> 3Gb) . Write it out
                if sys.getsizeof(aligned_basket[0]) * len(aligned_basket) > 3*10**9:
                    write_outfile(aligned_basket, filetype='h5')
                    aligned_basket = []
        
        # After last loop also write out 
        if aligned_basket:
            write_outfile(aligned_basket, filetype='h5')
                 
        self.final_units = self.obs.array.units
    
    def recollect(self):
        """
        Makes a dask dataframe object.
        """
        self.alignedobject = dd.read_csv(self.outfiles, 'intermediate')
        
obs = SurfaceObservations(alias = 'rr')
obs.load(tmin = '1995-06-14', tmax = '1995-06-25')
#obs.aggregatetime(freq = 'w', method = 'mean')
#obs.aggregatespace(step = 5, method = 'mean', by_degree = False)

test = ForecastToObsAlignment(season = 'JJA', observations=obs)
test.find_forecasts()
test.load_forecasts(n_members=11)
test.match_and_write()

temp = pd.read_hdf(test.outfiles[0])
#test2 = SurfaceObservations(alias = 'rr', tmin = '1950-01-01', tmax = '1990-01-01', timemethod = 'M_mean')
#test2.load()

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
    """
    
    def __init__(self, alignedobject):
        """
        Observation | members | 
        """
    
    def post_process():
        """
        jpsjods
        """
    
    def setup_cv():
        """
        okeedk
        """
    
    def score():
        """
        Check requirements for the forecast horizon of Buizza.
        """
    
        
