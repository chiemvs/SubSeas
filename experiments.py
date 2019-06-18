#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 12:11:08 2019

@author: straaten
"""

import os
import numpy as np
import xarray as xr
import pandas as pd
import dask.dataframe as dd
import dask.config
from observations import SurfaceObservations, Climatology, EventClassification
from forecasts import Forecast, ModelClimatology
from comparison import ForecastToObsAlignment, Comparison, ScoreAnalysis
from fitting import NGR, Logistic
import itertools
from copy import copy

class Experiment(object):
    
    def __init__(self, expname, basevar, cycle, season, newvar = None, method = 'mean', timeaggregations = ['1D', '2D', '3D', '4D', '7D'], spaceaggregations = [0.25, 0.75, 1.5, 3], quantiles = [0.5, 0.9, 0.95]):
        """
        Setting the relevant attributes. Timeaggregations are pandas frequency strings, spaceaggregations are floats in degrees.
        Quantiles can be None if you are already investigating an event variable and if you want to crps-score the whole distribution.
        But should be a multitude if these are desired.
        """
        self.resultsdir = '/nobackup/users/straaten/results/'
        self.expname = expname
        self.basevar = basevar
        self.newvar = newvar
        self.cycle = cycle
        self.season = season
        self.method = method
        self.timeaggregations = timeaggregations
        self.spaceaggregations = spaceaggregations
        self.quantiles = quantiles
    
    def setuplog(self):
        """
        Load an experiment log if it is present. Otherwise create one.
        Columns are for obsname and booksname, and possibly the highres climatologies. 
        For climname, scorefiles, bootstrap scores the amount of columns is times the amount of quantiles.
        """
        self.logpath = self.resultsdir + self.expname + '.h5'
        try:
            self.log = pd.read_hdf(self.logpath, key = 'exp')
        except OSError:
            if self.quantiles is not None:
                self.log = pd.DataFrame(data = None, index = pd.MultiIndex.from_product([self.spaceaggregations, self.timeaggregations, self.quantiles], names = ['spaceagg','timeagg','quantile']), 
                                        columns = ['climname','scorefiles','bootstrap','scores'])
                self.log = self.log.unstack(level = -1)
            else:
                self.log = pd.DataFrame(data = None, index = pd.MultiIndex.from_product([self.spaceaggregations, self.timeaggregations], names = ['spaceagg','timeagg']), 
                                        columns = pd.MultiIndex.from_product([['climname','scorefiles','bootstrap','scores'],['']])) # Also two levels for compatibility
            self.log = self.log.assign(**{'obsname':None,'booksname':None, 'obsclim':None,'modelclim':None}) # :''
    
    def savelog(self):
        """
        Saves the experiment log if it is present
        """
        if hasattr(self, 'log'):
            self.log.to_hdf(self.logpath, key = 'exp')
        else:
            print('No log to save')
    
    def iterateaggregations(self, func, column, overwrite = False, kwargs = {}):
        """
        Wrapper that calls the function with a specific spaceagg and timeagg and writes the returns of the function to the column if a name was given
        Invokes the quantile multiplier if the desired column has multiple quantiles
        """
        for spaceagg, timeagg in itertools.product(self.spaceaggregations[::-1],self.timeaggregations[::-1]):
            if self.log.loc[(spaceagg, timeagg),column].isna().any() or overwrite:
                multicolumns = len(self.log.loc[(spaceagg, timeagg),column]) > 1
                f = self.quantile_decorator(getattr(self, func), multiply= multicolumns)
                ret = f(spaceagg, timeagg, **kwargs)
                if (ret is not None):
                    self.log.loc[(spaceagg, timeagg),column] = ret
                    self.savelog()
            else:
                print('already filled')
    
    def prepareobs(self, spaceagg, timeagg, tmin, tmax, llcrnr = (25,-30), rucrnr = (75,75)):
        """
        Writes the observations that will be matched to forecasts. Also applies minfilter for average minimum amount of daily observations per season.
        """
        obs = SurfaceObservations(self.basevar)
        obs.load(tmin = tmin, tmax = tmax, llcrnr = llcrnr, rucrnr = rucrnr)
        obs.minfilter(season = self.season, n_min_per_seas = 40)
        
        if self.newvar == 'anom': # If anomalies then first highres classification, and later on the aggregation.
            highresclim = Climatology(self.basevar, **{'name':self.log.loc[(spaceagg, timeagg),('obsclim','')]})
            highresclim.localclim()
            getattr(EventClassification(obs, **{'climatology':highresclim}), self.newvar)(inplace = True) # Gives newvar attribute to the observations. Read in climatology.

        if timeagg != '1D':
            obs.aggregatetime(freq = timeagg, method = self.method)
        if spaceagg != 0.25:
            obs.aggregatespace(step = spaceagg, method = self.method, by_degree = True)
        
        if self.newvar is not None and self.newvar != 'anom':
            getattr(EventClassification(obs), self.newvar)(inplace = True)
            
        obs.savechanges()
        return(obs.name)
    
    def match(self, spaceagg, timeagg, loadkwargs = {}):
        """
        Writes the intermediate files. And returns the (possibly appended) booksname
        """
        if self.newvar is None:
            obs = SurfaceObservations(self.basevar, **{'name':self.log.loc[(spaceagg, timeagg),('obsname','')]})
        else:
            obs = SurfaceObservations(self.basevar, **{'name':self.log.loc[(spaceagg, timeagg),('obsname','')], 'newvar':self.newvar})
        obs.load()
        
        if self.newvar == 'anom':
            highresmodelclim = ModelClimatology(cycle=self.cycle, variable = self.basevar, 
                                                **{'name':self.log.loc[(spaceagg, timeagg),('modelclim','')]}) # Name for loading
            highresmodelclim.local_clim()
            highresmodelclim.change_units(newunit = obs.array.attrs['new_units'])
            newvarkwargs={'climatology':highresmodelclim}
        else:
            newvarkwargs={}
        alignment = ForecastToObsAlignment(season = self.season, observations=obs, cycle=self.cycle, **{'expname':self.expname})
        alignment.find_forecasts()
        alignment.load_forecasts(n_members = 11, loadkwargs = loadkwargs)
        alignment.match_and_write(newvariable = (self.newvar is not None), 
                                  newvarkwargs = newvarkwargs, 
                                  matchtime = (timeagg != '1D'), 
                                  matchspace= (spaceagg != 0.25))

        return(alignment.books_name)
    
    def makeclim(self, spaceagg, timeagg, climtmin, climtmax, llcrnr = (25,-30), rucrnr = (75,75), quantile = ''):
        """
        Possibility to make climatologies based on a longer period than the observations,
        when climtmin and climtmax are supplied.
        No observation minfilter needed. Climatology has its own filters.
        """
        obs = SurfaceObservations(self.basevar)
        obs.load(tmin = climtmin, tmax = climtmax, llcrnr = llcrnr, rucrnr = rucrnr)
        dailyobs = SurfaceObservations(self.basevar)
        dailyobs.load(tmin = climtmin, tmax = climtmax, llcrnr = llcrnr, rucrnr = rucrnr)
        
        if self.newvar == 'anom': # If anomalies then first highres classification, and later on the aggregation.
            highresclim = Climatology(self.basevar, **{'name':self.log.loc[(spaceagg, timeagg),('obsclim','')]})
            highresclim.localclim()
            getattr(EventClassification(obs, **{'climatology':highresclim}), self.newvar)(inplace = True) # Gives newvar attribute to the observations. Read in climatology.
            dailyobs = copy(obs) # Saves some computation time

        if timeagg != '1D':
            obs.aggregatetime(freq = timeagg, method = self.method)
        if spaceagg != 0.25:
            obs.aggregatespace(step = spaceagg, method = self.method, by_degree = True)
            dailyobs.aggregatespace(step = spaceagg, method = self.method, by_degree = True) # Aggregate the dailyobs for the climatology, daily_obs are temporally aggregated in the climatology.
        
        if self.newvar is not None and self.newvar != 'anom':
            getattr(EventClassification(obs), self.newvar)(inplace = True) # Only the observation needs to be transformed. Daily_obs are transformed after aggregation in local_climatology
    
        climatology = Climatology(self.basevar if self.newvar is None else self.basevar + '-' + self.newvar )
        if isinstance(quantile, float):
            climatology.localclim(obs = obs, daysbefore = 5, daysafter=5, mean = False, quant = quantile, daily_obs = dailyobs)
        else:
            if self.newvar is not None and self.newvar != 'anom': # Making a probability climatology of the binary newvar
                climatology.localclim(obs = obs, daysbefore = 5, daysafter=5, mean = True, quant = None, daily_obs = dailyobs)           
            else: # Make a 'random draws' climatology.
                climatology.localclim(obs = obs, daysbefore = 5, daysafter=5, mean = False, quant = None, n_draws = 11, daily_obs = dailyobs)
        climatology.savelocalclim()

        return(climatology.name)
    
    def makehighresmodelclim(self, spaceagg, timeagg, climtmin, climtmax, llcrnr = (None,None), rucrnr = (None,None)):
        """
        Only needed for later subtraction of mean from model fiels to create the anom variable (in matching)
        Performed at the highest possible model resolution (daily and 0.38 degrees). 
        Only done once (at the first iteration) and for other iterations this name is copied.
        Changing to the correct (observation) units is done later in the matching.
        """
        loadkwargs = dict(llcrnr = llcrnr, rucrnr = rucrnr)
        if spaceagg == self.spaceaggregations[-1] and timeagg == self.timeaggregations[-1]:
            modelclim = ModelClimatology(self.cycle,self.basevar)
            modelclim.local_clim(tmin = climtmin, tmax = climtmax, timemethod = '1D', daysbefore = 5, daysafter = 5, loadkwargs = loadkwargs)
            modelclim.savelocalclim()
            climnames = modelclim.name
        else:
            climnames = self.log.loc[(self.spaceaggregations[-1], self.timeaggregations[-1]), ('modelclim','')]
        return(climnames)
    
    def makehighresobsclim(self, spaceagg, timeagg, climtmin, climtmax, llcrnr = (25,-30), rucrnr = (75,75)):
        """
        Only needed for later subtraction of mean from observations to create the anom variable (in prepareobs and makeclim)
        Performed at the highest possible observation resolution (daily and 0.25 degrees). 
        Only done once (at the first iteration) and for other iterations this name is copied.
        """
        if spaceagg == self.spaceaggregations[-1] and timeagg == self.timeaggregations[-1]:
            obs = SurfaceObservations(self.basevar)
            obs.load(tmin = climtmin, tmax = climtmax, llcrnr = llcrnr, rucrnr = rucrnr)
            climatology = Climatology(self.basevar)
            climatology.localclim(obs = obs, daysbefore = 5, daysafter=5, mean = True, quant = None)
            climatology.savelocalclim()
            climnames = climatology.name
        else:
            climnames = self.log.loc[(self.spaceaggregations[-1], self.timeaggregations[-1]), ('obsclim','')]
        return(climnames)
    
    def score(self, spaceagg, timeagg, pp_model = None, quantile = ''):
        """
        Read the obs and clim. make a comparison object which computes the scores in the dask dataframe. 
        This dask dataframe is exported.
        Returns a list with intermediate filenames of the raw, climatological and corrected scores.
        Has a post-processing step if the pp_model is supplied. Fit is the same regardless of the quantile, so done only once.
        """
        alignment = ForecastToObsAlignment(season = self.season, cycle=self.cycle)
        alignment.recollect(booksname = self.log.loc[(spaceagg, timeagg),('booksname','')])
        
        climatology = Climatology(self.basevar, **{'name':self.log.loc[(spaceagg, timeagg),('climname', quantile)]})
        climatology.localclim() # loading in this case. Creation was done in the makeclim method.
        comp = Comparison(alignment = alignment, climatology = climatology)
                
        if isinstance(quantile, float):
            # Only in the first instance we are going to fit a model. Attributes are stored in memory and joined to the comp objects for other quantiles.
            if not pp_model is None:
                if self.quantiles.index(quantile) == 0:
                    comp.fit_pp_models(pp_model= pp_model, groupers = ['leadtime','latitude','longitude'])
                    firstfitname = comp.export(fits = True, frame = False)
                    firstfitgroupers = comp.fitgroupers
                    firstfitcoefcols = comp.coefcols
                else:
                    comp.fits = pd.read_hdf(comp.basedir + firstfitname + '.h5', key = 'fits') # Loading of the fits of the first quantile.
                    comp.fitgroupers = firstfitgroupers
                    comp.coefcols = firstfitcoefcols
                comp.make_pp_forecast(pp_model = pp_model)
            comp.brierscore()
            scorefile = comp.export(fits=False, frame = True)
        else:
            if not pp_model is None:
                comp.fit_pp_models(pp_model = pp_model, groupers = ['leadtime','latitude','longitude'])
                comp.export(fits=True, frame = False)
                comp.make_pp_forecast(pp_model = pp_model, n_members = 11 if isinstance(pp_model, NGR) else None)
            if (self.newvar is None) or (self.newvar == 'anom'):
                comp.crpsscore()
            else:
                comp.brierscore()
            scorefile = comp.export(fits=False, frame = True)
            
        return(scorefile)
    
    def bootstrap_scores(self, spaceagg, timeagg, bootstrapkwargs = dict(n_samples = 200, fixsize = 60), quantile = ''):
        """
        Will bootstrap the scores in the scoreanalysis files and export these samples 
        Such that these can be later analyzed in the skill function.
        """
        scoreanalysis = ScoreAnalysis(scorefile = self.log.loc[(spaceagg, timeagg),('scorefiles', quantile)], timeagg = timeagg)
        scoreanalysis.load()
        result = scoreanalysis.block_bootstrap_local_skills(**bootstrapkwargs)
        return(result)
    
    def save_charlengths(self, spaceagg, timeagg, quantile = ''):
        """
        Invokes characteristic timescale computation in the ScoreAnalysis object and returns the field
        """
        scoreanalysis = ScoreAnalysis(scorefile = self.log.loc[(spaceagg, timeagg),('scorefiles', quantile)], timeagg = timeagg)
        scoreanalysis.load()
        scoreanalysis.characteristiclength()
        return(scoreanalysis.charlengths)
    
    def skill(self, spaceagg, timeagg, usebootstrapped = False, analysiskwargs = {}, quantile = ''):
        """
        Reads the exported scoreanalysis file and analyses it. 
        Standard is to compute the mean score. But when bootstrapping has been performed in the previous step
        and when usebootstrapped is set to True then quantiles, forecast horizons, 
        can all be computed. These options are controlled with analysiskwargs
        """
        scoreanalysis = ScoreAnalysis(scorefile = self.log.loc[(spaceagg, timeagg),('scorefiles', quantile)], timeagg = timeagg)
        bootstrap_ready = [(spaceagg, timeagg),('bootstrap', quantile)]
        if usebootstrapped and bootstrap_ready:
            skillscore = scoreanalysis.process_bootstrapped_skills(**analysiskwargs)
        else:
            print('mean scoring')
            scoreanalysis.load()
            skillscore = scoreanalysis.mean_skill_score(**analysiskwargs)
            
        return(skillscore)
        
    def quantile_decorator(self, f, multiply = False):
        """
        Multiplying the returns with the amount of desired quantiles. And supplying this quantile to the function.
        Always making sure that an array is returned which is easily written to the pandas log
        """
        def wrapped(*args, **kwargs):
            if multiply:
                res = np.repeat(None,len(self.quantiles))
                for quantile in self.quantiles:
                    res[self.quantiles.index(quantile)] = f(*args, **kwargs, quantile = quantile)
                return(res)
            else:
                return(np.array(f(*args, **kwargs)))
        return(wrapped)


"""
Max temperature benchmarks.
"""
## Calling of the class        
#test1 = Experiment(expname = 'test1', basevar = 'tx', cycle = '41r1', season = 'JJA', method = 'max', timeaggregations = ['1D', '2D', '3D', '4D', '7D'], spaceaggregations = [0.25, 0.75, 1.5, 3], quantiles = [0.5, 0.9, 0.95])
#test1.setuplog()
##test1.iterateaggregations(func = 'prepareobs', column = 'obsname', kwargs = {'tmin':'1996-05-30','tmax':'2006-08-31'})
##test1.iterateaggregations(func = 'match', column = 'booksname')
##test1.iterateaggregations(func = 'makeclim', column = 'climname', kwargs = {'climtmin':'1980-05-30','climtmax':'2010-08-31'})
##test1.iterateaggregations(func = 'score', column = 'scores')
#
#scoreseries = test1.log.loc[:,('scores',slice(None,None,None))].stack(level = -1)['scores']
#temp = pd.concat(scoreseries.tolist(), keys = scoreseries.index)
#temp.index.set_names(scoreseries.index.names, level = [0,1,2], inplace=True)
#skill = 1 - temp['rawbrier'] / temp['climbrier'] # Nice one to write out.
##skill.to_hdf('/nobackup/users/straaten/results/test1skill.h5', key = 'skill')
#
## Plotting
#skill.loc[(0.75,slice(None), 0.9)].unstack(level = [0,1,2]).plot()
#skill.xs(0.5, level = 'quantile', drop_level = False).unstack(level = [0,1,2]).plot()
#


"""
Mean temperature benchmarks. Observations split into two decades. Otherwise potential memory error in matching.
"""

dask.config.set(temporary_directory='/nobackup_1/users/straaten/')

# Calling of the class        
#test2 = Experiment(expname = 'test2', basevar = 'tg', cycle = '41r1', season = 'DJF', method = 'mean', 
#                   timeaggregations = ['1D', '2D', '3D', '4D', '5D', '6D', '7D'], spaceaggregations = [0.25, 0.75, 1.25, 2, 3], quantiles = [0.1, 0.15, 0.25, 0.33, 0.66])
#test2.setuplog()
#test2.log = test2.log.assign(**{'obsname2':None}) # Extra observation column.
#test2.iterateaggregations(func = 'prepareobs', column = 'obsname', kwargs = {'tmin':'1995-11-30','tmax':'2005-02-28'}) 
#test2.iterateaggregations(func = 'prepareobs', column = 'obsname2', kwargs = {'tmin':'2005-03-01','tmax':'2015-02-28'})

#test2.iterateaggregations(func = 'match', column = 'booksname', kwargs = {'obscol':'obsname'})
#test2.iterateaggregations(func = 'match', column = 'booksname', kwargs = {'obscol':'obsname2'}, overwrite = True) # Replaces with updated books name.
#test2.iterateaggregations(func = 'makeclim', column = 'climname', kwargs = {'climtmin':'1980-05-30','climtmax':'2015-02-28'})
#test2.iterateaggregations(func = 'score', column = 'scorefiles', kwargs = {'pp_model':NGR()})
#test2.iterateaggregations(func = 'skill', column = 'scores', overwrite = True)

# Get all the scores for up to 0.75
#scoreseries = test2.log.loc[(slice(0.75,None,None),slice(None)),('scores', slice(None))].stack(level = -1)['scores']
#skillframe = pd.concat(scoreseries.tolist(), keys = scoreseries.index)
#skillframe.index.set_names(scoreseries.index.names, level = [0,1,2], inplace=True)
# Change the time aggregation index to integer days.
#renamedict = {a:int(a[0]) for a in np.unique(skillframe.index.get_level_values(1).to_numpy())}
#skillframe.rename(renamedict, index = 1, inplace = True)
# skillframe.sort_index()[['rawbrierskill','corbrierskill']].to_hdf('/nobackup/users/straaten/results/exp2_skill.h5', key = 'all_europe_mean')

# Get all the scores for up to 0.75
#scoreseries = test2.log.loc[(slice(0.75,None,None),slice(None)),('scores', slice(None))].stack(level = -1)['scores']
#skillframe = pd.concat(scoreseries.tolist(), keys = scoreseries.index)
#skillframe.index.set_names(scoreseries.index.names, level = [0,1,2], inplace=True)
# Change the time aggregation index to integer days.
#renamedict = {a:int(a[0]) for a in np.unique(skillframe.index.get_level_values(1).to_numpy())}
#skillframe.rename(renamedict, index = 1, inplace = True)
#skillframe['rawbrierskill'] = pd.to_numeric(skillframe['rawbrierskill'], downcast = 'float')
#skillframe['corbrierskill'] = pd.to_numeric(skillframe['corbrierskill'], downcast = 'float')
#skillframe.sort_index()[['rawbrierskill','corbrierskill']].to_hdf('/nobackup/users/straaten/results/exp2_skill.h5', key = 'local_mean')

"""
Experiment 3 setup. Climatology period same as exp 2. Make sure it does not append to bookfiles of experiment 2. Actually the same matchfiles can be used.
"""    
#test3 = Experiment(expname = 'test3', basevar = 'tg', cycle = '41r1', season = 'DJF', method = 'mean', 
#                   timeaggregations = ['1D', '2D', '3D', '4D', '5D', '6D', '7D'], spaceaggregations = [0.25, 0.75, 1.25, 2, 3], quantiles = None)
#test3.setuplog()
# Assigned the same matchfiles.
###test3.log.loc[:,('booksname','')] = test2.log.loc[:,('booksname','')]
###test3.iterateaggregations(func = 'prepareobs', column = 'obsname', kwargs = {'tmin':'1995-11-30','tmax':'2005-02-28'}) TODO: do not forget the filter step
#test3.iterateaggregations(func = 'makeclim', column = 'climname', kwargs = {'climtmin':'1980-05-30','climtmax':'2015-02-28'})
###test3.iterateaggregations(func = 'match', column = 'booksname', kwargs = {'obscol':'obsname'})
#test3.iterateaggregations(func = 'score', column = 'scorefiles', kwargs = {'pp_model':NGR(double_transform=True)})
#test3.iterateaggregations(func = 'skill', column = 'scores', overwrite = True)

"""
Experiment 4 setup. Western europe only. Same climatology period.
"""  
#test4 = Experiment(expname = 'west_eur', basevar = 'tg', cycle = '41r1', season = 'DJF', method = 'mean',
#                   timeaggregations = ['1D', '2D', '3D', '4D', '5D', '6D', '7D'], spaceaggregations = [0.25, 0.75, 1.25, 2, 3], quantiles = None)
#test4.setuplog()
#test4.iterateaggregations(func = 'prepareobs', column = 'obsname', kwargs = dict(tmin = '1995-01-01',tmax = '2015-01-10', llcrnr = (45,0), rucrnr = (55,6)))
#test4.iterateaggregations(func = 'makeclim', column = 'climname', kwargs = dict(climtmin = '1995-01-01', climtmax = '2015-01-10', llcrnr = (45,0), rucrnr = (55,6)))
#test4.iterateaggregations(func = 'match', column = 'booksname', kwargs = {'obscol':'obsname'})
#test4.iterateaggregations(func = 'score', column = 'scorefiles', kwargs = {'pp_model':NGR(double_transform=True)})
#test4.iterateaggregations(func = 'skill', column = 'scores', overwrite = True)


"""
Experiment 5 setup Probability of Precipitation.
"""
#self = Experiment(expname = 'test5', basevar = 'rr', newvar = 'pop', cycle = '41r1', season = 'DJF', method = 'mean', 
#                   timeaggregations = ['7D'], spaceaggregations = [3], quantiles = None)
#self.setuplog()
#self.iterateaggregations(func = 'prepareobs', column = 'obsname', kwargs = dict(tmin = '1995-01-01',tmax = '2015-02-10',  llcrnr = (45,0), rucrnr = (55,6)))
#self.iterateaggregations(func = 'match', column = 'booksname', kwargs = {'loadkwargs' : dict( llcrnr = (45,0), rucrnr = (55,6))})
#self.iterateaggregations(func = 'makeclim', column = 'climname', kwargs = dict(climtmin = '2000-01-01', climtmax = '2010-02-10', llcrnr = (45,0), rucrnr = (55,6)))
#self.iterateaggregations(func = 'score', column = 'scorefiles', kwargs = {'pp_model':Logistic()})
#self.iterateaggregations(func = 'bootstrap_scores', column = 'bootstrap', kwargs = {'bootstrapkwargs':dict(n_samples = 200, fixsize = True)})
#self.iterateaggregations(func = 'skill', column = 'scores', overwrite = True, kwargs = {'usebootstrapped' :True, 'analysiskwargs':dict(local = True, fitquantiles = False, forecast_horizon = False)})
#self.iterateaggregations(func = 'skill', column = 'scores', overwrite = True, kwargs = {'usebootstrapped' :False, 'analysiskwargs':dict(groupers = ['leadtime'])})

"""
Experiment 6 anomalies test for western Europe
"""
#self = Experiment(expname = 'westa6', basevar = 'tg', newvar = 'anom', cycle = '41r1', season = 'DJF', method = 'mean', 
#                   timeaggregations = ['1D','2D','3D','4D'], spaceaggregations = [0.25,0.75,1.25,2], quantiles = None)
#self.setuplog()
#self.iterateaggregations(func = 'makehighresobsclim', column = 'obsclim', kwargs = dict(climtmin = '1995-01-01',climtmax = '2015-01-11', llcrnr = (45,0), rucrnr = (55,6)))
#self.iterateaggregations(func = 'prepareobs', column = 'obsname', kwargs = dict(tmin = '1995-01-01',tmax = '2015-01-11', llcrnr = (45,0), rucrnr = (55,6)))
#self.iterateaggregations(func = 'makeclim', column = 'climname', kwargs = dict(climtmin = '1995-01-01', climtmax = '2015-01-11', llcrnr = (45,0), rucrnr = (55,6)))
#self.iterateaggregations(func = 'makehighresmodelclim', column = 'modelclim', kwargs = dict(climtmin = '1995-01-01', climtmax = '2015-01-11', llcrnr = (45,0), rucrnr = (55,6)))
#self.iterateaggregations(func = 'match', column = 'booksname', kwargs = {'loadkwargs' : dict( llcrnr = (45,0), rucrnr = (55,6))})
#self.iterateaggregations(func = 'score', column = 'scorefiles', kwargs = {'pp_model':NGR(double_transform = True)})
#self.iterateaggregations(func = 'bootstrap_scores', column = 'bootstrap', kwargs = {'bootstrapkwargs':dict(n_samples = 200, fixsize = True)})
#self.iterateaggregations(func = 'skill', column = 'scores', overwrite = True, kwargs = {'usebootstrapped' :True, 'analysiskwargs':dict(local = True, fitquantiles = False, forecast_horizon = True)})
#t = pd.concat(self.log['scores'].tolist(), keys = self.log.index, names=['spaceagg','timeagg'])

"""
Experiment 7 Maximum temperature for western Europe
"""
#
#test7 = Experiment(expname = 'westtx7', basevar = 'tx', cycle = '41r1', season = 'JJA', method = 'max',
#                   timeaggregations = ['1D', '2D', '3D', '4D', '5D', '6D', '7D'], spaceaggregations = [0.25], quantiles = None)
#test7.setuplog()
#test7.iterateaggregations(func = 'prepareobs', column = 'obsname', kwargs = dict(tmin = '1995-01-01',tmax = '2015-01-10', llcrnr = (45,0), rucrnr = (55,6)))
#test7.iterateaggregations(func = 'makeclim', column = 'climname', kwargs = dict(climtmin = '1995-01-01', climtmax = '2015-01-10', llcrnr = (45,0), rucrnr = (55,6)))
#test7.iterateaggregations(func = 'match', column = 'booksname', kwargs = {'loadkwargs' : dict( llcrnr = (45,0), rucrnr = (55,6))})
#test7.iterateaggregations(func = 'score', column = 'scorefiles', kwargs = {'pp_model':NGR(double_transform=True)})
#test7.iterateaggregations(func = 'skill', column = 'scores', overwrite = True)

"""
Experiment 8 anomalies maximum temperatures test for western Europe
"""
#test8 = Experiment(expname = 'westtxa8', basevar = 'tx', newvar = 'anom', cycle = '41r1', season = 'JJA', method = 'max', 
#                   timeaggregations = ['1D','2D','3D','4D','5D','6D','7D'], spaceaggregations = [0.25,0.75,1.25,2,3], quantiles = None)
#test8.setuplog()
#test8.iterateaggregations(func = 'makehighresobsclim', column = 'obsclim', kwargs = dict(climtmin = '1995-01-01',climtmax = '2015-01-11', llcrnr = (45,0), rucrnr = (55,6)))
#test8.iterateaggregations(func = 'prepareobs', column = 'obsname', kwargs = dict(tmin = '1995-01-01',tmax = '2015-01-11', llcrnr = (45,0), rucrnr = (55,6)))
#test8.iterateaggregations(func = 'makeclim', column = 'climname', kwargs = dict(climtmin = '1995-01-01', climtmax = '2015-01-11', llcrnr = (45,0), rucrnr = (55,6)))
#test8.iterateaggregations(func = 'makehighresmodelclim', column = 'modelclim', kwargs = dict(climtmin = '1995-01-01', climtmax = '2015-01-11', llcrnr = (45,0), rucrnr = (55,6)))
#test8.iterateaggregations(func = 'match', column = 'booksname', kwargs = {'loadkwargs' : dict( llcrnr = (45,0), rucrnr = (55,6))})
#test8.iterateaggregations(func = 'score', column = 'scorefiles', kwargs = {'pp_model':NGR(double_transform = True)})
#test8.iterateaggregations(func = 'bootstrap_scores', column = 'bootstrap', kwargs = {'bootstrapkwargs':dict(n_samples = 200, fixsize = 60)})
#test8.iterateaggregations(func = 'skill', column = 'scores', overwrite = True, kwargs = {'usebootstrapped' :True, 'analysiskwargs':dict(local = True, fitquantiles = False, forecast_horizon = True, skillthreshold = 0.2, average_afterwards = True)})

"""
Europe wide highres characteristic timescales
"""
# Would there be a difference between the characteristic timescales of the anomalies and of the regular values? The first have removed seasonality?
# Do this through the created scorefiles at high resolution. No post-processing.
#test9 = Experiment(expname = 'chartimescale', basevar = 'tg', cycle = '41r1', season = 'DJF', method = 'mean', 
#                   timeaggregations = ['1D','2D','3D','4D','5D','6D','7D'], spaceaggregations = [0.25], quantiles = None) # 
#test9.setuplog()
#test9.iterateaggregations(func = 'prepareobs', column = 'obsname', kwargs = dict(tmin = '1995-01-01', llcrnr = (None,-30), rucrnr = (None,48)))
#test9.iterateaggregations(func = 'makeclim', column = 'climname', kwargs = dict(climtmin = '1995-01-01', climtmax = '1999-01-11', llcrnr = (None,-30), rucrnr = (None,48))) 
#test9.iterateaggregations(func = 'match', column = 'booksname', kwargs = {'loadkwargs':dict(llcrnr = (None,-30), rucrnr = (None,48))})
#test9.iterateaggregations(func = 'score', column = 'scorefiles')
#test9.log['charlengths'] = None
#test9.iterateaggregations(func = 'save_charlengths', column = 'charlengths')