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
    
    def __init__(self, expname, basevar, cycle, season, clustername, newvar = None, method = 'mean', rolling = False, timeaggregations = ['1D', '2D', '3D', '4D', '7D'], spaceaggregations = [0,0.005,0.01,0.025,0.05,0.1,0.2,0.3,0.4,0.5,1], quantiles = [0.5, 0.9, 0.95]):
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
        self.rolling = rolling
        self.clustername = clustername
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
            obs.aggregatetime(freq = timeagg, method = self.method, rolling = self.rolling)
        if spaceagg != 0.25:
            obs.aggregatespace(level = spaceagg, clustername = self.clustername, method = self.method)
        
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
        alignment = ForecastToObsAlignment(season = self.season, observations=obs, cycle=self.cycle, n_members = 11, **{'expname':self.expname,'loadkwargs':loadkwargs})
        alignment.match_and_write(newvariable = (self.newvar is not None), 
                                  newvarkwargs = newvarkwargs, 
                                  matchtime = (timeagg != '1D'), 
                                  matchspace= True)

        return(alignment.books_name)
    
    def makeclim(self, spaceagg, timeagg, climtmin, climtmax, llcrnr = (25,-30), rucrnr = (75,75), quantile = ''):
        """
        Possibility to make climatologies based on a longer period than the observations,
        when climtmin and climtmax are supplied.
        No observation minfilter needed. Climatology has its own filters.
        """
        obs = SurfaceObservations(self.basevar)
        obs.load(tmin = climtmin, tmax = climtmax, llcrnr = llcrnr, rucrnr = rucrnr)
        
        if self.newvar == 'anom': # If anomalies then first highres classification, and later on the aggregation.
            highresclim = Climatology(self.basevar, **{'name':self.log.loc[(spaceagg, timeagg),('obsclim','')]})
            highresclim.localclim()
            getattr(EventClassification(obs, **{'climatology':highresclim}), self.newvar)(inplace = True) # Gives newvar attribute to the observations. Read in climatology.

        if timeagg != '1D':
            obs.aggregatetime(freq = timeagg, method = self.method, rolling = self.rolling)
        
        obs.aggregatespace(level = spaceagg, method = self.method, clustername = self.clustername)
        
        if self.newvar is not None and self.newvar != 'anom':
            getattr(EventClassification(obs), self.newvar)(inplace = True) # Only the observation needs to be transformed. Daily_obs are transformed after aggregation in local_climatology
    
        climatology = Climatology(self.basevar if self.newvar is None else self.basevar + '-' + self.newvar )
        if isinstance(quantile, float):
            climatology.localclim(obs = obs, daysbefore = 5, daysafter=5, mean = False, quant = quantile)
        else:
            if self.newvar is not None and self.newvar != 'anom': # Making a probability climatology of the binary newvar
                climatology.localclim(obs = obs, daysbefore = 5, daysafter=5, mean = True, quant = None)           
            else: # Make a 'random draws' climatology.
                climatology.localclim(obs = obs, daysbefore = 5, daysafter=5, mean = False, quant = None, n_draws = 11)
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
    
    def score(self, spaceagg, timeagg, store_minimum = False, pp_model = None, quantile = ''):
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
                    comp.fit_pp_models(pp_model= pp_model, groupers = ['leadtime','clustid'])
                    firstfitname = comp.export(fits = True, frame = False)
                    firstfitgroupers = comp.fitgroupers
                    firstfitcoefcols = comp.coefcols
                else:
                    comp.fits = dd.read_hdf(comp.basedir + firstfitname + '.h5', key = 'fits') # Loading of the fits of the first quantile.
                    comp.fitgroupers = firstfitgroupers
                    comp.coefcols = firstfitcoefcols
                comp.make_pp_forecast(pp_model = pp_model)
            comp.brierscore()
            scorefile = comp.export(fits=False, frame = True, store_minimum = store_minimum)
        else:
            if not pp_model is None:
                comp.fit_pp_models(pp_model = pp_model, groupers = ['leadtime','clustid'])
                comp.export(fits=True, frame = False)
                comp.make_pp_forecast(pp_model = pp_model, n_members = 11 if isinstance(pp_model, NGR) else None)
            if (self.newvar is None) or (self.newvar == 'anom'):
                comp.crpsscore()
            else:
                comp.brierscore()
            scorefile = comp.export(fits=False, frame = True, store_minimum = store_minimum)
            
        return(scorefile)
    
    def bootstrap_scores(self, spaceagg, timeagg, bootstrapkwargs = dict(n_samples = 200, fixsize = False), quantile = ''):
        """
        Will bootstrap the scores in the scoreanalysis files and export these samples 
        Such that these can be later analyzed in the skill function.
        """
        scoreanalysis = ScoreAnalysis(scorefile = self.log.loc[(spaceagg, timeagg),('scorefiles', quantile)], timeagg = timeagg, rolling = self.rolling)
        scoreanalysis.load()
        result = scoreanalysis.block_bootstrap_local_skills(**bootstrapkwargs)
        return(result)
    
    def save_charlengths(self, spaceagg, timeagg, quantile = ''):
        """
        Invokes characteristic timescale computation in the ScoreAnalysis object and returns the field
        """
        scoreanalysis = ScoreAnalysis(scorefile = self.log.loc[(spaceagg, timeagg),('scorefiles', quantile)], timeagg = timeagg, rolling = self.rolling)
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
        scoreanalysis = ScoreAnalysis(scorefile = self.log.loc[(spaceagg, timeagg),('scorefiles', quantile)], timeagg = timeagg, rolling = self.rolling)
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
                res = np.repeat(None,1)
                res[0] = f(*args, **kwargs)
                return(res)
        return(wrapped)

dask.config.set(temporary_directory='/nobackup_1/users/straaten/')

"""
Experiment 25 Test for cluster based aggregation, regular temperatures, some post-processing, only the larger end of spatial aggregation.
Split into two parts for parallel matching
"""
#test = Experiment(expname = 'clustga25', basevar = 'tg', newvar = 'anom', rolling = True, cycle = '45r1', season = 'DJF', clustername = 'tg-DJF',
#                 method = 'mean', timeaggregations= ['1D','3D','5D','7D','9D'], spaceaggregations=[0.05,0.1,0.2,0.3,0.5,1], quantiles = None)
#test.setuplog()
#test.iterateaggregations(func = 'makehighresobsclim', column = 'obsclim', kwargs = dict(climtmin = '1998-01-01', climtmax = '2018-12-31', llcrnr= (36,-24), rucrnr = (None,40)))
#test.iterateaggregations(func = 'prepareobs', column = 'obsname', kwargs = dict(tmin = '1998-06-07', tmax = '2018-12-31', llcrnr= (36,-24), rucrnr = (None,40)))
#test.iterateaggregations(func = 'makeclim', column = 'climname', kwargs = dict(climtmin = '1998-01-01', climtmax = '2018-12-31', llcrnr= (36,-24), rucrnr = (None,40)))
#test.iterateaggregations(func = 'makehighresmodelclim', column = 'modelclim', kwargs = dict(climtmin = '1998-06-07', climtmax = '2019-05-16', llcrnr= (36,-24), rucrnr = (None,40)))
#test.iterateaggregations(func = 'match', column = 'booksname', kwargs = {'loadkwargs' : dict(llcrnr= (36,-24), rucrnr = (None,40))})
#test.iterateaggregations(func = 'score', column = 'scorefiles', kwargs = {'pp_model':NGR()})
#test.iterateaggregations(func = 'bootstrap_scores', column = 'bootstrap', kwargs = {'bootstrapkwargs':dict(n_samples = 200, fixsize = False)})
#test.iterateaggregations(func = 'skill', column = 'scores', overwrite = True, kwargs = {'usebootstrapped' :True, 'analysiskwargs':dict(local = False, fitquantiles = False, forecast_horizon = False)})

#test2 = Experiment(expname = 'clustga25b', basevar = 'tg', newvar = 'anom', rolling = True, cycle = '45r1', season = 'DJF', clustername = 'tg-DJF',
#                 method = 'mean', timeaggregations= ['1D','3D','5D','7D','9D'], spaceaggregations=[0.05], quantiles = None)
#test2.setuplog()
##test2.log.loc[(0.05, slice(None)),:] = test.log.loc[(0.05, slice(None)),:]
##test2.savelog()
#test2.iterateaggregations(func = 'match', column = 'booksname', kwargs = {'loadkwargs' : dict(llcrnr= (36,-24), rucrnr = (None,40))})
#test2.iterateaggregations(func = 'score', column = 'scorefiles', kwargs = {'pp_model':NGR()})
#test2.iterateaggregations(func = 'bootstrap_scores', column = 'bootstrap', kwargs = {'bootstrapkwargs':dict(n_samples = 200, fixsize = False)})

"""
Experiment 26 Highest resolution, regular temperature, temperature anomalies and binarized precipitation. Two seasons each. Currently mean scoring.
"""
tgDJF = Experiment(expname = 'hr26tgDJF', basevar = 'tg', rolling = True, cycle = '45r1', season = 'DJF', clustername = 'tg-DJF',
        method = 'mean', timeaggregations = ['1D'], spaceaggregations = [0], quantiles = None)
tgDJF.setuplog()
#tgDJF.iterateaggregations(func = 'prepareobs', column = 'obsname', kwargs = dict(tmin = '1998-06-07', tmax = '2018-12-31', llcrnr= (36,-24), rucrnr = (None,40)))
#tgDJF.iterateaggregations(func = 'makeclim', column = 'climname', kwargs = dict(climtmin = '1998-01-01', climtmax = '2018-12-31', llcrnr= (36,-24), rucrnr = (None,40)))
#tgDJF.iterateaggregations(func = 'match', column = 'booksname', kwargs = {'loadkwargs' : dict(llcrnr= (36,-24), rucrnr = (None,40))})
tgDJF.iterateaggregations(func = 'score', column = 'scorefiles', kwargs = {'pp_model':NGR(), 'store_minimum':True})
tgDJF.iterateaggregations(func = 'skill', column = 'scores', kwargs = {'usebootstrapped':False, 'analysiskwargs':dict(groupers = ['leadtime','clustid'])})
#
#tgJJA = Experiment(expname = 'hr26tgJJA', basevar = 'tg', rolling = True, cycle = '45r1', season = 'JJA', clustername = 'tg-DJF',
#        method = 'mean', timeaggregations = ['1D'], spaceaggregations = [0], quantiles = None)
#tgJJA.setuplog()
##tgJJA.log.loc[:,['obsname','climname']] = tgDJF.log.loc[:,['obsname','climname']]
##tgJJA.savelog()
#tgJJA.iterateaggregations(func = 'match', column = 'booksname', kwargs = {'loadkwargs' : dict(llcrnr= (36,-24), rucrnr = (None,40))})
#tgJJA.iterateaggregations(func = 'score', column = 'scorefiles', kwargs = {'pp_model':NGR(), 'store_minimum':True})
#tgJJA.iterateaggregations(func = 'skill', column = 'scores', kwargs = {'usebootstrapped':False, 'analysiskwargs':dict(groupers = ['leadtime','clustid'])})
#
#tgaDJF = Experiment(expname = 'hr26tgaDJF', basevar = 'tg', newvar = 'anom', rolling = True, cycle = '45r1', season = 'DJF', clustername = 'tg-DJF', 
#        method = 'mean', timeaggregations= ['1D'], spaceaggregations=[0], quantiles = None)
#tgaDJF.setuplog()
##tgaDJF.log.loc[:,['obsclim','modelclim']] = test.log.loc[:,['obsclim','modelclim']].iloc[0].tolist() # Can be copied because even in test they are at the highest res. Not sure if this indexing works.
##tgaDJF.savelog()
#tgaDJF.iterateaggregations(func = 'prepareobs', column = 'obsname', kwargs = dict(tmin = '1998-06-07', tmax = '2018-12-31', llcrnr= (36,-24), rucrnr = (None,40)))
#tgaDJF.iterateaggregations(func = 'makeclim', column = 'climname', kwargs = dict(climtmin = '1998-01-01', climtmax = '2018-12-31', llcrnr= (36,-24), rucrnr = (None,40)))
#tgaDJF.iterateaggregations(func = 'match', column = 'booksname', kwargs = {'loadkwargs' : dict(llcrnr= (36,-24), rucrnr = (None,40))})
#tgaDJF.iterateaggregations(func = 'score', column = 'scorefiles', kwargs = {'pp_model':NGR(), 'store_minimum':True})
#tgaDJF.iterateaggregations(func = 'skill', column = 'scores', kwargs = {'usebootstrapped' :False, 'analysiskwargs':dict(groupers = ['leadtime','clustid'])})
#
#tgaJJA = Experiment(expname = 'hr26tgaJJA', basevar = 'tg', newvar = 'anom', rolling = True, cycle = '45r1', season = 'JJA', clustername = 'tg-DJF', 
#        method = 'mean', timeaggregations= ['1D'], spaceaggregations=[0], quantiles = None)
#tgaJJA.setuplog()
#tgaJJA.log.loc[:,['obsname','climname','obsclim','modelclim']] = tgaDJF.log.loc[:,['obsname','climname','obsclim','modelclim']]
#tgaJJA.iterateaggregations(func = 'match', column = 'booksname', kwargs = {'loadkwargs' : dict(llcrnr= (36,-24), rucrnr = (None,40))})
#tgaJJA.iterateaggregations(func = 'score', column = 'scorefiles', kwargs = {'pp_model':NGR(), 'store_minimum':True})
#tgaJJA.iterateaggregations(func = 'skill', column = 'scores', kwargs = {'usebootstrapped' :False, 'analysiskwargs':dict(groupers = ['leadtime','clustid'])})
#
#podDJF = Experiment(expname = 'hr26podDJF', basevar = 'rr', newvar = 'pod', rolling = True, cycle = '45r1', season = 'DJF', clustername = 'tg-DJF',
#        method = 'mean', timeaggregations = ['1D'], spaceaggregations = [0], quantiles = None)
#podDJF.setuplog()
#podDJF.iterateaggregations(func = 'prepareobs', column = 'obsname', kwargs = dict(tmin = '1998-06-07', tmax = '2018-12-31', llcrnr= (36,-24), rucrnr = (None,40)))
#podDJF.iterateaggregations(func = 'makeclim', column = 'climname', kwargs = dict(climtmin = '1998-01-01', climtmax = '2018-12-31', llcrnr= (36,-24), rucrnr = (None,40)))
#podDJF.iterateaggregations(func = 'match', column = 'booksname', kwargs = {'loadkwargs' : dict(llcrnr= (36,-24), rucrnr = (None,40))})
#podDJF.iterateaggregations(func = 'score', column = 'scorefiles', kwargs = {'pp_model':Logistic(), store_minimum:True})
#podDJF.iterateaggregations(func = 'skill', column = 'scores', kwargs = {'usebootstrapped':False, 'analysiskwargs':dict(groupers = ['leadtime','clustid'])})
#
#podJJA = Experiment(expname = 'hr26podJJA', basevar = 'rr', newvar = 'pod', rolling = True, cycle = '45r1', season = 'JJA', clustername = 'tg-DJF',
#        method = 'mean', timeaggregations = ['1D'], spaceaggregations = [0], quantiles = None)
#podJJA.setuplog()
#podJJA.log.loc[:,['obsname','climname']] = podDJF.log.loc[:,['obsname','climname']]
#podJJA.iterateaggregations(func = 'match', column = 'booksname', kwargs = {'loadkwargs' : dict(llcrnr= (36,-24), rucrnr = (None,40))})
#podJJA.iterateaggregations(func = 'score', column = 'scorefiles', kwargs = {'pp_model':Logistic(), 'store_minimum':True})
#podJJA.iterateaggregations(func = 'skill', column = 'scores', kwargs = {'usebootstrapped':False, 'analysiskwargs':dict(groupers = ['leadtime','clustid'])})
