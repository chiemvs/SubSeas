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
from observations import SurfaceObservations, Climatology, EventClassification, Clustering
from forecasts import Forecast, ModelClimatology
from comparison import ForecastToObsAlignment, Comparison, ScoreAnalysis
from fitting import NGR, Logistic
import itertools

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
        self.ndraws = 11 # Either equal to nmembers of the raw, or more for good draw estimate of equidistant climatology and NGR
    
    def setuplog(self):
        """
        Load an experiment log if it is present. Otherwise create one.
        Columns are for obsname and booksname, and possibly the highres climatologies. 
        For climname, scorefiles, external fits, bootstrap, scores the amount of columns is times the amount of quantiles.
        """
        self.logpath = self.resultsdir + self.expname + '.h5'
        try:
            self.log = pd.read_hdf(self.logpath, key = 'exp')
        except OSError:
            if self.quantiles is not None:
                self.log = pd.DataFrame(data = None, index = pd.MultiIndex.from_product([self.spaceaggregations, self.timeaggregations, self.quantiles], names = ['spaceagg','timeagg','quantile']), 
                                        columns = ['climname','modelclimname','scorefiles','externalfits','bootstrap','scores'])
                self.log = self.log.unstack(level = -1)
            else:
                self.log = pd.DataFrame(data = None, index = pd.MultiIndex.from_product([self.spaceaggregations, self.timeaggregations], names = ['spaceagg','timeagg']), 
                                        columns = pd.MultiIndex.from_product([['climname','modelclimname','scorefiles','externalfits','bootstrap','scores'],['']])) # Also two levels for compatibility
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
        alignment = ForecastToObsAlignment(season = self.season, observations=obs, cycle=self.cycle, n_members = 11, **{'expname':self.expname})
        alignment.match_and_write(newvariable = (self.newvar is not None), 
                                  newvarkwargs = newvarkwargs,
                                  loadkwargs = loadkwargs,
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
            else: # Make a 'equidistant draws' climatology.
                climatology.localclim(obs = obs, daysbefore = 5, daysafter=5, mean = False, quant = None, random = False, n_draws = self.ndraws)
        climatology.savelocalclim()

        return(climatology.name)
    
    def makemodelclim(self, spaceagg, timeagg, climtmin, climtmax, llcrnr = (25,-30), rucrnr = (75,75), quantile = ''):
        """
        Possible to make model climatologies for certain quantiles (not lead time dependend in contrast to the highresmodelclim)
        All aggregation and possble conversion. Units will not be changed in the meantime, Though quantiles of Kelvin anomalies later be compared to matched & aligned Celsius forecasts.
        """
        assert isinstance(quantile, float), 'only possible when a quantile is supplied. Did you mean to make an average highresmodelclim instead?'
        loadkwargs = dict(llcrnr = llcrnr, rucrnr = rucrnr)
        
        if self.newvar == 'anom': # If anomalies then we need to supply the highres mean array to the classification, and later on the aggregation.
            highresmodelclim = ModelClimatology(cycle=self.cycle, variable = self.basevar, 
                                                **{'name':self.log.loc[(spaceagg, timeagg),('modelclim','')]})
            highresmodelclim.local_clim()
            newvarkwargs = {'climatology':highresmodelclim}
        else:
            newvarkwargs = {}
        
        spacemethod = '-'.join([str(spaceagg), self.clustername, self.method])
        cl = Clustering(**{'name':self.clustername})
        clusterarray = cl.get_clusters_at(level = spaceagg)
        
        if timeagg != '1D':
            timemethod = '-'.join([timeagg,'roll',self.method]) if self.rolling else '-'.join([timeagg,'norm',self.method]) 
        else:
            timemethod = timeagg
        
        modelclim = ModelClimatology(cycle=self.cycle, variable = self.basevar if (self.newvar is None) else '-'.join([self.basevar, self.newvar]))
        modelclim.local_clim(tmin = climtmin, tmax = climtmax, timemethod = timemethod, spacemethod = spacemethod, 
                             mean = False, quant = quantile, clusterarray = clusterarray, loadkwargs = loadkwargs, newvarkwargs = newvarkwargs)
        modelclim.savelocalclim()
        
        return(modelclim.name)
    
    def makehighresmodelclim(self, spaceagg, timeagg, climtmin, climtmax, llcrnr = (None,None), rucrnr = (None,None)):
        """
        Only needed for later subtraction of mean from model fiels to create the anom variable (in matching or in model quantile climatology)
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
        If there are no quantiles to predict or binary variable, we force equidistant sampling (random = True led to overestimations of the crps)
        """
        alignment = ForecastToObsAlignment(season = self.season, cycle=self.cycle)
        alignment.recollect(booksname = self.log.loc[(spaceagg, timeagg),('booksname','')])
        
        climatology = Climatology(self.basevar, **{'name':self.log.loc[(spaceagg, timeagg),('climname', quantile)]})
        climatology.localclim() # loading in this case. Creation was done in the makeclim method.
        
        if not self.log.loc[(spaceagg, timeagg),('modelclimname',[quantile])].isna().any(): # Supply model quantile climatology if that was computed earlier. Will be preferred for the raw briescoring in the comparison Class
            modelclimatology = ModelClimatology(cycle = self.cycle, variable=self.basevar, **{'name':self.log.loc[(spaceagg, timeagg),('modelclimname',quantile)]})
            modelclimatology.local_clim()
            assert self.newvar == 'anom', 'This modelclimatology has likely no adapted units, only when anomalies the quantiles in Kelvin will be compatible with the aligned forecast anomalies in Celsius.'
        else:
            modelclimatology = None
        
        comp = Comparison(alignment = alignment, climatology = climatology, modelclimatology = modelclimatology)
                        
        # Fitting or accepting external fits (meaning the column is already filled):
        if not pp_model is None:
            if not isinstance(self.log.loc[(spaceagg, timeagg),('externalfits', quantile)], str):
                comp.fit_pp_models(pp_model= pp_model, groupers = ['leadtime','clustid'])
                firstfitname = comp.export(fits = True, frame = False)
                self.log.loc[(spaceagg, timeagg),('externalfits', slice(None))] = firstfitname # Specifically useful for the looping over quantiles.
            else:
                fitname = self.log.loc[(spaceagg, timeagg),('externalfits', quantile)]
                print('loading fit from:', fitname)
                comp.fits = dd.read_hdf(comp.basedir + fitname + '.h5', key = 'fits') # Loading of the fits of the first quantile.
                comp.fitgroupers = ['leadtime','clustid']
                     
        # Going to the scoring.
        if isinstance(quantile, float):
            if not pp_model is None:
                comp.make_pp_forecast(pp_model = pp_model)
            comp.brierscore()
        else:
            if not pp_model is None:
                comp.make_pp_forecast(pp_model = pp_model, random = False, n_members = self.ndraws if isinstance(pp_model, NGR) else None)
                comp.export(fits=False, frame = False, preds = True)
            if (self.newvar is None) or (self.newvar == 'anom'):
                comp.crpsscore()
            else: # Meaning a custom binary predictand
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
        bootstrap_ready = self.log.loc[(spaceagg, timeagg),('bootstrap', quantile)]
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
"""
#clustga25 = Experiment(expname = 'clustga25', basevar = 'tg', newvar = 'anom', rolling = True, cycle = '45r1', season = 'DJF', clustername = 'tg-DJF',
#                 method = 'mean', timeaggregations= ['1D','3D','5D','7D','9D','11D'], spaceaggregations=[0.025,0.05,0.1,0.2,0.3,0.5,1], quantiles = None)
#clustga25.setuplog()
#clustga25.iterateaggregations(func = 'makehighresobsclim', column = 'obsclim', kwargs = dict(climtmin = '1998-01-01', climtmax = '2018-12-31', llcrnr= (36,-24), rucrnr = (None,40)))
#clustga25.iterateaggregations(func = 'prepareobs', column = 'obsname', kwargs = dict(tmin = '1998-06-07', tmax = '2018-12-31', llcrnr= (36,-24), rucrnr = (None,40)))
#clustga25.iterateaggregations(func = 'makeclim', column = 'climname', kwargs = dict(climtmin = '1998-01-01', climtmax = '2018-12-31', llcrnr= (36,-24), rucrnr = (None,40)))
#clustga25.iterateaggregations(func = 'makehighresmodelclim', column = 'modelclim', kwargs = dict(climtmin = '1998-06-07', climtmax = '2019-05-16', llcrnr= (36,-24), rucrnr = (None,40)))
#clustga25.iterateaggregations(func = 'match', column = 'booksname', kwargs = {'loadkwargs' : dict(llcrnr= (36,-24), rucrnr = (None,40))})
#clustga25.iterateaggregations(func = 'score', column = 'scorefiles', kwargs = {'pp_model':NGR()})
#clustga25.iterateaggregations(func = 'bootstrap_scores', column = 'bootstrap', kwargs = {'bootstrapkwargs':dict(n_samples = 200, fixsize = False)})
#clustga25.iterateaggregations(func = 'skill', column = 'scores', overwrite = True, kwargs = {'usebootstrapped' :True, 'analysiskwargs':dict(local = True, fitquantiles = False, forecast_horizon = False)})

#backuplog = pd.read_hdf('/nobackup/users/straaten/results/clustga25_backup7.h5')

## Add characteristic length computations.
#clustga25.log['charlengths'] = None
#clustga25.iterateaggregations(func = 'save_charlengths', column = 'charlengths')

"""
Experiment 26 Highest resolution, regular temperature, temperature anomalies. Two seasons each. Currently mean scoring.
Above three are all matched. Only for third I started with scoring.
"""
#tgDJF = Experiment(expname = 'hr26tgDJF', basevar = 'tg', rolling = True, cycle = '45r1', season = 'DJF', clustername = 'tg-DJF',
#        method = 'mean', timeaggregations = ['1D'], spaceaggregations = [0], quantiles = None)
#tgDJF.setuplog()
#tgDJF.iterateaggregations(func = 'prepareobs', column = 'obsname', kwargs = dict(tmin = '1998-06-07', tmax = '2018-12-31', llcrnr= (36,-24), rucrnr = (None,40)))
#tgDJF.iterateaggregations(func = 'makeclim', column = 'climname', kwargs = dict(climtmin = '1998-01-01', climtmax = '2018-12-31', llcrnr= (36,-24), rucrnr = (None,40)))
#tgDJF.iterateaggregations(func = 'match', column = 'booksname', kwargs = {'loadkwargs' : dict(llcrnr= (36,-24), rucrnr = (None,40))})
#tgDJF.iterateaggregations(func = 'score', column = 'scorefiles', kwargs = {'pp_model':NGR(), 'store_minimum':True})
#tgDJF.iterateaggregations(func = 'skill', column = 'scores', kwargs = {'usebootstrapped':False, 'analysiskwargs':dict(groupers = ['leadtime','clustid'])})
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
##tgaDJF.log.loc[:,['obsclim','modelclim']] = clustga25.log.loc[:,['obsclim','modelclim']].iloc[0].tolist() # Can be copied because even in clustga25 they are at the highest res. Not sure if this indexing works.
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
##tgaJJA.log.loc[:,['obsname','climname','obsclim','modelclim']] = tgaDJF.log.loc[:,['obsname','climname','obsclim','modelclim']]
##tgaJJA.savelog()
#tgaJJA.iterateaggregations(func = 'match', column = 'booksname', kwargs = {'loadkwargs' : dict(llcrnr= (36,-24), rucrnr = (None,40))})
#tgaJJA.iterateaggregations(func = 'score', column = 'scorefiles', kwargs = {'pp_model':NGR(), 'store_minimum':True})
#tgaJJA.iterateaggregations(func = 'skill', column = 'scores', kwargs = {'usebootstrapped' :False, 'analysiskwargs':dict(groupers = ['leadtime','clustid'])})

"""
Experiment 27 Test for cluster based aggregation, dry periods rainfall, some post-processing, 1mm threshold
Uses the winter temperatures clustering.
"""
#rr27 = Experiment(expname = 'clusrrpod27', basevar = 'rr', newvar = 'pod', rolling = True, cycle = '45r1', season = 'DJF', clustername = 'tg-DJF',
#                 method = 'max', timeaggregations= ['1D','2D,'3D','5D','7D'], spaceaggregations=[0.025,0.05,0.1,0.2,0.3], quantiles = None)
#rr27.setuplog()
#rr27.iterateaggregations(func = 'prepareobs', column = 'obsname', kwargs = dict(tmin = '1998-06-07', tmax = '2018-12-31', llcrnr= (36,-24), rucrnr = (None,40)))
#rr27.iterateaggregations(func = 'makeclim', column = 'climname', kwargs = dict(climtmin = '1998-01-01', climtmax = '2018-12-31', llcrnr= (36,-24), rucrnr = (None,40)))
#rr27.iterateaggregations(func = 'match', column = 'booksname', overwrite = True, kwargs = {'loadkwargs' : dict(llcrnr= (36,-24), rucrnr = (None,40))})
#rr27.iterateaggregations(func = 'score', column = 'scorefiles', kwargs = {'pp_model':Logistic()})
#rr27.iterateaggregations(func = 'bootstrap_scores', column = 'bootstrap', kwargs = {'bootstrapkwargs':dict(n_samples = 200, fixsize = False)})
#rr27.iterateaggregations(func = 'skill', column = 'scores', overwrite = True, kwargs = {'usebootstrapped' :True, 'analysiskwargs':dict(local = True, fitquantiles = False, forecast_horizon = True)})


"""
Experiment 28 Test for cluster based aggregation, regular temperatures, some post-processing, only the larger end of spatial aggregation.
NOTE: based on a new JJA-based clustering.
"""
#clustga28 = Experiment(expname = 'clustga28', basevar = 'tg', newvar = 'anom', rolling = True, cycle = '45r1', season = 'JJA', clustername = 'tg-JJA', method = 'mean', timeaggregations= ['1D','3D','5D','7D','9D','11D'], spaceaggregations=[0.025,0.05,0.1,0.2,0.3,0.5,1], quantiles = None)
#clustga28.setuplog()
##clustga28.log.loc[:,['obsclim','modelclim']] = clustga25.log.loc[:,['obsclim','modelclim']] # Can be copied because even in clustga25 they are at the highest res. Not sure if this indexing works.
#clustga28.iterateaggregations(func = 'prepareobs', column = 'obsname', kwargs = dict(tmin = '1998-06-07', tmax = '2018-12-31', llcrnr= (36,-24), rucrnr = (None,40)))
#clustga28.iterateaggregations(func = 'makeclim', column = 'climname', kwargs = dict(climtmin = '1998-01-01', climtmax = '2018-12-31', llcrnr= (36,-24), rucrnr = (None,40)))
#clustga28.iterateaggregations(func = 'match', column = 'booksname', kwargs = {'loadkwargs' : dict(llcrnr= (36,-24), rucrnr = (None,40))})
#clustga28.iterateaggregations(func = 'score', column = 'scorefiles', kwargs = {'pp_model':NGR()})
#clustga28.iterateaggregations(func = 'bootstrap_scores', column = 'bootstrap', kwargs = {'bootstrapkwargs':dict(n_samples = 200, fixsize = False)})
#clustga28.iterateaggregations(func = 'skill', column = 'scores', overwrite = True, kwargs = {'usebootstrapped' :True, 'analysiskwargs':dict(local = True, fitquantiles = False, forecast_horizon = False)})

## Add characteristic length computations.
#clustga28.log['charlengths'] = None
#clustga28.iterateaggregations(func = 'save_charlengths', column = 'charlengths')

"""
Experiment 29. Brier score extension of experiment 25, the tgaDJF
"""
#clustga29 = Experiment(expname = 'clustga29', basevar = 'tg', newvar = 'anom', rolling = True, cycle = '45r1', season = 'DJF', clustername = 'tg-DJF',
#                 method = 'mean', timeaggregations= ['1D','3D','5D','7D','9D','11D'], spaceaggregations=[0.025,0.05,0.1,0.2,0.3,0.5,1], quantiles = [0.1, 0.15, 0.25, 0.33, 0.66, 0.75, 0.85, 0.9])
#clustga29.setuplog()
##clustga29.log.loc[:,['obsname','booksname','modelclim','obsclim']] = clustga25.log.loc[:,['obsname','booksname','modelclim','obsclim']]
##clustga29.log['externalfits'] = clustga25.log['scorefiles'].values[:,np.newaxis] # Supply the external fits.
##clustga29.savelog()
#clustga29.iterateaggregations(func = 'makeclim', column = 'climname', kwargs = dict(climtmin = '1998-01-01', climtmax = '2018-12-31', llcrnr= (36,-24), rucrnr = (None,40)))
#clustga29.iterateaggregations(func = 'score', column = 'scorefiles', kwargs = {'pp_model':NGR(),'store_minimum':True})
#clustga29.iterateaggregations(func = 'bootstrap_scores', column = 'bootstrap', kwargs = {'bootstrapkwargs':dict(n_samples = 200, fixsize = False)})
#clustga29.iterateaggregations(func = 'skill', column = 'scores', overwrite = True, kwargs = {'usebootstrapped' :True, 'analysiskwargs':dict(local = True, fitquantiles = False, forecast_horizon = False)})

"""
Experiment 30 Brier score extension of experiment 28
NOTE: based on a new JJA-based clustering.
"""
#clustga30 = Experiment(expname = 'clustga30', basevar = 'tg', newvar = 'anom', rolling = True, cycle = '45r1', season = 'JJA', clustername = 'tg-JJA', method = 'mean', timeaggregations= ['1D','3D','5D','7D','9D','11D'], spaceaggregations=[0.025,0.05,0.1,0.2,0.3,0.5,1], quantiles =[0.1, 0.15, 0.25, 0.33, 0.66, 0.75, 0.85, 0.9]) # [0.33,0.66,0.9]
#clustga30.setuplog()
##clustga30.log.loc[:,['obsname','booksname','modelclim','obsclim']] = clustga28.log.loc[:,['obsname','booksname','modelclim','obsclim']]
##clustga30.log['externalfits'] = clustga28.log['scorefiles'].values[:,np.newaxis] # Supply the external fits.
##clustga30.savelog()
#clustga30.iterateaggregations(func = 'makeclim', column = 'climname', kwargs = dict(climtmin = '1998-01-01', climtmax = '2018-12-31', llcrnr= (36,-24), rucrnr = (None,40)))
#clustga30.iterateaggregations(func = 'score', column = 'scorefiles', kwargs = {'pp_model':NGR()})
#clustga30.iterateaggregations(func = 'bootstrap_scores', column = 'bootstrap', kwargs = {'bootstrapkwargs':dict(n_samples = 200, fixsize = False)})
#clustga30.iterateaggregations(func = 'skill', column = 'scores', overwrite = True, kwargs = {'usebootstrapped' :True, 'analysiskwargs':dict(local = True, fitquantiles = False, forecast_horizon = False)})

## Merging with the remaining quantiles from experiment 32
##clustga30.log = pd.concat([clustga30.log, clustga32.log.loc[:,['climname','scorefiles','bootstrap','scores']]], axis = 1)
##clustga30.savelog()

"""
Experiment 31 Test for cluster based aggregation, dry periods rainfall, some post-processing, 1mm threshold
Uses the summer rainfall clustering.
"""
#rr31 = Experiment(expname = 'clusrrpod31', basevar = 'rr', newvar = 'pod', rolling = True, cycle = '45r1', season = 'JJA', clustername = 'rr-JJA',
#                 method = 'max', timeaggregations= ['1D','2D','3D','5D','7D'], spaceaggregations=[0.2,0.3,0.4,0.5], quantiles = None)
#rr31.setuplog()
#rr31.iterateaggregations(func = 'prepareobs', column = 'obsname', kwargs = dict(tmin = '1998-06-07', tmax = '2018-12-31', llcrnr= (36,-24), rucrnr = (None,40)))
#rr31.iterateaggregations(func = 'makeclim', column = 'climname', kwargs = dict(climtmin = '1998-01-01', climtmax = '2018-12-31', llcrnr= (36,-24), rucrnr = (None,40)))
#rr31.iterateaggregations(func = 'match', column = 'booksname', overwrite = True, kwargs = {'loadkwargs' : dict(llcrnr= (36,-24), rucrnr = (None,40))})
#rr31.iterateaggregations(func = 'score', column = 'scorefiles', kwargs = {'pp_model':Logistic()})
#rr31.iterateaggregations(func = 'bootstrap_scores', column = 'bootstrap', kwargs = {'bootstrapkwargs':dict(n_samples = 200, fixsize = False)})
#rr31.iterateaggregations(func = 'skill', column = 'scores', overwrite = True, kwargs = {'usebootstrapped' :True, 'analysiskwargs':dict(local = True, fitquantiles = False, forecast_horizon = False)})

"""
Experiment 32 Brier score extension of experiment 28, for the quantiles that have not been computed in experiment 30
"""
#clustga32 = Experiment(expname = 'clustga32', basevar = 'tg', newvar = 'anom', rolling = True, cycle = '45r1', season = 'JJA', clustername = 'tg-JJA', method = 'mean', timeaggregations= ['1D','3D','5D','7D','9D','11D'], spaceaggregations=[0.025,0.05,0.1,0.2,0.3,0.5,1], quantiles = [0.1, 0.15, 0.25, 0.75, 0.85])
#clustga32.setuplog()
##clustga32.log.loc[:,['obsname','booksname','modelclim','obsclim']] = clustga28.log.loc[:,['obsname','booksname','modelclim','obsclim']]
##clustga32.log['externalfits'] = clustga28.log['scorefiles'].values[:,np.newaxis] # Supply the external fits.
##clustga32.savelog()
#clustga32.iterateaggregations(func = 'makeclim', column = 'climname', kwargs = dict(climtmin = '1998-01-01', climtmax = '2018-12-31', llcrnr= (36,-24), rucrnr = (None,40)))
#clustga32.iterateaggregations(func = 'score', column = 'scorefiles', kwargs = {'pp_model':NGR()})
#clustga32.iterateaggregations(func = 'bootstrap_scores', column = 'bootstrap', kwargs = {'bootstrapkwargs':dict(n_samples = 200, fixsize = False)})
#clustga32.iterateaggregations(func = 'skill', column = 'scores', overwrite = True, kwargs = {'usebootstrapped' :True, 'analysiskwargs':dict(local = True, fitquantiles = False, forecast_horizon = True)})

"""
Experiment 33 More sample version of experiment 25. Uses the same matched files, with renamed books. Requires new climatologies with more samples. Creates new scorefiles, but uses the fitted models from experiment 25.
"""
#clustga33 = Experiment(expname = 'clustga33', basevar = 'tg', newvar = 'anom', rolling = True, cycle = '45r1', season = 'DJF', clustername = 'tg-DJF', method = 'mean', timeaggregations= ['1D','3D','5D','7D','9D','11D'], spaceaggregations=[0.025,0.05,0.1,0.2,0.3,0.5,1], quantiles = None)
#clustga33.ndraws = 100
#clustga33.setuplog()
##clustga33.log.loc[:,['obsname','modelclim','obsclim']] = clustga25.log.loc[:,['obsname','modelclim','obsclim']]
##clustga33.log['externalfits'] = clustga25.log['scorefiles'] # Supply the external fits.
##clustga33.savelog()
#clustga33.iterateaggregations(func = 'makeclim', column = 'climname', kwargs = dict(climtmin = '1998-01-01', climtmax = '2018-12-31', llcrnr= (36,-24), rucrnr = (None,40)))
#clustga33.iterateaggregations(func = 'score', column = 'scorefiles', kwargs = {'pp_model':NGR(),'store_minimum':True})
#clustga33.iterateaggregations(func = 'bootstrap_scores', column = 'bootstrap', kwargs = {'bootstrapkwargs':dict(n_samples = 200, fixsize = False)})
#clustga33.iterateaggregations(func = 'skill', column = 'scores', overwrite = True, kwargs = {'usebootstrapped' :True, 'analysiskwargs':dict(local = True, fitquantiles = False, forecast_horizon = True)})

"""
Experiment 34 More sample version of experiment 28. Uses the same matched files, with renamed books. Requires new climatologies with more samples. Creates new scorefiles, but uses the fitted models from experiment 28.
"""
#clustga34 = Experiment(expname = 'clustga34', basevar = 'tg', newvar = 'anom', rolling = True, cycle = '45r1', season = 'JJA', clustername = 'tg-JJA', method = 'mean', timeaggregations= ['1D','3D','5D','7D','9D','11D'], spaceaggregations=[0.025,0.05,0.1,0.2,0.3,0.5,1], quantiles = None)
#clustga34.ndraws = 100
#clustga34.setuplog()
##clustga34.log.loc[:,['obsname','modelclim','obsclim']] = clustga28.log.loc[:,['obsname','modelclim','obsclim']]
##clustga34.log['externalfits'] = clustga28.log['scorefiles'] # Supply the external fits.
##clustga34.savelog()
#clustga34.iterateaggregations(func = 'makeclim', column = 'climname', kwargs = dict(climtmin = '1998-01-01', climtmax = '2018-12-31', llcrnr= (36,-24), rucrnr = (None,40)))
#clustga34.iterateaggregations(func = 'score', column = 'scorefiles', kwargs = {'pp_model':NGR(),'store_minimum':True})
#clustga34.iterateaggregations(func = 'bootstrap_scores', column = 'bootstrap', kwargs = {'bootstrapkwargs':dict(n_samples = 200, fixsize = False)})
#clustga34.iterateaggregations(func = 'skill', column = 'scores', overwrite = True, kwargs = {'usebootstrapped' :True, 'analysiskwargs':dict(local = True, fitquantiles = False, forecast_horizon = True)})

"""
Experiment 35 Subset of experiment 29 but then with the reviewers suggestion 
to use model climatology quantiles as threshold for brier scoring the raw forecasts
Uses the same matched files, with renamed books
"""
#clustga35 = Experiment(expname = 'clustga35', basevar = 'tg', newvar = 'anom', rolling = True, cycle = '45r1', season = 'DJF', clustername = 'tg-DJF',
#                 method = 'mean', timeaggregations= ['1D','9D'], spaceaggregations=[0.025], quantiles = [0.15, 0.33, 0.66, 0.85])
#clustga35.setuplog()
##clustga35.log.loc[:,['obsname','obsclim']] = clustga29.log.loc[(clustga35.spaceaggregations,clustga35.timeaggregations),['obsname','obsclim']]
##clustga35.log.loc[:,['externalfits','climname']] = clustga29.log.loc[(clustga35.spaceaggregations,clustga35.timeaggregations), (['externalfits','climname'], clustga35.quantiles)] # Supply the external fits and the climatologies
##clustga35.log['modelclim'] = 'tg_45r1_1998-06-07_2019-05-16_1D_0.38-degrees_5_5_mean'
##modelclimfiles = [f[:-3] for f in os.listdir('/nobackup/users/straaten/modelclimatology/') if (f.startswith('tg-anom') and 'DJF' in f)]
##modelclimfiles.sort()
##clustga35.log.loc[(0.025, ['1D','9D']),('modelclimname',slice(None))] = np.array(modelclimfiles).reshape((2,4))
##clustga35.log['booksname'] = ['books_clustga35_tg-anom_DJF_45r1_1D_0.025-tg-DJF-mean.csv', 'books_clustga35_tg-anom_DJF_45r1_9D-roll-mean_0.025-tg-DJF-mean.csv']
##clustga35.savelog()

#clustga35.iterateaggregations(func = 'score', column = 'scorefiles', kwargs = {'pp_model':NGR(),'store_minimum':True})
#clustga35.iterateaggregations(func = 'bootstrap_scores', column = 'bootstrap', kwargs = {'bootstrapkwargs':dict(n_samples = 200, fixsize = False)})
#clustga35.iterateaggregations(func = 'skill', column = 'scores', overwrite = True, kwargs = {'usebootstrapped' :True, 'analysiskwargs':dict(local = True, fitquantiles = False, forecast_horizon = False)})

"""
Experiment 36 Subset of experiment 30 but then with the reviewers suggestion 
to use model climatology quantiles as threshold for brier scoring the raw forecasts
Uses the same matched files, with renamed books. Computations were done with season = 'DJF' and clustername = 'tg-DJF', but this did not lead to mistakes.
"""
#clustga36 = Experiment(expname = 'clustga36', basevar = 'tg', newvar = 'anom', rolling = True, cycle = '45r1', season = 'JJA', clustername = 'tg-JJA',
#                 method = 'mean', timeaggregations= ['1D','9D'], spaceaggregations=[0.025], quantiles = [0.15, 0.33, 0.66, 0.85])
#clustga36.setuplog()
##clustga36.log.loc[:,['obsname','obsclim']] = clustga30.log.loc[(clustga36.spaceaggregations,clustga36.timeaggregations),['obsname','obsclim']]
##clustga36.log.loc[:,['climname']] = clustga30.log.loc[(clustga36.spaceaggregations,clustga36.timeaggregations), (['climname'], clustga36.quantiles)] # Supply the climatologies
##extvals = clustga30.log.loc[(clustga36.spaceaggregations,clustga36.timeaggregations),('externalfits',0.9)].values # External fit detour because the experiment 30 log does not have all the quantiles.
##clustga36.log.loc[:,['externalfits']] = np.repeat(extvals[:,np.newaxis], 4, axis = 1)
##clustga36.log['modelclim'] = 'tg_45r1_1998-06-07_2019-05-16_1D_0.38-degrees_5_5_mean'
##modelclimfiles = [f[:-3] for f in os.listdir('/nobackup/users/straaten/modelclimatology/') if (f.startswith('tg-anom') and 'JJA' in f)]
##modelclimfiles.sort()
##clustga36.log.loc[(0.025, ['1D','9D']),('modelclimname',slice(None))] = np.array(modelclimfiles).reshape((2,4))
##clustga36.log['booksname'] = ['books_clustga36_tg-anom_JJA_45r1_1D_0.025-tg-JJA-mean.csv', 'books_clustga36_tg-anom_JJA_45r1_9D-roll-mean_0.025-tg-JJA-mean.csv']
##clustga36.savelog()

#clustga36.iterateaggregations(func = 'score', column = 'scorefiles', kwargs = {'pp_model':NGR(),'store_minimum':True})
#clustga36.iterateaggregations(func = 'bootstrap_scores', column = 'bootstrap', kwargs = {'bootstrapkwargs':dict(n_samples = 200, fixsize = False)})
#clustga36.iterateaggregations(func = 'skill', column = 'scores', overwrite = True, kwargs = {'usebootstrapped' :True, 'analysiskwargs':dict(local = True, fitquantiles = False, forecast_horizon = False)})
