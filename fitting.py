#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 09:41:45 2019

@author: straaten
"""

import pandas as pd
from scipy import optimize
import properscoring as ps
import numpy as np

#dfpop = pd.read_hdf('/usr/people/straaten/Documents/python_tests/poptest.h5', key = 'intermediate')
#tx_JJA_41r1_2D_max_0.25_degrees_56299e8288fb412e93e40ca2345024b2.h5
dftx = pd.read_hdf('/nobackup/users/straaten/match/tx_JJA_41r1_7D_max_3_degrees_max_f7fb5e7325ec4554afe3685cf542e810.h5')
#testset = dftx.loc[np.logical_and(dftx['latitude'] == 25.375, dftx['longitude'] == -11.625),:] # one location all leadtimes.
testset = dftx.loc[np.logical_and(dftx['latitude'] == 26.75, dftx['longitude'] == 31.5),:]
#testset = dftx.loc[dftx['latitude'] == 51.125,:] # one location all leadtimes.
testset = testset.reset_index(drop = True)
testset.loc[:,'mu_ens'] = testset.loc[:,'forecast'].mean(axis = 1)
testset['std_ens'] = testset['forecast'].std(axis = 1)


def eval_ngr_crps(parameters, mu_ens, std_ens, obs):
    """
    parameters contains [a0,a1,b0,b1] for N(a0 + a1 * mu_ens, b0 + b1 * sig_ens)
    """
    mu = parameters[0] + parameters[1] * mu_ens
    std = parameters[2] + parameters[3] * std_ens
    return(ps.crps_gaussian(obs, mu, std).sum()) # obs can be vector.

def fit_ngr(train):
    res = optimize.minimize(eval_ngr_crps, x0 = [0,1,0,1], 
                            args=(train['mu_ens'], train['std_ens'], train['observation']), 
                            method='L-BFGS-B', bounds = [(-20,20),(0,10),(-10,10),(0,10)],
                            options = {'disp':True, 'eps':0.00001}) # Nelder-Mead
    return(res)

def eval_ngr_crps2(parameters, mu_ens, std_ens, obs):
    """
    parameters contains [a0,a1,b0,b1] for N(a0 + a1 * mu_ens, b0 + b1 * sig_ens)
    """
    mu = parameters[0] + parameters[1] * mu_ens
    std = parameters[2] + parameters[3] * std_ens
    #crps, grad = ps.crps_gaussian(obs, mu, std, grad = True)
    return(ps.crps_gaussian(obs, mu, std).mean()) # obs can be vector.
    
def fit_ngr2(train):
    res = optimize.minimize(eval_ngr_crps2, x0 = [0,1,0,1], 
                            args=(train['mu_ens'], train['std_ens'], train['observation']), 
                            method='L-BFGS-B', bounds = [(-20,20),(0,10),(-10,10),(0,10)],
                            options = {'disp':True, 'eps':0.00001}) # Nelder-Mead
    return(res)

def eval_ngr_crps3(parameters, mu_ens, std_ens, obs):
    """
    parameters contains [a0,a1,b0,b1] for N(a0 + a1 * mu_ens, b0 + b1 * sig_ens)
    """
    mu = parameters[0] + parameters[1] * mu_ens
    logstd = parameters[2] + parameters[3] * std_ens
    #crps, grad = ps.crps_gaussian(obs, mu, std, grad = True)
    return(ps.crps_gaussian(obs, mu, np.exp(logstd)).mean()) # obs can be vector.

def fit_ngr3(train):
    res = optimize.minimize(eval_ngr_crps3, x0 = [0,1,0,1], 
                            args=(train['mu_ens'], train['std_ens'], train['observation']), 
                            method='L-BFGS-B', bounds = [(-20,20),(0,10),(-10,10),(-10,10)],
                            options = {'disp':True, 'eps':0.00001}) # Nelder-Mead
    return(res)
    
#fit_ngr2(dftx.loc[dftx['leadtime'] == 1,:])

fitres = fit_ngr3(testset.loc[testset['leadtime'] == 10,])
params = fitres.x
testset['mu_cor'] = params[0] + params[1] * testset['mu_ens']
testset['std_cor'] = np.exp(params[2] + params[3] * testset['std_ens'])

testset.loc[testset['leadtime'] == 10, ['mu_ens', 'std_ens', 'mu_cor', 'std_cor', 'observation']].boxplot()




fit_ngr(testset.loc[testset['leadtime'] == 42,])
fit_ngr2(testset.loc[testset['leadtime'] == 42,:])
fit_ngr3(testset.loc[testset['leadtime'] == 42,:])

fits = testset.groupby('leadtime').apply(fit_ngr)
fits2 = testset.groupby('leadtime').apply(fit_ngr2)
fits3 = testset.groupby('leadtime').apply(fit_ngr3)
