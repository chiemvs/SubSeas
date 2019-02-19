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



"""
Temperature
"""
#'/nobackup/users/straaten/match/tx_JJA_41r1_7D_max_3_degrees_max_f7fb5e7325ec4554afe3685cf542e810.h5'
#tx_JJA_41r1_2D_max_0.25_degrees_56299e8288fb412e93e40ca2345024b2.h5
#dftx = pd.read_hdf('/nobackup/users/straaten/match/tx_JJA_41r1_2D_max_0.25_degrees_56299e8288fb412e93e40ca2345024b2.h5')
#testset = dftx.loc[np.logical_and(dftx['latitude'] == 25.375, dftx['longitude'] == -11.625),:] # one location all leadtimes.
#testset = dftx.loc[np.logical_and(dftx['latitude'] == 26.75, dftx['longitude'] == 31.5),:]
#testset = dftx.loc[dftx['latitude'] == 51.125,:] # one location all leadtimes.
#testset = testset.reset_index(drop = True)
#testset.loc[:,'mu_ens'] = testset.loc[:,'forecast'].mean(axis = 1)
#testset['std_ens'] = testset['forecast'].std(axis = 1)


def eval_ngr_crps3(parameters, mu_ens, std_ens, obs):
    """
    Cost function for CRPS based minimization of an NGR model.
    Model contains [a0,a1,b0,b1] for N(a0 + a1 * mu_ens, exp(b0 + b1 * sig_ens))
    So a logarithmic transformation on sigma to keep those positive.
    To be independent of the amount of observations, the mean (and not sum) of CRPS is minimized
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

def eval_ngr_crps4(parameters, mu_ens, std_ens, obs):
    """
    Cost function for CRPS based minimization of an NGR model.
    Model contains [a0,a1,b0,b1] for N(a0 + a1 * mu_ens, exp(b0 + b1 * sig_ens))
    So a logarithmic transformation on sigma to keep those positive.
    To be independent of the amount of observations, the mean (and not sum) of CRPS is minimized
    
    Also returns the (analytical??) gradient, translated from sigma and mu to the model parameters,
    for better optimization (need not be approximated)
    """
    mu = parameters[0] + parameters[1] * mu_ens
    logstd = parameters[2] + parameters[3] * std_ens
    std = np.exp(logstd)
    crps, grad = ps.crps_gaussian(obs, mu, std, grad = True) # grad is returned as np.array([dmu, dsig])
    dcrps_d0 = grad[0,:]
    dcrps_d1 = grad[0,:] * mu_ens
    dcrps_d2 = grad[1,:] * std
    dcrps_d3 = grad[1,:] * std * std_ens
    return(crps.mean(), np.array([dcrps_d0.mean(), dcrps_d1.mean(), dcrps_d2.mean(), dcrps_d3.mean()])) # obs can be vector.

def fit_ngr4(train):
    res = optimize.minimize(eval_ngr_crps4, x0 = [0,1,0.5,0.2], jac = True,
                            args=(train['mu_ens'], train['std_ens'], train['observation']), 
                            method='L-BFGS-B', bounds = [(-20,20),(0,10),(-10,10),(-10,10)],
                            options = {'disp':True}) 
    return(res)

    

#fitres = fit_ngr3(testset.loc[testset['leadtime'] == 1,])
#params = fitres.x
#testset['mu_cor'] = params[0] + params[1] * testset['mu_ens']
#testset['std_cor'] = np.exp(params[2] + params[3] * testset['std_ens'])

#testset.loc[testset['leadtime'] == 10, ['mu_ens', 'std_ens', 'mu_cor', 'std_cor', 'observation']].boxplot()

#fits3 = testset.groupby('leadtime').apply(lambda dat: fit_ngr3(dat).x)
#fits4 = testset.groupby('leadtime').apply(lambda dat: fit_ngr4(dat).x)


"""
Probability of precipitation.
"""
dfpop = pd.read_hdf('/nobackup/users/straaten/match/pop_DJF_41r1_1D_0.25_degrees_66339b514c834afc9efb08bd93fcdf3a.h5', key = 'intermediate')
testset = dfpop.loc[dfpop['latitude'] == 52.375,:]

from sklearn.linear_model import LogisticRegression
#X = pd.DataFrame({'pi':testset['pi'], 'ens_mean':testset['forecast'].mean(axis = 1)})
#X.reset_index(drop = True, inplace = True)
#y = testset['observation']
#clf = LogisticRegression(solver='liblinear')
#clf.fit(X = X, y = y)
#clf.predict_proba(X) # Gives an (n,2) array. With first column the probability of non event and the second of event.

def fitlogistic_retcoef(train):
    X = pd.DataFrame({'pi':train['pi'], 'ens_mean':train['forecast'].mean(axis = 1)})
    y = train['observation']
    clf = LogisticRegression(solver='liblinear')
    clf.fit(X = X, y = y)
    return(clf.intercept_, clf.coef_)

def fitlogistic_predict(train):
    X = pd.DataFrame({'pi':train['pi'], 'ens_mean':train['forecast'].mean(axis = 1)})
    y = train['observation']
    clf = LogisticRegression(solver='liblinear')
    clf.fit(X = X, y = y)
    result = pd.DataFrame(data = {'pi_cor': clf.predict_proba(X)[:,1]}, index = pd.MultiIndex.from_arrays([train['latitude'], train['longitude']]))
    return(result)
    
fits = dfpop.groupby('leadtime').apply(fitlogistic_predict)
test = testset.groupby('leadtime').apply(fitlogistic_predict)
