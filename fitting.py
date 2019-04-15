#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 09:41:45 2019

@author: straaten
"""

from scipy import optimize
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
import properscoring as ps
import numpy as np
import pandas as pd


class NGR(object):
    """
    Parametric model of a non-homogeneous gaussian regression
    is specified as N(a0 + a1 * mu_ens, exp(b0 + b1 * sig_ens))
    So a logarithmic transformation on sigma to keep those positive.
    """
    def __init__(self, predcols = ['ensmean','ensstd'], obscol = 'observation'):
        """
        Requires per parameter the names of the 
        associated predictor column in the supplied dataframes
        """
        self.model_coefs = ['a0','a1','b0','b1']
        self.predcols = predcols
        self.obscol = obscol
        self.need_std = True
    
    def crpscostfunc(self, parameters, mu_ens, std_ens, obs):
        """
        Cost function returns the mean crps (to be independent from the amount of observations) and also the analytical gradient, 
        translated from sigma and mu to the model parameters,
        for better optimization (need not be approximated)
        """
        mu = parameters[0] + parameters[1] * mu_ens
        std = np.exp(parameters[2] + parameters[3] * std_ens)
        crps, grad = ps.crps_gaussian(obs, mu, std, grad = True) # grad is returned as np.array([dmu, dsig])
        dcrps_d0 = grad[0,:]
        dcrps_d1 = grad[0,:] * mu_ens
        dcrps_d2 = grad[1,:] * std
        dcrps_d3 = dcrps_d2 * std_ens
        return(crps.mean(), np.array([dcrps_d0.mean(), dcrps_d1.mean(), dcrps_d2.mean(), dcrps_d3.mean()]))
    
    def fit(self, train):
        """
        Uses CRPS-minimization for the fitting to the train dataframe.
        Returns an array with 4 model coefs. Internal conversion to float64 for more precise optimization.
        """
        res = optimize.minimize(self.crpscostfunc, x0 = [0,1,0.5,0.2], jac = True,
                        args=(train[self.predcols[0]].values.astype('float64'),
                              train[self.predcols[1]].values.astype('float64'), 
                              train[self.obscol].values.astype('float64')), 
                        method='L-BFGS-B', bounds = [(-40,40),(0,10),(-10,10),(-10,10)])
                         
        return(res.x)
    
    def predict(self, test, quant_col = 'climatology', parameters = None):
        """
        Test dataframe should contain columns with the model_coefs parameters. Otherwise a (4,) array should be supplied
        Predicts climatological quantile exceedence
        """
        try:
            mu_cor = test['a0'] + test['a1'] * test[self.predcols[0]]
            std_cor = np.exp(test['b0'] + test['b1'] * test[self.predcols[1]])
        except KeyError:
            mu_cor = parameters[0] + parameters[1] * test[self.predcols[0]]
            std_cor = np.exp(parameters[2] + parameters[3] * test[self.predcols[1]])
        return(norm.sf(x = test[quant_col], loc = mu_cor, scale = std_cor).astype('float32'))

class NGR2(object):
    """
    Parametric model of a non-homogeneous gaussian regression
    is specified as N(a0 + a1 * mu_ens, exp(b0 + b1 * ln(sig_ens)))
    So a logarithmic transformation on sigma to keep those positive.
    Performs worse than the version with no extra transform on sig_ens
    """
    def __init__(self, predcols = ['ensmean','ensstd'], obscol = 'observation'):
        """
        Requires per parameter the names of the 
        associated predictor column in the supplied dataframes
        """
        self.model_coefs = ['a0','a1','b0','b1']
        self.predcols = predcols
        self.obscol = obscol
        self.need_std = True
    
    def crpscostfunc(self, parameters, mu_ens, std_ens, obs):
        mu = parameters[0] + parameters[1] * mu_ens
        std = np.exp(parameters[2]) * std_ens**parameters[3]
        crps, grad = ps.crps_gaussian(obs, mu, std, grad = True) # grad is returned as np.array([dmu, dsig])
        dcrps_d0 = grad[0,:]
        dcrps_d1 = grad[0,:] * mu_ens
        dcrps_d2 = grad[1,:] * std
        dcrps_d3 = dcrps_d2 * np.log(std_ens)
        return(crps.mean(), np.array([dcrps_d0.mean(), dcrps_d1.mean(), dcrps_d2.mean(), dcrps_d3.mean()]))

    def fit(self, train):
        """
        Uses CRPS-minimization for the fitting to the train dataframe.
        Returns an array with 4 model coefs. Internal conversion to float64 for more precise optimization.
        """
        res = optimize.minimize(self.crpscostfunc, x0 = [0,1,0.5,1], jac = True,
                        args=(train[self.predcols[0]].values.astype('float64'),
                              train[self.predcols[1]].values.astype('float64'), 
                              train[self.obscol].values.astype('float64')), 
                        method='L-BFGS-B', bounds = [(-40,40),(0,10),(-10,10),(-10,10)])
                         
        return(res.x)

    def predict(self, test, quant_col = 'climatology', n_draws = None, parameters = None):
        """
        Test dataframe should contain columns with the model_coefs parameters. Otherwise a (4,) array should be supplied
        Predicts climatological quantile exceedence. Or a random draw from the normal distribution if number of members is supplied.
        """
        try:
            mu_cor = test['a0'] + test['a1'] * test[self.predcols[0]]
            std_cor = np.exp(test['b0']) * test[self.predcols[1]]**test['b1']
        except KeyError:
            mu_cor = parameters[0] + parameters[1] * test[self.predcols[0]]
            std_cor = np.exp(parameters[2]) * test[self.predcols[1]]**parameters[3]
        if n_draws is not None:
            if n_draws == 1:
                return(pd.Series(norm.rvs(loc=mu_cor, scale=std_cor), dtype = 'float32'))
            else:
                return(pd.DataFrame(norm.rvs(loc=mu_cor, scale=std_cor, size=(n_draws, len(mu_cor))).T, dtype = 'float32'))
                #return(norm.rvs(loc=mu_cor, scale=std_cor, size=(n_draws, len(mu_cor))).T.astype('float32')) # Returning an array.            
        else:
            return(norm.sf(x = test[quant_col], loc = mu_cor, scale = std_cor).astype('float32')) # Returning a scalar vector.
   
    
class Logistic(object):
    """
    Parametric model for a logistic regression
    ln(p/(1-p)) = a0 + a1 * raw_pi + a2 * mu_ens
    """
    def __init__(self, predcols = [None,'pi','ensmean'], obscol = 'observation'):
        """
        Requires per parameter the names of the 
        associated predictor column in the supplied dataframes
        """
        self.model_coefs = ['a0','a1','a2']
        self.predcols = predcols
        self.obscol = obscol
        self.need_std = False
    
    def fit(self, train):
        """
        Uses L2-loss minimization to fit a logistic model
        """
        clf = LogisticRegression(solver='liblinear')
        clf.fit(X = train[self.predcols[1:]], y = train[self.obscol])
            
        return(np.concatenate([clf.intercept_, clf.coef_.squeeze()]))
        
    def predict(self,test, parameters = None):
        """
        Test dataframe should contain columns with the model_coefs parameters. Otherwise a (3,) array should be supplied
        p = exp(a0 + a1 * raw_pop + a2 * mu_ens) / (1 + exp(a0 + a1 * raw_pop + a2 * mu_ens))
        """
        try:
            exp_part = np.exp(test['a0'] + test['a1'] * test[self.predcols[1]] + test['a2'] * test[self.predcols[2]])
        except KeyError:
            exp_part = np.exp(parameters[0] + parameters[1] * test[self.predcols[1]] + parameters[2] * test[self.predcols[2]])
        
        return(exp_part/ (1 + exp_part))

