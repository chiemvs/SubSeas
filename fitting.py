#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 09:41:45 2019

@author: straaten
"""

from scipy import optimize
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
import statsmodels.formula.api as smf
import properscoring as ps
import numpy as np
import pandas as pd


class NGR(object):
    """
    Parametric model of a non-homogeneous gaussian regression.
    can be specified as N(a0 + a1 * mu_ens, exp(b0 + b1 * sig_ens))
    or as N(a0 + a1 * mu_ens, exp(b0 + b1 * ln(sig_ens))), with a second transform on ensemble sigma
    But always the logarithmic transformation on the model sigma to keep those positive.
    """
    def __init__(self, predcols = ['ensmean','ensstd'], obscol = 'observation', double_transform = True):
        """
        Requires per parameter the names of the 
        associated predictor column in the supplied dataframes
        """
        self.model_coefs = ['a0','a1','b0','b1']
        self.predcols = predcols
        self.obscol = obscol
        self.need_std = True
        if double_transform:
            self.stdfunc = lambda x : x  # Used for the specification of S_model = e^b0 * stdfunc(S_ens)**b1
            self.gradfunc = lambda x : np.log(x)
            self.startpoint = [0,1,0.5,1]
        else:
            self.stdfunc = lambda x : np.exp(x)  # Used for the specification of S_model = e^b0 * stdfunc(S_ens)**b1
            self.gradfunc = lambda x : x
            self.startpoint = [0,1,0.5,0.2]
            
    def crpscostfunc(self, parameters, mu_ens, std_ens, obs):
        """
        Evaluates the analytical gaussian crps for vectorized input of training samples and 4 parameters
        Also analytical calculation of the gradient to each parameter, which is also returned.
        """
        mu = parameters[0] + parameters[1] * mu_ens
        std = np.exp(parameters[2]) * self.stdfunc(std_ens)**parameters[3]
        crps, grad = ps.crps_gaussian(obs, mu, std, grad = True) # grad is returned as np.array([dmu, dsig])
        dcrps_d0 = grad[0,:]
        dcrps_d1 = grad[0,:] * mu_ens
        dcrps_d2 = grad[1,:] * std
        dcrps_d3 = dcrps_d2 * self.gradfunc(std_ens)
        return(crps.mean(), np.array([dcrps_d0.mean(), dcrps_d1.mean(), dcrps_d2.mean(), dcrps_d3.mean()]))

    def fit(self, train):
        """
        Uses CRPS-minimization for the fitting to the train dataframe.
        Returns an array with 4 model coefs. Internal conversion to float64 for more precise optimization.
        """
        res = optimize.minimize(self.crpscostfunc, x0 = self.startpoint, jac = True,
                        args=(train[self.predcols[0]].values.astype('float64'),
                              train[self.predcols[1]].values.astype('float64'), 
                              train[self.obscol].values.astype('float64')), 
                        method='L-BFGS-B', bounds = [(-40,40),(0,10),(-10,10),(-10,10)])
                         
        return(res.x)

    def predict(self, test, quant_col = 'climatology', n_draws = None, random = False, q_equidistant = None, parameters = None):
        """
        Test dataframe should contain columns with the model_coefs parameters. Otherwise a (4,) array should be supplied
        Predicts climatological quantile exceedence. Or random/equidistant draws from the normal distribution if number of members is supplied.
        If only one (non-random) member is wanted, q_equidistant should be supplied, range [0,1].
        """
        try:
            mu_cor = test['a0'] + test['a1'] * test[self.predcols[0]]
            std_cor = np.exp(test['b0']) * self.stdfunc(test[self.predcols[1]])**test['b1']
        except KeyError:
            mu_cor = parameters[0] + parameters[1] * test[self.predcols[0]]
            std_cor = np.exp(parameters[2]) * self.stdfunc(test[self.predcols[1]])**parameters[3]
        if n_draws is not None:
            if n_draws == 1:
                if random:
                    return(pd.Series(norm.rvs(loc=mu_cor, scale=std_cor), dtype = 'float32'))
                else:
                    return(pd.Series(norm.ppf(q = q_equidistant, loc = mu_cor, scale = std_cor), dtype = 'float32'))
            else:
                if random:
                    return(pd.DataFrame(norm.rvs(loc=mu_cor, scale=std_cor, size=(n_draws, len(mu_cor))).T, dtype = 'float32'))
                else:
                    return(pd.DataFrame(norm.ppf(q = np.linspace(start=1/(n_draws+1), stop = 1, num = n_draws, endpoint = False)[np.newaxis,:], loc = mu_cor[:,np.newaxis], scale = std_cor[:,np.newaxis]), dtype = 'float32'))
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
        Fitting with liblinear fails if training set has only one category (usually happens when always dry, but there are some cases of always wet).
        If it fails we check if there is indeed one class in the observations. And then return the intercept values that will approximate it with a high precision
        """
        try:
            clf = LogisticRegression(solver='liblinear')
            clf.fit(X = train[self.predcols[1:]], y = train[self.obscol])           
            return(np.concatenate([clf.intercept_, clf.coef_.squeeze()]))
        except ValueError:
            classes = np.unique(train[self.obscol])
            if len(classes) == 1:
                if classes[0] == 0.:
                    return(np.array([-20,0,0], dtype = np.float32))
                elif classes[0] == 1.:
                    return(np.array([20,0,0], dtype = np.float32))
                else:
                    raise ValueError('Logistic fitting failed on single class but it is not 1 or 0')
            else:
                raise ValueError('Logistic fitting failed even though 2 classes are present')
        
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


class ExponentialQuantile(object):
    """
    Parametric model for y = a0 + a1 * ln(x)
    After fitting stores the parameters internally for each desired quantile
    Prediction for a certain quantile in y uses these stored parameters
    """
    def __init__(self, predcol = 'leadtime', obscol = 'forecast_crpss'):
        """
        Possibility to chose the x and the y columns
        """
        self.model_coefs = ['a0','a1']
        self.predcol = predcol
        self.obscol = obscol
    
    def fit(self, train, quantiles = [0.025, 0.975], startx = None, endx = None):
        """
        Uses the statsmodel implementation of quantile regression. Quantile weighted least squares.
        Possibility to only fit the exponential decay beyond a certain leadtime
        Works on dataframe with resetted index. (i.e. leadtime as a column)
        """
        train = train.reset_index('leadtime')
        if startx is not None:
            train = train.loc[train[self.predcol] >= startx,:]
        if endx is not None:
            train = train.loc[train[self.predcol] <= endx,:]
        
        mod = smf.quantreg(self.obscol + ' ~ np.log(' + self.predcol + ')', train)
        self.fits = pd.DataFrame(np.zeros((len(quantiles),2)), index = quantiles, columns = self.model_coefs)
        for q in quantiles:
            res = mod.fit(q=q)
            self.fits.loc[q,:] = res.params.values
    
    def predict(self, test, quantiles = [0.025, 0.975], startx = None, endx = None, restoreindex = True):
        """
        Uses the coefficients that were stored as attributes in fit method.
        Needs a dataframe with resetted index (i.e. leadtime as a column)
        """
        test = test.reset_index('leadtime')
        if startx is not None:
            test = test.loc[test[self.predcol] >= startx,:]
        if endx is not None:
            test = test.loc[test[self.predcol] <= endx,:]
        mod = smf.quantreg(self.obscol + ' ~ np.log(' + self.predcol + ')', test)
        predictions = [None] * len(quantiles)
        for q in quantiles:
            # Predict function of mod produces an array. Need to reconstruct the index.
            if restoreindex:
                prediction = pd.DataFrame({self.obscol:mod.predict(self.fits.loc[q,:]),self.predcol:test[self.predcol]}, index = test.index)
                prediction.set_index('leadtime', append = True, inplace = True) # Prepend leadtime as an index
            else:
                prediction = pd.DataFrame({self.obscol:mod.predict(self.fits.loc[q,:])}, index = test[self.predcol])
            
            predictions[quantiles.index(q)] = prediction.drop_duplicates() # and remove duplicates
        return(pd.concat(objs = predictions, keys = quantiles, names = ['quantile'] + prediction.index.names))
