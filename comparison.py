#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 09:31:16 2018

@author: straaten
"""

class ForecastObsAlignment(object):
    
    def __init__(self, leadtime, temporal_ext, variable):
        """
        Temporal extend is the season under inspection. Also here groups should be defined.
        Here the forecast files corresponding to the observations are determined
        """
        
    def force_resolution(self):
        """
        Check if the same resolution and force spatial/temporal aggregation if that is not the case.
        """
        
    def load_and_match(self, n_members):
        """
        Neirest neighbouring to match pairs.
        Creates the dataset. Possibly writes to disk too.
        """

        
class Comparison(object):
    
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
    
        