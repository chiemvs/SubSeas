# SubSeas
Predictability of European weather on sub-seasonal timescales.

## General idea
Framework to analyse observed and forcasted weather variables at varying leadtimes and varying resolutions, to fit statistical models on forecast-observation pairs and to score them. My goal for the framework is an object-oriented piece of code that is applicable to out-of-memory datasets.

## Setup
Developed in python 3 with use of both the array data format (xarray and dask) and the dataframe format (pandas and dask). On disk data-formats are GRIB, netCDF and HDF5.

## Specific implementation
Uses ECMWF extended range forecasts, initialized twice a week and available on the MARS archive. These are matched to gridded E-OBS surface observations. Spatio-temporal aggregation methods transform the raw data to gridded continuous variables at varying resolutions. These are either directly evaluated and post-processed using Non-homogeneous gaussian regression, or are transformed by an event-based classifier to a binary variables and post-processed with logistic regression.
