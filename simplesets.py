import numpy as np
import xarray as xr
import pandas as pd

from pathlib import Path

from forecasts import Forecast, ModelClimatology
from observations import EventClassification
"""
Get daily forecasts at a specific lead time. 

Make them into anomalies with the model climatologies (trend is dealt with in statistical modeling)

Perhaps to this only upon processing. Perhaps not so useful to have a whole extra directory with anomalies

What is time aggregation of predictands?

Certainly no regime information on mean Z300. Then better work with frequencies of daily regimes within some period 
"""

# Creation of anomalies for all leadtimes
# Model drift in the mean is accounted for by constructing anomalies this way

varib_unit = {'swvl13':'m**3 m**-3', 'z':'m**2 s**-2'}

def make_anomalies(f: Forecast, var: str, highresmodelclim: ModelClimatology, loadkwargs: dict = {}):
    """ Inplace transformation to anomalies """
    f.load(var, **loadkwargs)
    f.array.attrs.update({'units':varib_unit[var]}) # Avoid the current absence of units
    if not hasattr(highresmodelclim, 'clim'):
        highresmodelclim.local_clim()
        highresmodelclim.clim.attrs.update({'units':varib_unit[var]}) # Avoid the current absence of units
    method = getattr(EventClassification(obs = f, **{'climatology':highresmodelclim}), 'anom')
    method(inplace = True)

def lead_time_1_z300() -> xr.DataArray:
    """
    attempt to get the daily 'analysis' dataset (time,space) of Z300 anomalies. For later regime computations
    Truest to the analysis in this data is leadtime 1 day (12 UTC), control member
    """
    variable = 'z'
    forbasedir = '/nobackup/users/straaten/EXT_extra/'
    cycle = '45r1'
    forecast_paths = (Path(forbasedir) / cycle).glob('*_processed.nc')  # Both for- and hindcasts
    listofforecasts = [Forecast(indate = path.name.split('_')[1], prefix = path.name[:4], cycle = cycle, basedir = forbasedir) for path in forecast_paths]

    # Reference model climatology
    modelclim = ModelClimatology(cycle='45r1', variable = variable, **{'name':f'{variable}_45r1_1998-06-07_2018-08-31_1D_1.5-degrees_5_5_mean'}) # Name for loading

    # Pre-allocate an array for anomalies
    example = listofforecasts[0]
    example.load(variable)
    lats = example.array.coords['latitude']
    lons = example.array.coords['longitude']
    timestamps = pd.Index([pd.Timestamp(f.indate) for f in listofforecasts])
    data = np.full((len(timestamps), len(lats), len(lons)), np.nan, dtype = np.float32)
    almost_analysis = xr.DataArray(data, name = f'{variable}-anom', dims = ('time','latitude','longitude'), coords = {'time':timestamps,'latitude':lats,'longitude':lons}, attrs = {'units':varib_unit[variable]})

    # Extract!
    for forc in listofforecasts:
        make_anomalies(f = forc, var = variable, highresmodelclim = modelclim, loadkwargs = {'n_members':1, 'tmax':forc.indate}) # To limit the amount of data loaded into memory
        almost_analysis.loc[forc.indate,:,:] = forc.array.sel(time = forc.indate, number = 0) # number 0 is the control member
    return almost_analysis
        

if __name__ == '__main__':
    arr = lead_time_1_z300()
