import numpy as np
import xarray as xr

from pathlib import Path

from forecasts import for_netcdf_encoding, ModelClimatology, Forecast
from observations import SurfaceObservations, Clustering
from comparison import ForecastToObsAlignment

def merge_soilm(directory: Path, variables: list = ['swvl1','swvl2','swvl3'], weights: list = [7,21,72], newvar: str = 'swvl13'):
    """
    Iterates through for(hind)casts present in the directory
    Computes a weighted average of the listed variables
    Writes a new variables to the file
    """
    paths = directory.glob('*_processed.nc') # Both prefix for_ and hin_
    weights = xr.DataArray(weights, dims = ('dummy',)) # Preparation of weighted mean outside loop
    for path in paths: 
        print(path)
        ds = xr.open_dataset(path)
        if newvar in ds.variables:
            print(f'{newvar} already present')
            ds.close()
        else:
            temp = xr.concat([ds[var] for var in variables], dim = 'dummy') # becomes zeroth axis
            temp_weighted = temp.weighted(weights) 
            new = ds.assign({newvar:temp_weighted.mean('dummy')})
            new.load()
            ds.close() # Close before rewriting (bit ugly, netCDF4 could do appending to file)
            particular_encoding = {key : for_netcdf_encoding[key] for key in new.variables.keys()}
            new.to_netcdf(path, encoding = particular_encoding)
            print(f'{newvar} computed and file rewritten')
            #return path

if __name__ == "__main__":
    #merge_soilm(directory = Path('/nobackup/users/straaten/EXT_extra/45r1'))

    """
    Creation of matched set to ERA5 observations. First surface observation
    """
    t2m = SurfaceObservations(basevar = 'tg', basedir = '/nobackup/users/straaten/ERA5/', name = 't2m-anom_1979-01-01_2019-12-31_1D_0.25-degree') # Override normal E-OBS directory
    t2m.load(tmin = '2000-01-01', tmax = '2000-02-01')
    t2m.aggregatespace(clustername = 't2m-q095-adapted', level = 15)
    # write output. Correct the name for later matching to forecasts?
    # Utins match, namely K
    t2m.newvar = 'anom' # Actually already done
    t2m.construct_name(force = True) # Adds new tim/spacemethod
    print(t2m.name)
    out = xr.Dataset({'clustidfield':t2m.clusterarray,t2m.array.name:t2m.array})

    # Matching. preparation with a highresmodelclim 
    highresmodelclim = ModelClimatology(cycle='45r1', variable = t2m.basevar, **{'name':'tg_45r1_1998-06-07_2019-05-16_1D_0.38-degrees_5_5_mean'}) # Name for loading
    highresmodelclim.local_clim()
    newvarkwargs={'climatology':highresmodelclim}
    loadkwargs = {'llcrnr':(30,None),'rucrnr':(None,42)} # Limiting the domain a bit.

    f = Forecast('2000-01-07', prefix = 'hin_', cycle = '45r1')
    f.load('tg',**loadkwargs)

    #alignment = ForecastToObsAlignment(season = 'DJF', observations=t2m, cycle='45r1', n_members = 11, **{'expname':'paper3-1'}) # Season subsets the obs
    #alignment.match_and_write(newvariable = True, # Do I need loadkwargs
    #                          newvarkwargs = newvarkwargs,
    #                          loadkwargs = loadkwargs,
    #                          matchtime = False, 
    #                          matchspace= True)
