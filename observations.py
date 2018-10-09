import os

class SurfaceObservations(object):
    
    def __init__(self, variable, **kwds):
        self.variable = variable
        self.filepath = ''.join(["/nobackup/users/straaten/E-OBS/", variable, ".nc"])
        for key in kwds.keys():
            setattr(self, key, kwds[key])
    
    def downloadraw(self):
        """
        Downloads highres observations on regular 0.25 degree lat-lon grid
        """
        import urllib3
        
        urls = {"tg":"https://www.ecad.eu/download/ensembles/data/Grid_0.25deg_reg/tg_0.25deg_reg_v17.0.nc.gz",
        "tn":"https://www.ecad.eu/download/ensembles/data/Grid_0.25deg_reg/tn_0.25deg_reg_v17.0.nc.gz",
        "tx":"https://www.ecad.eu/download/ensembles/data/Grid_0.25deg_reg/tx_0.25deg_reg_v17.0.nc.gz",
        "rr":"https://www.ecad.eu/download/ensembles/data/Grid_0.25deg_reg/rr_0.25deg_reg_v17.0.nc.gz",
        "pp":"https://www.ecad.eu/download/ensembles/data/Grid_0.25deg_reg/pp_0.25deg_reg_v17.0.nc.gz"}
        
        zippath = self.filepath + '.gz'
        f = open(zippath, 'wb')
        http = urllib3.PoolManager()
        u = http.request('GET', urls[self.variable], preload_content = False)
        filesize = int(u.info().getheaders("Content-Length")[0])
        print("saving", filesize, "to", f.name)
        
        filesize_done = 0
        blocksize = 8192
        while True:
            buffer = u.read(blocksize)
            if not buffer:
                break
            filesize_done += len(buffer)
            f.write(buffer)
            status = r"%10d  [%3.2f%%]" % (filesize_done, filesize_done * 100. / filesize)
            print(status)
        
        u.release_conn()
        f.close()
        os.system("gunzip -k " + zippath) # Results in file written at filepath
        
    def loadraw(self):
        """
        Creates new attributes: marray with the masked data, units, lons and lats
        """
        import scipy.io as sio
        import numpy as np
        
        if not os.path.isfile(self.filepath):
            self.downloadraw()
        
        ifile = sio.netcdf_file(self.filepath, mode = 'r')
        variable = ifile.variables[self.variable] # Netcdf variable object referring to file location
        missval = variable._get_missing_value()
        self.units = variable.units.decode()
        self.lons = ifile.variables['longitude'].data
        self.lats = ifile.variables['latitude'].data
        self.marray = np.ma.masked_array(variable.data, np.equal(variable.data, missval))
        del(variable, missval)
        ifile.close()
        

