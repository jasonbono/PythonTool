from gm2 import rootbase
import numpy as np

class SurfaceCoil(rootbase, object):
    """gm2 SurfaceCoil class.
    """
    def __init__(self, runs = [], load=True, prefix=None):
        self.runs = runs
        self.loadSettings()
        super(SurfaceCoil, self).__init__('surfaceCoil', prefix=prefix) # name of library to load
        self.loadFiles()
        if load:
            self.loadBasicMode()
        
    def loadSettings(self):
        """ SurfaceCoils specific settings """
        self.fname_path = "TreeGenSurfaceCoil/surfaceCoils" # only required setting is the path in the root file
        self.n_probes = 100
        self.n_probe_azi = 4

    def loadBasicMode(self):
        self.time, self.cur_t, self.cur_b, self.cur_az = self.getBasics()

    def getTime(self, position = 'top'):
        """read times in ns.
        
        Args:
            position (['top', 'bottom'], optional) : top or bottom coils. Defaults to 'top'.

        Returns:
            ndarray(dobule, nevents x 100) : times 
        """
        if position == 'top':
            return np.frombuffer(self.data.data_TopTime, dtype='uint64')
        else:
            return np.frombuffer(self.data.data_BotTime, dtype='uint64')


    def getCurrent(self, position = 'top', setpoint = False):
        """SurfaceCoil currents read or setpoints
        
        Args:
            position (['top', 'bottom', 'azi'], optional) : top, bottom or azi coils. Defaults to 'top'.
            setpoint (bool, optional) : if True the setpoints are returned. Defaults to False.

        Returns:
            ndarray(dobule, nevents x 100) : currents
        """
        if setpoint:
            if position == 'top':
                return np.frombuffer(self.data.data_TopCurrentSetPoints, dtype='double')
            elif position == 'azi':
                return np.frombuffer(self.data.data_AzCurrentSetPoints, dtype='double')
            else:
                return np.frombuffer(self.data.data_BotCurrentSetPoints, dtype='double')
        else:
            if position == 'top':
                return np.frombuffer(self.data.data_TopCurrents, dtype='double')
            elif position == 'azi':
                return np.frombuffer(self.data.data_AzCurrents, dtype='double')
            else:
                return np.frombuffer(self.data.data_BotCurrents, dtype='double')


    def getTemperature(self, position = 'top'):
        """SurfaceCoil temperature.
        
        Args:
            position (['top', 'bottom', 'azi'], optional) : top, bottom or azi coils. Defaults to 'top'.

        Returns:
            ndarray(dobule, nevents x 100) : temperature
        """
        if position == 'top':
            return np.frombuffer(self.data.data_TopTemps, dtype='double')
        elif position == 'azi':
            return np.frombuffer(self.data.data_AzTemps, dtype='double')
        else:
            return np.frombuffer(self.data.data_BotTemps, dtype='double')


    def getBasics(self):
        def callback():
            return [self.getTime(), self.getCurrent('top'), self.getCurrent('bottom'), self.getCurrent('azi')]
        return self.loop(callback)
    
