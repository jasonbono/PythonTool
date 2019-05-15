from gm2 import rootbase
import numpy as np

class Fluxgate(rootbase, object):
    def __init__(self, runs = []):
        self.runs = runs
        self.loadSettings()
        super(Fluxgate, self).__init__('fluxgate', None) # name of library to load
        self.loadFiles()
        
    def loadSettings(self):
        """ Fluxgate specific settings """
        self.fname_path = "TreeGenFluxgate/fluxgate" # only required setting is the path in the root file
        self.n_probes = 4
        self.n_8  = 8      # todo whats 8?
        self.n_24  = 24     # todo whats 24?
        self.tr_l = 7500   # trace length 

    def getTimeGPS(self):
        return self.data.waveform_gps_clock;

    def getTimeSystem(self):
        return self.data.waveform_sys_clock

    def getR(self, m=-1):
        data = np.frombuffer(self.data.waveform_fg_r, dtype='double')
        if (m < 0):
            return data
        else:
            return data[m]

    def getTheta(self, m=-1):
        data = np.frombuffer(self.data.waveform_fg_theta, dtype='double')
        if (m < 0):
            return data
        else:
            return data[m]

    def getZ(self, m=-1):
        data = np.frombuffer(self.data.waveform_fg_z, dtype='double')
        if (m < 0):
            return data
        else:
            return data[m]

    def getRate(self):
        return self.data.waveform_eff_rate

    def getWaveform(self, m=-1):
        data = np.frombuffer(self.data.waveform_trace, dtype='double').reshape([self.n_24, self.tr_l])
        if (m < 0):
            return data
        else:
            return data[m,:]

    def getBasics(self):
        def callback():
            return [self.getTimeGPS(), self.getWaveform()]
        return self.loop(callback)
    
