# package gm2
# module galil
# Author: Simon Corrodi
#         scorrodi@anl.gov
#         September 2018

import numpy as np
from gm2 import rootbase

#import matplotlib.pyplot as plt
#from scipy.interpolate import interp1d
#from gm2util import *

class Galil(rootbase, object):
    def __init__(self, runs = []):
        self.runs = runs
        self.loadSettings()
        super(Galil, self).__init__('galil', None)
        self.loadFiles()
        
    def loadSettings(self):
        """ Trolley specific settings """
        self.fname_path     = "TreeGenGalilTrolley/tGalil"
        self.n_probes = 1

    ### Access Galil Data from the ROOT file ###
    def getTime(self):
        return self.data.Trolley_TimeStamp

    def getTension(self, m = -1):
        data =  np.frombuffer(self.data.Trolley_Tensions, dtype='double').reshape([2])
        if m < 0:
            return data
        else:
            return data[m]

    def getTemperature(self, m):
        data =  np.frombuffer(self.data.Trolley_Temperature, dtype='double').reshape([2])
        if m < 0:
            return data
        else:
            return data[m]

    def getControleZVoltage(self, m):
        data =  np.frombuffer(self.data.Trolley_ControleVoltage, dtype='double').reshape([3])
        if m < 0:
            return data
        else:
            return data[m]

    def getPosition(self, m=-1):
        data =  np.frombuffer(self.data.Trolley_Positions, dtype='double').reshape([3])
        if m < 0:
            return data
        else:
            return data[m]

    def getVelocity(self, m):
        data =  np.frombuffer(self.data.Trolley_Velocities, dtype='double').reshape([3])
        if m < 0:
            return data
        else:
            return data[m]

    def loop(self, func, *args):
        """ loop function with trolley specific selections """
        self.getEntry(0)
        return self.theLoop([], func, *args)



