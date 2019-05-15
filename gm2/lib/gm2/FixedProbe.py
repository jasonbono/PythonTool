# package gm2
# module fixedprobe
# Author: Simon Corrodi
#         scorrodi@anl.gov
#         September 2018


import numpy as np
from gm2 import rootbase, util

class FixedProbe(rootbase, object):
    def __init__(self, runs = [], load=False, prefix=None):
        self.runs = runs
        self.loadSettings()
        super(FixedProbe, self).__init__('fixedProbe',prefix=prefix)
        self.loadFiles()
        if load:
            self.loadBasicMode()

    def loadBasicMode(self):
        self.loadPhi()
        self.time, self.freq = self.getBasics()
        self.freqAtTime = []
        for probe in range(self.n_probes):
            self.freqAtTime.append(util.interp1d(self.time[:, probe], self.freq[:, probe]))
    
    def loadPhi(self, event=1):
        if(self.getEntries() >= event):
            self.load(event)
            self.phi = self.getPhi()

    def getStationPhi(self, event=1):
        self.loadPhi(event)
        station_phi = []
        for station in self.getStations():
            s = self.select(yokes=[station[0]], aziIds=[station[1]])
            station_phi.append(self.phi[s].mean())
        return np.array(station_phi)

    ## selection in basic mode
    def t(ids=[], yokes=[], radIds=[], aziIds=[], layers=[]):
        s = self.select(ids=[], yokes=[], radIds=[], aziIds=[], layers=[])
        return self.time[s]

    def f(ids=[], yokes=[], radIds=[], aziIds=[], layers=[]):
        s = self.select(ids=[], yokes=[], radIds=[], aziIds=[], layers=[])
        return self.freq[s]

    def tf(ids=[], yokes=[], radIds=[], aziIds=[], layers=[]):
        s = self.select(ids=[], yokes=[], radIds=[], aziIds=[], layers=[])
        return self.time[s], self.freq[s]
        
    def loadSettings(self):
        """ FP specific settings """
        self.fname_path    = "TreeGenFixedProbe/fixedProbe"
        self.n_probes      = 378
        self.yokes  = np.arange(ord('A'), ord('L')+1)
        self.aziIds = np.arange(1,6+1)

    def getStations(self):
        stations = []
        for yoke_ in self.yokes:
            for aziId in self.aziIds:
                stations.append((chr(yoke_), aziId))
        return stations

    def loadIds(self, event=1): 
        #self.activateBranches(["Header"])
        if(self.getEntries() >= event):
            self.load(event)
            self.id = { 'yoke':  self.getYoke(),
                        'azi':   self.getAziId(),
                        'rad':   self.getRadId(),
                        'layer': self.getLayer(),
                        'mux':   self.getMux(),
                        'round': self.getRound()


                      };
            self.loadPos()
        else:
            print("Cannot load ids, no data present")

    def loadPos(self):
        from gm2.constants import FP
        self.pos_r     = np.full(self.n_probes, 0.0)
        self.pos_theta = np.full(self.n_probes, 0.0)
        for layer in ['T','B']:
            for rad in ['I','M','O']:
                self.pos_r[     (self.id['layer'] == (layer)) & (self.id['rad']==(rad))] = FP.probes.position.getR(layer, rad)
                self.pos_theta[ (self.id['layer'] == (layer)) & (self.id['rad']==(rad))] = FP.probes.position.getTheta(layer, rad)

    ### Access FP Data from the ROOT file ###
    def getTimeSystem(self):
        return np.frombuffer(self.data.Frequency_SystemTimeStamp, dtype='u8')

    def getTimeGPS(self):
        return np.frombuffer(self.data.Frequency_GpsTimeStamp, dtype='u8')

    def getTriggerDelay(self):
        """Trigger delay (used in mode 3) in 10 us."""
        return np.frombuffer(self.data.Trigger_Delay, dtype='uint16')

    def getQuality(self):
       """Quality Flag. 0 is good.
       
       Bit 0: exluded by default
       Bit 1: amplitude
       Bit 2: length
       Bit 3: power
       Bit 4: resolution
       Bit 5: amplitude template
       Bit 6: length template
       Bit 7: power template
       Bit 8: resolution template
       Bit 9: resolution < 0.001
       
       Returns:
           numpy.array(int, [nevents x nprobes]) : quality flag (see above). 0 means healty.
       """
       pass

    def getTimeDevice(self):
        return np.frombuffer(self.data.Frequency_DeviceTimeStamp, dtype='u8')

    def getFrequency(self, p=-1):
        if (p < 0)|(p >= 6):
            return np.frombuffer(self.data.Frequency_Frequency, dtype='double').reshape([self.n_probes, -1])
        else:
            return np.frombuffer(self.data.Frequency_Frequency, dtype='double').reshape([self.n_probes, -1])[:,0]

    def getFrequencyUncertainty(self, p=-1):
        if p >= 6:
            raise ValueError('getFrequency() Method p='+str(p)+' >= 6 does not exist.')
        if (p < 0):
            return np.frombuffer(self.data.Frequency_FrequencyUncertainty, dtype='double').reshape([self.n_probes, -1])
        else:
            return np.frombuffer(self.data.Frequency_FrequencyUncertainty, dtype='double').reshape([self.n_probes, -1])[:,p]

    def getAmplitude(self):
        return np.frombuffer(self.data.Signal_Amplitude, dtype='double')

    def getPower(self):
        return np.frombuffer(self.data.Signal_FidPower, dtype='double')

    def getSNR(self):
        return np.frombuffer(self.data.Signal_SNR, dtype='double') 

    def getFidLength(self):
        return np.frombuffer(self.data.Signal_FidLength, dtype='double')

    def getFidChi2(self):
        return np.frombuffer(self.data.Signal_FitChi2, dtype='double')

    def getPhi(self, correct=True):
        data = np.frombuffer(self.data.Position_Phi, dtype='double')
        #if correct:
        #    if data[self.select(yokes=['L'])].mean() < 305.:
        #        data[self.select(yokes=['L'])] += 30.
        return data

    def getX(self):
        return np.frombuffer(self.data.Position_X, dtype='double')

    def getY(self):
        return np.frombuffer(self.data.Position_Y, dtype='double')

    def getAziId(self):
        return np.frombuffer(self.data.Header_AziId, dtype='uint16')

    def getId(self):
        return np.frombuffer(self.data.Header_ProbeId, dtype='a1')

    def getMux(self):
        return np.frombuffer(self.data.Header_MuxId, dtype='uint16')

    def getRound(self):
        return np.frombuffer(self.data.Header_RoundId, dtype='uint16')

    def getLayer(self):
        return np.array([chr(c) for c in np.fromstring(self.data.Header_LayerId, dtype='B')])
        #return np.frombuffer(self.data.Header_LayerId, dtype='a1')

    def getYoke(self): 
        return np.array([chr(c) for c in np.fromstring(self.data.Header_YokeId, dtype='B')])
        #return np.frombuffer(self.data.Header_YokeId, dtype='a1')

    def getRadId(self):
        return np.array([chr(c) for c in np.fromstring(self.data.Header_RadId,dtype='B')])
        #return np.frombuffer(self.data.Header_RadId, dtype='a1')

    def getAlignment(self, n=10):
        self.load(n)
        return [self.getId(), self.getPhi(), self.getYoke(), self.getAziId(), self.getLayer(), self.getRadId()]

    def getHealt(self):
        return np.frombuffer(self.data.Header_Health, dtype='uint16')

    def loop(self, func, ids=[], yokes=[], radIds=[], aziIds=[], layers=[], *args):
        """ loop function with fp specific selections """
        self.loadIds()
        self.getEntry(0)
        sel = self.select(ids=ids, yokes=yokes, radIds=radIds, aziIds=aziIds, layers=layers)
        return self.theLoop(sel, func, *args)
        
    def select(self, ids=[], yokes=[], radIds=[], aziIds=[], layers=[]):
        """ constructs selection, try to run this only once

            Parameters: 
            ids:    0 to 377
            yokes:  'A' to 'L'
            radIds: 'I', 'M', 'O'
            aziIds: 1 to 6
            layers: 'T','B'
        """
        sel = np.full(self.n_probes, True)
        if ids != []:
            sel = sel & np.isin(self.getId(), ids)
        if yokes != []:
            sel = sel & np.isin(self.getYoke(), yokes)
        if radIds != []:
            sel = sel & np.isin(self.getRadId(), radIds)
        if aziIds != []:
            sel = sel & np.isin(self.getAziId(), aziIds)
        if layers != []:
            sel = sel & np.isin(self.getLayer(), self.toInt(layers))
        return sel


    def getBasics(self, time_lims=[], ids=[], yokes=[], radIds=[], aziIds=[], layers=[], mode_freq=0):
        self.activateBranches(["Frequency", "Header"])
        def callback():
            return [self.getTimeGPS(), self.getFrequency(mode_freq)] 
        time, freq = self.loop(callback, ids=ids, yokes=yokes, radIds=radIds, aziIds=aziIds, layers=layers)
        sel = np.full(time.shape[0], True)
        if time_lims != []:
            if len(time_lims) == 2:
                sel = sel & (time.max(axis=1) > time_lims[0]) & (time.min(axis=1) < time_lims[1]) 
            else:
                raise ValueError('getBasics() time_lims needs to be of length 2 ('+str(len(time_lim))+'): phi_lims=[min, max]')
        return time[sel], freq[sel]

