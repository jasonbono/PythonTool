# package gm2
# module fixedprobe
# Author: Simon Corrodi
#         scorrodi@anl.gov
#         September 2018
from __future__ import print_function
import numpy as np
from gm2 import rootbase
import gm2.util as util

class Trolley(rootbase, object):
    """ Class to access gm2 field trolley data

    Args:
        runs (list(int), optional): run numbers.
            If run numbers are provided, the corresponding files are loaded,
            otherwise :meth:`~gm2.Trolley.loadFiles` has to be evoked by the user.
        load (bool): if True the basic trolley properties are loaded and are avaiable as:
             aattr:`gm2.Trolley.time`, :attr:`gm2.Trolley.azi`, :attr:`gm2.Trolley.freq`. Defaults to False.
        prefix (str, optional) : prefix of the root file name. If None 'FieldGraphOut' is used. Defaults to None. For run1 offline files use 'FieldPlainRootOutput_'. 

    Attributes:
        time (array(int, events x n_probes)): time of NMR measurement in nano seconds GPS time. 
            Only present if :meth:`~gm2.Trolley.loadBasicMode` is used (e.g. load=True in the constructur). 
        azi (array(float, events x n_probes)): azimuthal position at the NMR measurement in degrees.
            The position from the odb is used.
            Only present if :meth:`~gm2.Trolley.loadBasicMode` is used (e.g. load=True in the constructur).
        freq (array(float, events x n_probes)): the extracted frequency of the NMR measurement.
            The method is used.
            Only present if :meth:`~gm2.Trolley.loadBasicMode` is used (e.g. load=True in the constructur).

    Examples:
        Fast access to the basics (time, azimuth, frequency)

        >>> import gm2
        >>> from gm2 import plt 
        >>> tr = gm2.Trolley([3997], True)
        >>> probe = 4
        >>> plt.plot(tr.azi[:,probe], tr.freq[:,probe])
        >>> plt.xlabel("azimuth [deg]")
        >>> plt.ylabel("NMR [Hz]")
        >>> plt.title("Probe %i" % (i+1))
        >>> plt.show()

        Use the encoder:
 
        >>> import gm2
        >>> from gm2 import plt
        >>> tr = gm2.Trolley([3997])
        >>> time, azi, freq = tr.getBasics(mode_phi=0) # default mode_phi=2 => barcode
        >>> probe = 4
        >>> plt.plot(azi[:,probe], freq[:,probe])
        >>> plt.xlabel("azimuth (encoder) [deg]")
        >>> plt.ylabel("NMR [Hz]")
        >>> plt.title("Probe %i" % (i+1))
        >>> plt.show()

        The call above is equivalent to:
        
        >>> import gm2
        >>> tr = gm2.Trolley([3997])
        >>> azi_method = 3
        >>> freq_method = 0
        >>> def callback():
        >>>     return [tr.getTimeGPS(), tr.getPhi(azi_method), tr.getFrequency(freq_method)]
        >>> time, azi, freq = tr.loop(callback)

        Access to user specified gm2 field trolley data

        >>> import gm2
        >>> tr = gm2.Trolley([3997], True)
        >>> freq_method = 3
        >>> def callback():
        >>>     return [tr.getFrequency(freq_method), tr.getFidLength(), tr.getTimeInterface()]
        >>> freq, fid_len, time = tr.loop(callback)

        User defined loop, access gm2 field trolley data (root) event wise

        >>> import gm2
        >>> tr = gm2.Trolley([3997], True)
        >>> for ev in range(tr.getEntries()):
        >>>     tr.load(ev)
        >>>     print(ev, tr.getFrequency(), tr.getFidLength())

    """
    def __init__(self, runs = [], load=False, prefix=None):
        self.runs = runs
        self.loadSettings()
        super(Trolley, self).__init__('trolley', prefix=prefix)
        self.loadFiles()
        if load:
            self.loadBasicMode()

    def loadBasicMode(self):
        """ Description.

        Args:
            a: test.
            b: test.
        
        Returns:
            bla

        """
        self.time, self.azi, self.freq = self.getBasics()
        self.freqAtTime = []
        s = ((self.time.min(axis=1) > 1e18)&
             (self.freq.min(axis=1) > 1e3)&
             (self.freq.max(axis=1) < 80e3))
        self.time = self.time[s]
        self.azi  = self.azi[s]
        self.freq = self.freq[s]
        for probe in range(self.n_probes):
            self.freqAtTime.append(util.interp1d(self.time[:, probe], self.freq[:, probe], fill_value='extrapolate'))
    
    #def multipole

    def loadSettings(self):
        """ Trolley specific settings """
        self.fname_path     = "TreeGenTrolley/trolley"
        self.n_probes       = 17 
        self.barcode_length = 300
        self.barcode_n      = 6

    def checkProbe(self, prob):
        return True if ( prob >= 0 )&( prob < self.n_probes ) else False

    def checkMode(self, m, m_max):
        if m > m_max:
            raise ValueError('Method m='+str(m)+' > '+str(m_max)+' does not exist.')

    ### Access Trolley Data from the ROOT file ###
    def getTimeGPS(self):
        return np.frombuffer(self.data.TimeStamp_GpsCycleStart, dtype='uint64') 

    def getTimeInterface(self):
        return np.frombuffer(self.data.TimeStamp_InterfaceCycleStart, dtype='uint64') 

    def getTimeBarcodes(self):
        return np.frombuffer(self.data.TimeStamp_BarcodeStart, dtype='uint64')

    def getTimeNMR(self):
        return np.frombuffer(self.data.TimeStamp_NMRStart, dtype='uint64')

    def getV1(self, lim=None):
        """trolley supply voltgae min and max over a readout cycle. 

        A drop in this voltage is an idicator for trolley ground loops in run1.

        Args:
           lim [None, 'min', 'max'] : if 'min'/'max' only one v1 is returned. Defaults to None.

        Returns:
           v1 ndarray(min, max) : if lim is None. Otherwise corresponding voltage.
        """
        if lim is None:
            return np.array([np.frombuffer(self.data.Monitor_V1Min, dtype='double'),
                         np.frombuffer(self.data.Monitor_V1Max, dtype='double')])
        if lim == 'min':
            return np.frombuffer(self.data.Monitor_V1Min, dtype='double')
        if lim == 'max':
            return np.frombuffer(self.data.Monitor_V1Max, dtype='double')
    
    def getTemperature(self, ext = False):
        """trolley temperature. 

        Args:
           ext (bool): if ture returns external temperature. Defaults to None.

        Returns:
            temp ndarray(double) : temperature 
        """
        if ext:
             return np.frombuffer(self.data.Monitor_TemperatureExt, dtype='double')
        else:
             return np.frombuffer(self.data.Monitor_TemperatureIn, dtype='double')

    def getRFPower(self, interface=False):
        """ Trolley rf power factor.
        
        Args:
            interface (bool, optional) : if True returns power factor of interface. Defaults to False.
        Returns:
            pwr ndarray(double): power factor
        """
        if interface:
           return np.frombuffer(self.data.Monitor_InterfacePowerFactor, dtype='double')
        else:
           return np.frombuffer(self.data.Monitor_RFPowerFactor, dtype='double')

    

    def getFrequency(self, m=-1):
        self.checkMode(m,6)
        data =  np.frombuffer(self.data.ProbeFrequency_Frequency, dtype='double').reshape([self.n_probes, -1])
        #data =  np.frombuffer(self.data.ProbeFrequency_Frequency, dtype='double').reshape([self.n_probes, 5])
        if (m < 0):
            return data
        else:
            return data[:,m]
  
    def getFrequencyUncs(self, m=-1):
        return self.getFrequencyUncertainty(m)

    def getFrequencyUncertainty(self, m=-1):
        self.checkMode(m,6)
        data =  np.frombuffer(self.data.ProbeFrequency_FrequencyUncertainty, dtype='double').reshape([self.n_probes, -1])
        #data =  np.frombuffer(self.data.ProbeFrequency_Frequency, dtype='double').reshape([self.n_probes, 5])
        if (m < 0):
            return data
        else:
            return data[:,m]

    def getMultipole(self):
        return np.frombuffer(self.data.FieldMultipole_Multipole, dtype='double')

    def getMultipoleTime(self):
        return np.array([self.data.FieldMultipole_Time]) 

    def getMultipolePhi(self):
        return np.array([self.data.FieldMultipole_Phi])

    def getAmplitude(self):
        """ Fid Amplitude of the 17 probes in the loaded radout cycle.

        Returns:
           array(float): amplitude of the 17 fids of the current cycle.
        """
        return np.frombuffer(self.data.ProbeSignal_Amplitude, dtype='double')

    def getFidLength(self):
        """ Fid Length of the 17 probes in the loaded radout cycle.

        Returns:
            array(float): length in seconds of the 17 fids of the current cycle.
        """
        return np.frombuffer(self.data.ProbeSignal_FidLength, dtype='double')

    def getFidPower(self):
        """ Fid Power of the 17 probes in the loaded radout cycle.

        Returns:
            array(float): power of the 17 fids of the current cycle.
        """
        return np.frombuffer(self.data.ProbeSignal_FidPower, dtype='double')


    def getFidFitChi2(self):
        """ Fid Fit Chi2of the 17 probes in the loaded radout cycle.

        Returns:
            array(float): chi2 of the fid fit of the 17 fids of the current cycle.
        """
        return np.frombuffer(self.data.ProbeSignal_FitChi2, dtype='double')


    def getFid(self, probe=-1):
        """ Fid Profile (envelope) of the selected pronbe are all 17 probes of the loaded radout cycle.

        Args:
            probe (int, optional): supply probe number to select the fid profile of one specific probe.
                If probe < 0 an array with all fid profiles is returned.

        Returns:
            array(float): if probe >= 0: 1D array with the fid profile (100 points long).
                          if probe > 0:  2D array of size (100, 17) with the fid profiles of all 17 probes.
        """
        self.checkMode(probe, 17)
        data = np.frombuffer(self.data.ProbeSignal_FidProfile, dtype='double').reshape(-1,100).T
        
        if probe < 0:
            return data
        else:
            return data[:, probe]


    def getPhi(self, m=-1):
        """Access trolley position.
        
        Three different positions methodes are present per probe:
        0: the odb value. This corresponds to the online position determination.
        1: position from the encoders. Thats recalculated in the production.
        2: barcode. The barcode is only valid if the position source getPosSource() is .

        Args:
            m (int, optional) : select position method described above. Defaults to -1 which returns all methods.

        Returns:
            numpy.array(float, nevents x probes x [3]) : position in degrees. If no method is selected all three are returned.
        
        """
        self.checkMode(m, 3)
        
        data = np.frombuffer(self.data.Position_Phi, dtype='double').reshape([self.n_probes, 3])
        if m < 0:
            return data
        else:
            return data[:, m]

    def getPosSource(self, m=-1):
        """Position Source: -1 odb, 1: encoder, 0: barecode
        """ 
        self.checkMode(m, 3)
        
        data = np.frombuffer(self.data.Position_PosSource, dtype='double').reshape([self.n_probes, 3])
        if m < 0:
            return data
        else:
            return data[:, m]

    def getPosQuality(self, m=-1):
        """Position Quality: 
            
            bit 0: rush
            bit 1: tension
            bit 3: offset
            bit 4: overall offset with respect to encoder
            bit 5: timing alignment

        Returns:
            numpy.array(float, nevents x probes x [3]) : quality flag. If no method is selected all three are returned. Only m=2 is miningful.
        """ 
        self.checkMode(m, 3)
        
        data = np.frombuffer(self.data.Position_PosQuality, dtype='double').reshape([self.n_probes, 3])
        if m < 0:
            return data
        else:
            return data[:, m]

    def getPhiUncs(self, m=-1):
        self.checkMode(m, 3)
        data =  np.frombuffer(self.data.Position_PhiUncertainty, dtype='double').reshape([self.n_probes, 3])
        if m < 0:
            return data
        else:
            return data[:, m]

    def getX(self):
        return np.frombuffer(self.data.Position_X, dtype='double')

    def getY(self):
        return np.frombuffer(self.data.Position_Y, dtype='double')

    def getBarcodeTraces(self):
        return np.frombuffer(self.data.Barcode_traces, dtype='float32').reshape([self.n_probes, self.barcode_n, self.barcode_length])

    def loadBarcode(self, add = None):
        if add is None:
            self.activateBranches(["TimeStamp","Barcode"])
        else:
            self.activateBranches(["TimeStamp","Barcode"] + add)
        nentries = self.getEntries()  
        self.bc = np.empty([7, nentries * self.n_probes * self.barcode_length])
        index = np.uint64(0)  
        for ev in range(1, nentries): # skip the first event, its usually not complete
            self.getEntry(ev)
            if ev % 100 == 0:
                print("\rReading barcode "+str(ev)+"/"+str(nentries)+" " + "%0.2f" % (100.0 * ev/nentries)+"%", end=' ')
            n = self.getBarcodeLengths().sum()
            self.bc[:, index:(index+n)] = self.getCroppedBarcode()
            index = index + n
        self.bc = self.bc[:,:-int(nentries * self.n_probes * self.barcode_length-index)]

    def getBarcodeAdcRefs(self):
        return np.frombuffer(self.data.Barcode_adc_reference, dtype='double')

    def getBarcodeLengths(self):
        return np.frombuffer(self.data.Barcode_lengths, dtype='uint32')

    def getBarcodePeriods(self):
        return np.frombuffer(self.data.Barcode_sampling_periode, dtype='uint16')

    def getBarcodeDelays(self):
        return np.frombuffer(self.data.Barcode_acquisition_delay, dtype='uint16')

    def getBarcodeRefs(self):
        return np.frombuffer(self.data.Barcode_ref_cm, dtype='uint16')

    ### Access Quality Table ### 
    '''def getOutlier(self, m=-1):
        self.hasQt()
        self.checkMode(m,6)
        data =  np.frombuffer(self.qt.QT_FrequencyOutlier, dtype='double').reshape([self.n_probes, 6])
        if (m < 0):
            return data
        else:
            return data[:,m]

    def getFrequencySmooth(self, m=-1):
        self.hasQt()
        self.checkMode(m,6)
        data =  np.frombuffer(self.qt.Frequency_Frequency, dtype='double').reshape([self.n_probes, 6])
        if (m < 0):
            return data
        else:
            return data[:,m]

    def getNum(self):
        self.hasQt()
        return np.frombuffer(self.qt.QT_num, dtype='uint16')

    def isComplete(self):
        self.hasQt()
        return np.frombuffer(self.qt.QT_complete, dtype='uint16')'''

    def loop(self, func, probes=[], *args):
        """ loop function with trolley specific selections """
        self.getEntry(0)
        sel = self.select(probes)
        return self.theLoop(sel, func, *args)
        
    def select(self, probes=[]):
        """ constructs selection, try to run this only once

            Parameters: 
            probes: 1-17
            phis:   min, max
            times:  min, max
        """
        sel = np.full(self.n_probes, True)
        if  probes != []:
            sel = sel & np.isin(np.arange(1, 1+self.n_probes), probes)
        return sel
        ### legacy stuff ###

    def getMultipoleBasics(self):
        self.activateBranches(["FieldMultipole"])
        def callback():
            return [self.getMultipoleTime(), self.getMultipolePhi(), self.getMultipole()]
        return self.loop(callback)

    def getBasics(self, phi_lims=[], time_lims=[], probes=[], mode_freq=0, mode_phi=2):
        """Load the basic trolley information: time, frequency and positio.
        
        Args:
           mode_freq (int, optional) : Specifies which frequency method is returned. Defaults to 0.
           mode_phi (int, optioanl) : Specifies which position method is used. 0: odb, 1: encoder, 2: barcode. Defaults to 2.
           phi_lims ([float, float], optional) : limits returned data to azimuthal range [start, end]. Azimuth in deg.
           time_lims ([float, float], optional) : limits returned data to a time range [start, end]. Time is specified in timestamp in ns.
           probes (list[int], optioanl) : data of probes in this lists is returned. Defaults to []. All probes are used.
        
        Returns:
           time, frequency, azimuth ([nevents x nprobes]) : time as timestamps in ns, frequencies in Hz, azimuth in degrees.
        """
        
        self.activateBranches(["ProbeFrequency", "TimeStamp", "Position"])
        def callback():
            return [self.getTimeGPS(), self.getPhi(mode_phi), self.getFrequency(mode_freq)] 
        time, phi, freq = self.loop(callback, probes=probes)
        sel = np.full(phi.shape[0], True)
        if phi_lims != []:
            if len(phi_lims) == 2:
                sel = sel & (phi.max(axis=1) > phi_lims[0]) & (phi.min(axis=1) < phi_lims[1]) 
            else:
                raise ValueError('getBasics() phi_lims needs to be of length 2 ('+str(len(phi_lims))+'): phi_lim=[min, max]')
        if time_lims != []:
            if len(time_lims) == 2:
                sel = sel & (time.max(axis=1) > time_lims[0]) & (time.min(axis=1) < time_lims[1]) 
            else:
                raise ValueError('getBasics() time_lims needs to be of length 2 ('+str(len(time_lim))+'): phi_lims=[min, max]')
        return time[sel], phi[sel], freq[sel]

    '''
    def show(self, probe):
        print("Probe:\t", probe)
        print("TimeGPS:\t",    "%i," % self.getTimeGPS(probe))
        print("Frequency:") 
        for i in range(6):
            ff = self.getFrequency(probe, i)
            print ("\t\t", ff[0], "+/-", ff[1] )
        print("Amplitude:\t", "%f," % self.getAmplitude(probe))
        print("FidLength:\t", "%f," % self.getFidLength(probe))
        print("Phi:")
        for i in range(3):
            ff = self.getPhi(probe, i)
            print("\t\t",       "%f," % ff[0], "+/-", ff[1])
        print("X:\t\t",         "%f," % self.getX(probe))
        print("Y:\t\t",         "%f," % self.getY(probe))
        print(" ")
    '''
            
    def getRunAtPhi(self, phi, probe = 0):
        if self.phis == None:
            def callback():
                return [self.getPhis()]
            self.phis = self.loop(callback)[0]

        return nearestIndex(self.phis[:, probe, 0], phi) 

    def getCroppedBarcode(self):
        lengths = self.getBarcodeLengths()
        ts_l = self.getTimeBarcodes()
        ts_f = util.interp1d(np.append(0, lengths[:-1].cumsum()), ts_l , fill_value='extrapolate')
        mask = (np.tile(np.arange(self.barcode_length), (self.n_probes, 1))< lengths[:,None])
        return np.concatenate([ts_f(np.arange(lengths.sum()))[None,:], np.rot90(self.getBarcodeTraces(),1, (0,1))[:,mask]])

    def getBarcodeTrace(self):
        return np.flipud(np.rot90(self.getBarcodeTraces(),1, (0,1)).reshape([self.barcode_n, self.barcode_length * self.n_probes]))

    def getBarcodeTs(self):
        ts_l = self.getTimeBarcodes()
        ts_f = util.interp1d(np.arange(0,self.n_probes * self.barcode_length + 1, self.barcode_length), np.append(ts_l, [(2 * ts_l[-1] - ts_l[-2])]))
        return ts_f(np.arange(self.n_probes * self.barcode_length)) 

    def getBarcode(self, start, nframes):
        tr = np.zeros([6,0])
        ts = np.zeros([0])
        for ev in np.arange(start, start+nframes):
            self.getEntry(ev)
            tr_l = self.getBarcodeTrace()
            tr = np.concatenate((tr, tr_l), axis=1)
            #ts_l = 
            #self.getTimeBarcodes()
            #ts_f = interp1d(np.arange(0,self.n_probes * self.barcode_length + 1, self.barcode_length), np.append(ts_l, [(2 * ts_l[-1] - ts_l[-2])]))
            ts = np.concatenate((ts, self.getBarcodeTs()))
 
        s = (tr[0,:]>1.0)&(tr[0,:]<5.0)
        return tr[:,s], ts[s]


        #if plot:
        #    plt.plot(tr[0,(tr[0,:]>1.0)&(tr[0,:]<5.0)])
        #    plt.plot(tr[1,(tr[1,:]>1.0)&(tr[1,:]<5.0)] - tr[2,(tr[2,:]>1.0)&(tr[2,:]<5.0)])
        #    plt.show()
        #    #raw_input();
        #return tr


    def getCalibration(self, fname="trCalibration.json"):
        import os
        path = os.environ['GM2']
        import json
        with open(path+"/data/"+fname, "r") as f:
            data = json.load(f)
        return np.array([data['calibration'][p]['offset'] for p in range(17)])[None,:]

