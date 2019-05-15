from gm2 import rootbase
import numpy as np
import gm2

class Feedback(rootbase, object):
    """gm2 PSFeedback class.
    """
    def __init__(self, runs = [], prefix=None):
        self.runs = runs
        self.prefix = prefix
        self.loadSettings()
        super(Feedback, self).__init__('psFeedback', prefix=prefix) # name of library to load
        self.loadFiles()
        #if load:
        #    self.loadBasicMode()
        
    def loadSettings(self):
        """ Feedback specific settings """
        self.fname_path = "TreeGenPSFeedback/psFeedback" # only required setting is the path in the root file
        self.n_probes = 1

    def getProbeList(self):
        return np.frombuffer(self.data.data_ProbeList, dtype='uint16') == 1

    def getMeanFreq(self):
        return np.array([self.data.data_FilteredMeanFreq])

    def getTime(self):
        return np.array([self.data.data_GpsClock])

    def getCurrentSetPoint(self):
        return np.array([self.data.data_CurrentSetPoint])

    def getFieldSetPoint(self):
        return np.array([self.data.data_FieldSetPoint])

    def getICoef(self):
        return np.array([self.data.data_ICoeff])

    def getCurrent(self):
        return np.array([self.data.data_Current])

    def getBasics(self):
        def callback():
            return [self.getTime(), self.getMeanFreq(), self.getProbeList(), ]
        return self.loop(callback)

    def probeList(self, ev=1):
        if self.getEntries() > 0:
            self.load(ev)
            return self.getProbeList()
        else:
            return np.nan

    def plot(self, issues=False):
        def callback():
            return [self.getTime(), self.getMeanFreq(), self.getFieldSetPoint(), self.getCurrent(), self.getCurrentSetPoint(), self.getICoef()]
        time, freq, freqSet, cur, curSet, coef = self.loop(callback)
        if issues:
            i = gm2.Issues(self.runs, prefix=self.prefix)
        
        s = time > 0
        ax1 = gm2.plt.subplot(111)
        gm2.plotutil.plot_ts(time[s], freq[s], '.', label="mean freq", markersize=3)
        gm2.plotutil.plot_ts(time[s], freqSet[s], '.', label="freq set point", markersize=2)
        ax1.set_ylabel("freq [Hz]")
        ax2 = ax1.twinx()
        gm2.plotutil.plot_ts(time[s], cur[s], '.', label="current", markersize=2, color=gm2.sns.color_palette()[2])
        ax2.set_ylabel("current  [mA]")
        #gm2.plotutil.plot_ts(time, cur, '.', label="current")
        ax3 = ax1.twinx()
        gm2.plotutil.plot_ts(time[s], coef[s], '.', label="mode", markersize=2, color=gm2.sns.color_palette()[3])
        if issues: 
          s_i = (i.system == 4)
          gm2.plotutil.plot_ts(i.time[s_i], gm2.np.ones_like(i.time[s_i])*0.2, 'o', label="issue",  markersize=4, color=gm2.sns.color_palette()[5])
        ax3.get_yaxis().set_visible(False)
        l  = ax1.get_legend_handles_labels()[0]+ax2.get_legend_handles_labels()[0]+ax3.get_legend_handles_labels()[0]
        ll = ax1.get_legend_handles_labels()[1]+ax2.get_legend_handles_labels()[1]+ax3.get_legend_handles_labels()[1]
        gm2.plt.legend(l, ll, markerscale=3)
        gm2.despine()
        gm2.plt.show()
        

    '''
    def getProbeList(self):
        return np.frombuffer(self.data.data_BotTime, dtype='uint64')
    '''

    '''
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

    
    '''
    
