from gm2 import rootbase
import numpy as np

class PlungingProbeWf(rootbase, object):
    """gm2 PlungingProbeWf class.
    """
    def __init__(self, runs = [], load=True):
        self.runs = runs
        self.loadSettings()
        super(PlungingProbeWf, self).__init__('plungingProbeWf', None) # name of library to load
        self.loadFiles()
        #if load:
        #self.loadBasicMode()
        
    def loadSettings(self):
        """ Feedback specific settings """
        self.fname_path = "PlungingProbeWfExtraction/plungingProbeWf" # only required setting is the path in the root file
        self.n_probes = 378

    #def getEvent(self):
    #    return np.array([self.data.FixedProbeFid_EntryID])

    #def getTime(self, n=-1):
    #    data = np.frombuffer(self.data.FixedProbeFid_GpsTime, dtype='u8')
    #    if n == -1:
    #        return data
    #    else:
    #        return np.array([data[n]])

    #def getBasics(self, n=-1):
    #    def callback():
    #        return [self.getEvent(), self.getTime(n)]
    #    return self.loop(callback)

    #def loadBasicMode(self):
    #    self.event, self.time = self.getBasics(self.n_probes//2)

    #def getFidN(self,n, probe=-1):
    #    return self.getFid(self.event[n])

    #def getFid(self, ev, probe=-1):
    #    if ev in self.event:
    #        ev_ = np.where(self.event[:,0] == ev)[0][0]
    #        self.load(ev_)
    #        data = np.frombuffer(self.data.FixedProbeFid_RawFid, dtype='u8').reshape(self.n_probes,-1)
    #        if probe == -1:
    #            return data
    #        else:
    #            return data[probe,:]
    #    else:
    #        raise NameError("Fids for event %i are not present." % ev)

    #def overview(self):
    #    import gm2
    #    s = self.time > 0
    #    gm2.plotutil.plot_ts(self.time[s], self.event[s])
    #    gm2.plt.ylabel("time")
    #    gm2.plt.xlabel("event")
    #    gm2.despine()
    #    gm2.plt.show()

    #def plotFid(self, ev, probe, alpha=1):
    #    data = self.getFid(ev, probe)
    #    import gm2
    #    time = gm2.util.ts2datetime(np.array([self.getTime(probe)]))[0,0]
    #    gm2.plt.plot(np.arange(data.shape[0])*1e-3, data, '.-', markersize=2, alpha=alpha, label="event %i, probe %i\n%s" % (ev, probe, time.strftime("%H:%M:%S.%f")))
    #    gm2.plt.xlabel("time [ms]")
    #    gm2.plt.ylabel("amplitude [adc]")
    #    gm2.despine()
