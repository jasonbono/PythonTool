from gm2 import rootbase
import numpy as np

class Issues(rootbase, object):
    """gm2 Issues class.
    """
    def __init__(self, runs = [], load=True, prefix=None):
        self.runs = runs
        self.loadSettings()
        super(Issues, self).__init__('issues', prefix=prefix) # name of library to load
        self.loadFiles()
        #if load:
        self.loadBasicMode()
        
    def loadSettings(self):
        """ Issues specific settings """
        self.fname_path = "IssueCollector/issues" # only required setting is the path in the root file
        self.n_probes = 1
        self.system_names = ["","Trolley","FixedProbe","Galil","Feedback","Fluxgate","SurfaceCoil"]
        self.system_names_short = ["", "tr", "fp", "g", "fb" ,"fl" ,"sc"]
        self.type_name = {0 : ["0"],
                          1 : ["0","1","3","4","5","6"],
                          2 : ["0","spk","jump","4","5"],
                          3 : ["0","1","3","4","5","6"],
                          4 : ["enable","state","setpoint","P","I","D"],
                          5 : ["0","1","3","4","5","6"],
                          6 : ["0","1","3","4","5","6"]}

    def getEvent(self):
        return np.array([self.data.Issue_event])

    def getSystem(self):
        return np.array([self.data.Issue_system])

    def getTime(self):
        return np.array([self.data.Issue_timestamp])
    
    def getIds(self):
        n = self.data.Issue_n_id
        return np.frombuffer(self.data.Issue_ids, dtype='u4')[:n]

    def getRun(self):
        return np.array([self.data.Issue_run])

    def getIdN(self):
        return np.array([self.data.Issue_n_id])

    def getType(self):
        return np.array([self.data.Issue_type])

    def getBasics(self):
        def callback():
            return [self.getEvent(), self.getTime(), self.getSystem(), self.getType(), self.getIdN()]
        return self.loop(callback)

    def getFpSpikeType(self, n):
        ''' If the current issue is a fixed probe spike this function returns the spike type.
        
        Args:
            n (int): probe index according to getIds().

        Returns:
            spike_type (int): bit0: Freq, bit 1: Amplitude, bit 2: Length, bit 3: power. 
        '''
        if (self.getType()[0] != 1)|(self.getSystem()[0] != 2):
            raise ValueError("getFpSpikeType is only valid for fixed probe spike isssues.")
        if n >= self.getIdN():
            raise ValueError("This event contains only %i spikes. Requested %i." % (self.getIfN(), n) )

        return(self.data.Issue_values[n*8])

    def getFpSpikeValues(self, n):
        ''' If the current issue is a fixed probe spike this function returns the spike values.

        Args:
            n (int): probe index according to getIds().

        Returns:
            [] : Freq, Amnp, Length, Power 
        '''
        if (self.getType()[0] != 1)|(self.getSystem()[0] != 2):
            raise ValueError("getFpSpikeType is only valid for fixed probe spike isssues.")
        if n >= self.getIdN():
            raise ValueError("This event contains only %i spikes. Requested %i." % (self.getIfN(), n) )

        return np.frombuffer(self.data.Issue_values, dtype='double')[8*n+1:8*n+5]

    def loadBasicMode(self):
        event, time, system, typ, n  = self.getBasics()
        self.event  = event[:,0]
        self.time   = time[:,0]
        self.system = system[:,0]
        self.typ    = typ[:,0]
        self.n      = n[:,0]

        s = (self.system == 2)&(self.typ == 1)
        #self.fp_spike_event = [set()]*378
        n = 378 
        self.fp_spike_event  = [ [] for i in range(n) ]
        self.fp_spike_times  = [ [] for i in range(n) ]
        self.fp_spike_type   = [ [] for i in range(n) ]
        self.fp_spike_values = [ [] for i in range(n) ]
        #self.fp_spike_time = []*378
        if len(self.chain.GetListOfFiles()) == 1:
          for i_ev, ev_ in enumerate(self.event[s]):
            #print "START", i_ev, ev_
            
            self.load(np.argwhere(self.event==ev_)[0])
            for j, probe_ in enumerate(self.getIds()):
                #print probe_
                if not int(ev_) in self.fp_spike_event[probe_]:
                    self.fp_spike_event[probe_].append(int(ev_))
                    self.fp_spike_times[probe_].append(self.getTime()[0])
                    self.fp_spike_type[probe_].append(int(self.getFpSpikeType(j)))
                    #if probe_ in [86]:
                    #    print "DEBUG", i_ev, j, self.getFpSpikeValues(j)
                    self.fp_spike_values[probe_].append(self.getFpSpikeValues(j))
                #self.fp_spike_event[probe_].add(int(ev_))

        #print "DEBUG DEBUG", len(self.fp_spike_event[86])


    def plot(self):
        s = (self.time > 0)&(self.system>0)
        import gm2
        for system in set(self.system[s]):
            for type_ in set(self.typ[s]):
                s_ = s&(self.system == system)&(self.typ==type_)
            gm2.plotutil.plot_ts(self.time[s_], self.typ[s_],'.', label=self.system_names[int(system)]+" (%.0f)" % type_, color=gm2.sns.color_palette()[int(system)-1])
        gm2.plt.ylabel("type")
        gm2.plt.legend()
        gm2.despine()
