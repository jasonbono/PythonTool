from gm2 import np

class Spikes(object):
    def __init__(self, phis=None, freqs=None, th=None):
        self.debug = True
        if (freqs is not None):
            self.init(phis, freqs, th)

    def init(self, phis, freqs, th):
        self.th = th
        self.rm = np.zeros([freqs.shape[-1]])
        self.freq = freqs.copy()
        self.outl = np.full([freqs.shape[0]], np.nan)

        def outlier(event, th):
            n = 1
            nn = 0
            if(event-n >=0)&(event+n<freqs.shape[0]):
                #for probe in range(17):
                mean = (self.freq[event-1] + self.freq[event+1])/2.0
                dphi  = phis[event + 1] - phis[event - 1]
                dfreq = (self.freq[event+1] - self.freq[event -1])
                if(abs(dphi) > 0.1e-4):  # use interpolation if dphi is large enough ...
                    mean = self.freq[event-1] + dfreq * (phis[event] - phis[event-1]) / dphi;
                self.outl[event] = self.freq[event] - mean
                d_pre  = self.freq[event] - self.freq[event-1]
                d_post = self.freq[event] - self.freq[event+1]
                if ((np.abs(self.outl[event]) > th)&(np.abs(d_pre) > th)&(np.abs(d_post) > th)&((d_pre * d_post)> 0)):
                    self.freq[event] = mean
                    self.rm[event]   = 1
                    nn += outlier(event-1, th)
            return nn

        for event in range(freqs.shape[0]):
            outlier(event, th)

    def isOutl(self):
        return self.rm > 0

    def isOk(self):
        return (self.rm == 0)&(np.isnan(self.outl) == False)
