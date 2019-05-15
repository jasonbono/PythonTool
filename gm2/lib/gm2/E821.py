import numpy as np

class E821:
    def __init__(self, fname):
        self.fname = fname
        self.data = np.loadtxt(fname, dtype={'names': ('number','encoder0','encoder1','frequency','flag'),
                                                        'formats': ('i4', 'i4', 'i4', 'f','i4')}, skiprows=5)
        self.p_index = np.argwhere(self.data['flag']==9999)[:,0]

    def getProbeIndex(self, probe):
        i_end = self.p_index[probe]
        if probe in [0]:
            i_start = 0
        else:
            i_start = self.p_index[probe-1]
        return i_start, i_end

    def getProbe(self, probe):
        return np.concatenate((self.getEncoder(probe, 0), self.getEncoder(probe, 1), self.getFrequency(probe)))

    def getEncoder(self, probe, no=0):
        i_start, i_end = self.getProbeIndex(probe)
        s = (self.data['flag'][i_start:i_end] == 0)&(self.data['frequency'][i_start:i_end]>0.0)
        return self.data['encoder'+str(no)][i_start:i_end][s]

    def getPhi(self, probe):
        en = (self.getEncoder(probe, 0) + self.getEncoder(probe, 1))/2.
        return en / (108450. - 270.) * np.pi * 2.

    def getFrequency(self, probe):
        i_start, i_end = self.getProbeIndex(probe)
        s = (self.data['flag'][i_start:i_end] == 0)&(self.data['frequency'][i_start:i_end]>0.0)
        return self.data['frequency'][i_start:i_end][s]

    def convert(self, outname='out.csv'):
        f1=open(outname, 'w+')
        for probe in np.arange(17):
            for i in np.arange(e1.getEncoder(probe,0).shape[0]):
                f1.write(str(probe)+", "+str(self.getEncoder(probe,0)[i])+", "+str(self.getEncoder(probe,1)[i])+", "+str(self.getFrequency(probe)[i])+"\n")
        f1.close()
