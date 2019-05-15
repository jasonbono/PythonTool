from __future__ import print_function
import gm2
        #if len(self.data.ProbeFrequency_Frequency)/17 == 6:
        #else:
        #    data = np.frombuffer(self.data.Position_Phi, dtype='float32').reshape([self.n_probes, 3])
from gm2 import np
import ROOT

class Interpolation(object):
    """Loads fixed probe and trolley data and provides utilities needed for interpolation.
    
    Attributes:
        fp (FixedProbe): Fixed Probe Class
        tr (Trolley): Trolley Class"""
    def __init__(self, runs):
        self.fp = gm2.FixedProbe(runs, True)
        self.tr = gm2.Trolley(runs, True)
        self.load();

    def load(self, freq_type=0):
        pass
        #def fp_callback():
        #    return [self.fp.getTimeGPS(), self.fp.getFrequency(freq_type)]
        #self.fp_time, self.fp_freq = self.fp.loop(fp_callback)

        #def tr_callback():
        #    return [self.tr.getTimeGPS(), self.tr.getPhi(0), self.tr.getFrequency(freq_type) ]
        #self.tr_time, self.tr_pos, self.tr_freq  = self.tr.loop(tr_callback)

    def loadFpRes(self):
        h = []
        res_simon = np.full_like(self.fp.freq, np.nan)
        #res_ran   = np.full_like(fp_freq, np.nan)
        fp_res = []
        for probe in np.arange(self.fp.n_probes):
            print("resolution probe ", probe)
            h.append(ROOT.TH1F("h"+str(probe), "h"+str(probe), 10001, -500, 500))
            for event, freq in enumerate(self.fp.freq[:1000,probe]):
                if (event - 1 >= 0)&(event+1 < self.fp.freq.shape[0] ):
                    res_simon[event, probe] = freq - (self.fp.freq[event-1, probe] + self.fp.freq[event+1, probe])/2.0
                    h[probe].Fill(res_simon[event, probe])
                #if event+10 < fp_freq.shape[0]:
                #    res_ran[event, probe]   = fp_freq[event:event+10, probe].std()
            fp_res.append(h[probe].GetRMS()/1.2)
        self.fp_res = np.array(fp_res)

    def getFpTrlyTimes(self, cw=False, plot=[]):
        ''' loads the time the trolley passes by the individuall fixed probe'''
        skip = 1
        rise_time   = np.full(self.fp.n_probes, np.nan)
        for probe in np.arange(self.fp.n_probes):
            if cw:
                rise_time[probe], th   = gm2.util.cf_local(np.flip(self.fp.time[skip:, probe]), np.flip(self.fp.freq[skip:, probe]), returnTh=True, baseline_o=10)
            else:
                rise_time[probe], th   = gm2.util.cf_local(self.fp.time[skip:, probe], self.fp.freq[skip:, probe], returnTh=True, baseline_o=10)
            if probe in plot:
                gm2.plt.plot(self.fp.time[skip:, probe], self.fp.freq[skip:, probe])
                gm2.plt.title("Probe "+str(probe))
                gm2.plt.xlabel("time [ns]")
                gm2.plt.ylabel("frequency [Hz]")
                if not np.isnan(rise_time[probe]):
                    gm2.plt.plot([rise_time[probe]],[th],'.')
                    gm2.plt.xlim([rise_time[probe]-50e9, rise_time[probe]+50e9])
                gm2.despine()
                gm2.plt.show()
        return rise_time 

    def loadTrMultipoles(self, n=9, probe_=8, freqs = None):
        if freqs is None:
            freqs = np.array([self.tr.freqAtTime[probe](self.tr.time[:,probe_]) for probe in range(17)]).T
        self.tr_mp = np.zeros([freqs.shape[0], n], dtype='float')
        for ev in np.arange(freqs.shape[0]):
            if ev%100==0:
                print("\rCalculate Trolley Multipoles % 3.2f" % (100.0*ev/freqs.shape[0]), "%", end=' ')
            self.tr_mp[ev, :] = gm2.util.getTrMultipole(freqs[ev,:].T, n)
        print("")
        self.tr_mpAt = []
        for mp in range(n):
            self.tr_mpAt.append(gm2.util.interp1d(self.tr.time[:,probe_], self.tr_mp[:, mp]))
        return self.tr_mp

    def loadFpMultipoles(self, yokes, azis, n=3, useWeight=False):
        for yoke in yokes:
            for azi in azis:
                self.fp_mp = {yoke : {azi : np.array([self.getFpMultipole(yoke, azi, ev, n, useWeight) for ev in range(self.fp.freq.shape[0])]) }}
        if ((len(yokes)==1)&(len(azis)==1)):
            return self.fp_mp[yoke][azi]

    def getFpMultipole(self, yoke, azi, ev, n=3, useWeight=True):
        if (useWeight & (not hasattr(self, 'fp_res'))):
            print("Loading fixed probe resolution")
            self.loadFpRes()
        s = self.fp.select(yokes=[yoke], aziIds=[azi])
        station_n = np.argwhere(s).shape[0]
        pos_      = (self.fp.pos_r[s], self.fp.pos_theta[s])
        freq_     = self.fp.freq[ev, s]
        if not useWeight:
            return gm2.util.getFpMultipole(pos_, freq_, n=n)
        else:
            return gm2.util.getFpMultipole(pos_, freq_, sigma=self.fp_res[s], n=n)

    def loadTrySpikes(self, skip=0, threshold = gm2.PPM_HZ/2.):
        if skip > 0:
            self.spk = [gm2.Spikes(self.tr.azi[skip:-skip,probe], self.tr.freq[skip:-skip, probe], threshold) for probe in range(17)]
        else:
            self.spk = [gm2.Spikes(self.tr.azi[:,probe], self.tr.freq[:, probe], threshold) for probe in range(17)]
        self.trFreqSmooth = np.array([self.spk[p].freq for p in range(17)]).T

