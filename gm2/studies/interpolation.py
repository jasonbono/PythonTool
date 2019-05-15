import gm2
import numpy as np
from gm2 import plt

class interpolation(object):
    def __init__(self, runs=[]):
        self.fp = gm2.FixedProbe(runs)
        self.tr = gm2.Trolley(runs)
        self.load();
    
    def load(self, freq_type = 0):
        def fp_callback():
            return [self.fp.getTimeGPS(), self.fp.getFrequency(freq_type)]
        self.fp_time, self.fp_freq = self.fp.loop(fp_callback) 
        
        def tr_callback():
            return [self.tr.getTimeGPS(), self.tr.getPhi(0), self.tr.getFrequency(freq_type), self.tr.getBarcodeTrace(), self.tr.getBarcodeTs(), self.tr.getTimeBarcodes()]
        self.tr_time, self.tr_pos, self.tr_freq, bc_trcs, bc_ts, bc_t = self.tr.loop(tr_callback)
        bc_s = (bc_trcs[:,0,:].reshape([-1])>1.0)&(bc_trcs[:,0,:].reshape([-1])<5.0)
        self.bc_clk = [bc_trcs[:,1,:].reshape([-1])[bc_s],
                       bc_trcs[:,4,:].reshape([-1])[bc_s]]
        self.bc_abs = [bc_trcs[:,0,:].reshape([-1])[bc_s],
                       bc_trcs[:,3,:].reshape([-1])[bc_s]]
        if(bc_ts.shape[0]>2):
            probe_ = 0
            s = (bc_t[1:-1,probe_] > 1e5)&(self.tr_time[1:-1, probe_] > 1e5)
            popt_ = gm2.util.fit_lin(bc_t[1:-1,probe_][s], self.tr_time[1:-1, probe_][s], p0=[self.tr_time[1:-1, probe_][s].min(), 1e6])
            if True:
                plt.subplot(211)
                plt.plot(self.tr_time[1:-1, probe_][s], bc_t[1:-1,probe_][s],'.')
                plt.plot(gm2.util.func_lin(bc_t[1:-1,probe_],popt_[0],popt_[1]), bc_t[1:-1,probe_])
                plt.xlabel("gps times [ns]")
                plt.ylabel("barcode times [ns]")
                plt.subplot(212)
                plt.plot(bc_t[1:-1,probe_][s], gm2.util.func_lin(bc_t[1:-1,probe_][s],popt_[0],popt_[1]) - self.tr_time[1:-1, probe_][s])
                plt.xlabel("gps time [ns]")
                plt.ylabel("residuals [ns]")
                gm2.despine()
                plt.show()
                print(popt_)
            if(abs(popt_[1] - 1.0) > 0.001):
                raise ValueError('barcode timestamp conversion failed')
        else:
            popt_ = [0, 1e6]
        self.bc_time = bc_ts.reshape([-1])[bc_s] * popt_[1] + popt_[0]

    def getStationId(self):
        return (self.fp.id['yoke'] - 65) * 10 + self.fp.id['azi']

    def getFpTrlyTimes(self, cw=False, plot=[]):
        ''' loads the time the trolley passes by the individuall fixed probe'''
        skip = 1
        rise_time   = np.full(self.fp.n_probes, np.nan)
        for probe in np.arange(self.fp.n_probes):
            if cw:
                rise_time[probe], th   = gm2.util.cf_local(np.flip(self.fp_time[skip:, probe]), np.flip(self.fp_freq[skip:, probe]), returnTh=True, baseline_o=10)
            else:
                rise_time[probe], th   = gm2.util.cf_local(self.fp_time[skip:, probe], self.fp_freq[skip:, probe], returnTh=True, baseline_o=10)
            if probe in plot:
                plt.plot(self.fp_time[skip:, probe], self.fp_freq[skip:, probe])
                plt.title("Probe "+str(probe))
                plt.xlabel("time [ns]")
                plt.ylabel("frequency [Hz]")
                if not np.isnan(rise_time[probe]):
                    plt.plot([rise_time[probe]],[th],'.')
                    plt.xlim([rise_time[probe]-50e9, rise_time[probe]+50e9])
                gm2.despine()
                plt.show()
        return rise_time

    def getTrlyPosAtTime(self, time, probe=0):
        index_next_ = np.argwhere(self.tr_time[:, probe] > time)
        if (index_next_.shape[0] == 0):
            return np.nan
        index_next = index_next_[0]
        index_prev = index_next-1
        dt   = self.tr_time[index_next, probe] - self.tr_time[index_prev, probe]
        dpos = self.tr_pos[index_next, probe]  - self.tr_pos[index_prev, probe]
        return self.tr_pos[index_prev, probe] + dpos/dt * (time - self.tr_time[index_prev, probe])


inter_ccw      = interpolation([5217]);
rise_times_ccw = inter_ccw.getFpTrlyTimes(plot=[0])
trly_pos_ccw   = np.full(inter_ccw.fp.n_probes, np.nan)
for probe in np.arange(inter_ccw.fp.n_probes):
    trly_pos_ccw[probe] = inter_ccw.getTrlyPosAtTime(rise_times_ccw[probe])

inter_cw = interpolation([5216, 5218]);
rise_times_cw = inter_cw.getFpTrlyTimes(cw=True, plot=[0])
trly_pos_cw   = np.full(inter_cw.fp.n_probes, np.nan)
for probe in np.arange(inter_cw.fp.n_probes):
    trly_pos_cw[probe] = inter_cw.getTrlyPosAtTime(rise_times_cw[probe])


ids = inter_ccw.getStationId()
dt_ccw = []
dt_cw  = []
dp_ccw = []
dp_cw  = []
for id_ in np.unique(ids):
    dt_ccw.append(rise_times_ccw[np.argwhere(ids == id_)].std())
    dt_cw.append( rise_times_cw[ np.argwhere(ids == id_)].std())

    dp_ccw.append(trly_pos_ccw[np.argwhere(ids == id_)].std() * gm2.constants.R)
    dp_cw.append( trly_pos_cw[ np.argwhere(ids == id_)].std() * gm2.constants.R)

plt.plot(np.unique(ids), dt_ccw, '.', label='CCW')
plt.plot(np.unique(ids), dt_cw,  '.', label='CW')
plt.xlabel("fp station [yoke * 10 + radiId]")
plt.ylabel("time spread [ns]")
plt.legend()
gm2.despine()
plt.show()

plt.plot(np.unique(ids), dp_ccw, '.', label='CCW')
plt.plot(np.unique(ids), dp_cw,  '.', label='CW')
plt.xlabel("fp station [yoke * 10 + radiId]")
plt.ylabel("position spread [mm]")
plt.legend()
gm2.despine()
plt.show()

dt = 5e9
tt = -0.0e9
freq_cut = 0.001
from scipy import signal
b, a = signal.butter(3, freq_cut)
for probe in np.arange(1):
    s_ccw = (inter_ccw.bc_time > rise_times_ccw[probe] - dt + tt) & (inter_ccw.bc_time < rise_times_ccw[probe] + dt + tt)
    s_cw  = (inter_cw.bc_time  > rise_times_cw[probe]  - dt + tt) & (inter_cw.bc_time  < rise_times_cw[probe]  + dt + tt)

    y_ccw = [signal.filtfilt(b, a, inter_ccw.bc_clk[0][s_ccw]), 
             signal.filtfilt(b, a, inter_ccw.bc_clk[1][s_ccw])]
    zeros_ccw = [np.where(np.diff(np.sign(np.diff(y_ccw[0], n=1))))[0], 
                 np.where(np.diff(np.sign(np.diff(y_ccw[1], n=1))))[0]]
    y_cw =  [signal.filtfilt(b, a, inter_cw.bc_clk[0][s_cw]), 
             signal.filtfilt(b, a, inter_cw.bc_clk[1][s_cw])]
    zeros_cw  = [np.where(np.diff(np.sign(np.diff(y_cw[0], n=1))))[0], 
                 np.where(np.diff(np.sign(np.diff(y_cw[1], n=1))))[0]]
    plt.plot((  inter_ccw.bc_time[s_ccw] - rise_times_ccw[probe] + tt)[1:-1],      inter_ccw.bc_abs[0][s_ccw][1:-1])
    plt.plot((  inter_ccw.bc_time[s_ccw] - rise_times_ccw[probe] + tt)[zeros_ccw[0]], inter_ccw.bc_abs[0][s_ccw][zeros_ccw[0]],'.')

    plt.plot((-(inter_cw.bc_time[s_cw]   - rise_times_cw[probe] + tt))[1:-1],      inter_cw.bc_abs[0][s_cw][1:-1])
    plt.plot((-(inter_cw.bc_time[s_cw]   - rise_times_cw[probe] + tt))[zeros_cw[0]], inter_cw.bc_abs[0][s_cw][zeros_cw[0]],'.')

    ccw_v = inter_ccw.bc_abs[0][s_ccw][zeros_ccw[0]]
    ccw_v_l = ccw_v.shape[-1]
    cw_v  = inter_cw.bc_abs[0][s_cw][zeros_cw[0]]
    cw_v_l = cw_v.shape[-1]

    n_skip = 1
    n_min = np.min([ccw_v_l, cw_v_l])
    chi2  = []
    chi22 = []
    for n in np.arange(n_skip, n_min-n_skip):
        chi2.append(((ccw_v[:n]  - np.flip(cw_v[:n]))**2).sum()/n)
        chi22.append(((ccw_v[-n:] - np.flip(cw_v[-n:]))**2).sum()/n)
    plt.plot(chi2)    
    plt.plot(chi22)    
    plt.show()

    plt.plot(inter_ccw.bc_abs[0][s_ccw][zeros_ccw[0]][-42:],'.')
    plt.plot(np.flip(inter_cw.bc_abs[0][s_cw][zeros_cw[0]][-42:]),'.')
    plt.show()

    plt.plot(inter_ccw.bc_abs[0][s_ccw][zeros_ccw[0]][:37],'.')
    plt.plot(np.flip(inter_cw.bc_abs[0][s_cw][zeros_cw[0]][:37]),'.')
    plt.show()
    #plt.plot(  np.arange(n_min),    inter_ccw.bc_abs[0][s_ccw][zeros_ccw[0]][:n_min],'.')
    #plt.plot( (np.arange(n_min)),   np.flip(inter_cw.bc_abs[0][s_cw][zeros_cw[0]][:n_min]),'.')

    plt.xlabel("time [ps]")
    plt.ylabel("barcode abs")
    gm2.despine()
    plt.show()





'''
import matplotlib.pyplot as plt

for idd in np.unique(ccw[0]):
    s = ccw[0] == idd
    plt.errorbar([ccw[0][s]], [ccw[3][s].mean()], yerr=ccw[3][s].std(), fmt=' ', color=gm2.sns.color_palette()[0])
    s = cw[0] == idd
    plt.errorbar([cw[0][s]], [cw[3][s].mean()], yerr=cw[3][s].std(), fmt=' ', color=gm2.sns.color_palette()[1])
#plt.plot(cw[0],  cw[3],  '.', label="CW")
plt.xlabel("fixed probe station [Yoke * 10 + radId]")
plt.ylabel("trolley mean [Hz]")
gm2.despine()
plt.title("runs 5216-5218")
plt.show()


ddd = []
for idd in np.unique(ccw[0]):
    s = ccw[0] == idd
    s = cw[0] == idd
    ddd.append(ccw[3][s].mean() - cw[3][s].mean())
    plt.errorbar([ccw[0][s]], [ccw[3][s].mean() - cw[3][s].mean()], yerr=np.sqrt(ccw[3][s].std() + cw[3][s].std()), fmt=' ', color=gm2.sns.color_palette()[0])
    #plt.errorbar([cw[0][s]], [cw[3][s].mean()], yerr=cw[3][s].std(), fmt=' ', color=gm2.sns.color_palette()[1])
#plt.plot(cw[0],  cw[3],  '.', label="CW")
plt.xlabel("fixed probe station [Yoke * 10 + radId]")
plt.ylabel("trolley mean: difference [Hz]")
gm2.despine()
plt.title("runs 5216-5218")
plt.show()

plt.hist(ddd, bins=np.arange(-300, 300, 20))
plt.xlabel("trolley mean: difference [Hz]")
gm2.despine()
plt.show()
'''
