import gm2
from gm2 import plt, np
from gm2.plotutil import plot_ts
import ROOT

#runs = np.arange(5809, 5835+1) #[5807]
#runs = np.arange(5809, 5811+1) #[5807]
runs = [5217]


class Interpolation(object):
    def __init__(self, runs):
        self.fp = gm2.FixedProbe(runs)
        self.tr = gm2.Trolley(runs)
        self.load();

    def load(self, freq_type=0):
        def fp_callback():
            return [self.fp.getTimeGPS(), self.fp.getFrequency(freq_type)]
        self.fp_time, self.fp_freq = self.fp.loop(fp_callback)

        def tr_callback():
            return [self.tr.getTimeGPS(), self.tr.getPhi(0), self.tr.getFrequency(freq_type) ]
        self.tr_time, self.tr_pos, self.tr_freq  = self.tr.loop(tr_callback)

    def loadFpRes(self):
        h = []
        res_simon = np.full_like(self.fp_freq, np.nan)
        #res_ran   = np.full_like(fp_freq, np.nan)
        fp_res = []
        for probe in np.arange(self.fp.n_probes):
            print("resolution probe ", probe)
            h.append(ROOT.TH1F("h"+str(probe), "h"+str(probe), 10001, -500, 500))
            for event, freq in enumerate(self.fp_freq[:1000,probe]):
                if (event - 1 >= 0)&(event+1 < self.fp_freq.shape[0] ):
                    res_simon[event, probe] = freq - (self.fp_freq[event-1, probe] + self.fp_freq[event+1, probe])/2.0
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



inter = Interpolation(runs)

inter.loadFpRes()
tr_times = inter.getFpTrlyTimes()

event = np.empty_like(tr_times, dtype='int')
for probe in np.arange(inter.fp.n_probes):
    event[probe] = gm2.util.nearestIndex(inter.fp_time[:,0], tr_times[probe])



s_ti = (fp.id['layer'] == ord('T'))&(fp.id['rad']==ord('I'))
s_tm = (fp.id['layer'] == ord('T'))&(fp.id['rad']==ord('M'))
s_to = (fp.id['layer'] == ord('T'))&(fp.id['rad']==ord('O'))
s_bi = (fp.id['layer'] == ord('B'))&(fp.id['rad']==ord('I'))
s_bm = (fp.id['layer'] == ord('B'))&(fp.id['rad']==ord('M'))
s_bo = (fp.id['layer'] == ord('B'))&(fp.id['rad']==ord('O'))

##### Station Mean #####
'''
s_ti = (fp.id['layer'] == ord('T'))&(fp.id['rad']==ord('I'))
s_tm = (fp.id['layer'] == ord('T'))&(fp.id['rad']==ord('M'))
s_to = (fp.id['layer'] == ord('T'))&(fp.id['rad']==ord('O'))
s_bi = (fp.id['layer'] == ord('B'))&(fp.id['rad']==ord('I'))
s_bm = (fp.id['layer'] == ord('B'))&(fp.id['rad']==ord('M'))
s_bo = (fp.id['layer'] == ord('B'))&(fp.id['rad']==ord('O'))

s_pos = [ s_ti, s_tm, s_to, s_bi, s_bm, s_bo ]

s_station = (fp.id['yoke'] == ord('A'))&(fp.id['azi']==1)
station_n = np.argwhere(s_station).shape[0]


dipole_sum   = np.zeros(fp_freq.shape[0], dtype='float')
dipole_wsum  = 0.0 
dipole_wsum2 = 0.0 
dipole_n = 0
for s in s_pos:
    s_ = s_station&s
    if fp_freq[:,s_].shape[1] > 0:
        # probe present
        w = 1.0/fp_res[s_]**2
        dipole_sum    = dipole_sum + fp_freq[:,s_][:,0]*w
        dipole_wsum  += w
        dipole_wsum2 += w**2
        dipole_n += 1
if dipole_n > 0:
    dipole     = dipole_sum/dipole_wsum
    dipole_res = 1.0/np.sqrt(dipole_wsum2)
else:
    dipole     = np.nan
    dipole_res = np.nan
'''

##### Multipole Fits #####
from gm2.util import getFpMultipole


mp_1_no  = np.full([12*6, 5], np.nan)
mp_4_no  = np.full([12*6, 5], np.nan)
mp_a_no  = np.full([12*6, 5], np.nan)
mp_1     = np.full([12*6, 5], np.nan)
mp_4     = np.full([12*6, 5], np.nan)
mp_a     = np.full([12*6, 5], np.nan)
mp_1_res = np.full([12*6, 5], np.nan)
mp_4_res = np.full([12*6, 5], np.nan)
mp_a_res = np.full([12*6, 5], np.nan)

ref_bm   = np.full([12*6], np.nan)
ref_tm   = np.full([12*6], np.nan)

ref_bm_res   = np.full([12*6], np.nan)
ref_tm_res   = np.full([12*6], np.nan)

multipoles_1_no  = np.full([12*6, inter.fp_freq.shape[0], 5], np.nan)
multipoles_4_no  = np.full([12*6, inter.fp_freq.shape[0], 5], np.nan)
multipoles_a_no  = np.full([12*6, inter.fp_freq.shape[0], 5], np.nan)
multipoles_1     = np.full([12*6, inter.fp_freq.shape[0], 5], np.nan)
multipoles_4     = np.full([12*6, inter.fp_freq.shape[0], 5], np.nan)
multipoles_a     = np.full([12*6, inter.fp_freq.shape[0], 5], np.nan)
multipoles_1_res = np.full([12*6, inter.fp_freq.shape[0], 5], np.nan)
multipoles_4_res = np.full([12*6, inter.fp_freq.shape[0], 5], np.nan)
multipoles_a_res = np.full([12*6, inter.fp_freq.shape[0], 5], np.nan)


def calcStationMultipoles(yoke, azi, ev, n):
#def calcStationMultipoles(yoke, azi):
    s_station = (fp.id['yoke'] == yoke)&(fp.id['azi']==azi)
    station_n = np.argwhere(s_station).shape[0]
    pos_   = (fp.pos_r[s_station], fp.pos_theta[s_station])
    freq_  = inter.fp_freq[ev, s_station] 
    sigma_ = fp_res[s_station]
    multipoles_1_no[n,ev,:1]                                = getFpMultipole(pos_, freq_, n= 2) # only dipole
    multipoles_4_no[n,ev,:3]                                = getFpMultipole(pos_, freq_, n= 4) # only dipole and quad
    multipoles_a_no[n,ev,:station_n-1]                      = getFpMultipole(pos_, freq_, n=-1)
    multipoles_1[n,ev,:1], multipoles_1_res[n, ev, :1]                     = getFpMultipole(pos_, freq_, sigma=sigma_, n= 2) # only dipole
    multipoles_4[n,ev,:3], multipoles_4_res[n, ev, :3]                     = getFpMultipole(pos_, freq_, sigma=sigma_, n= 4)
    multipoles_a[n,ev,:station_n-1], multipoles_a_res[n, ev, :station_n-1] = getFpMultipole(pos_, freq_, sigma=sigma_, n=-1)
        #ma[ev,:station_n-1], mar[ev,:station_n-1] = getFpMultipole(pos_, freq_, sigma=sigma_, n=-1)
    #return multipoles_1_no, multipoles_4_no, multipoles_a_no, multipoles_1, multipoles_4, multipoles_a, multipoles_1_res, multipoles_4_res, multipoles_a_res


station_id  = []
station_phi = []
station_nn  = []
station_event = []
#import threading
threads = []
for yoke in np.arange(ord('A'), ord('A')+12):
    for azi in np.arange(1, 7):
        station_id.append((yoke-ord('A'))*10 + azi)
        s_station = (fp.id['yoke'] == yoke)&(fp.id['azi']==azi)
        station_phi.append((np.arctan2(fp.getY(), fp.getX())[s_station][0]/np.pi*180 + 360) % 360.)
        station_nn.append(np.argwhere(s_station).shape[0])
        station_event.append(event[s_station][0]-100)
        n = len(station_id)-1
        if n in []:
            for ev in np.arange(inter.fp_freq.shape[0]):
                if ev%1000==0:
                    print(chr(yoke), azi, ev)
                calcStationMultipoles(yoke, azi, ev, n) 
        for ev in [station_event[n]]:
            calcStationMultipoles(yoke, azi, ev, n)
            mp_1_no[n, :] = multipoles_1_no[n, ev, :]
            mp_4_no[n, :] = multipoles_4_no[n, ev, :]
            mp_a_no[n, :] = multipoles_a_no[n, ev, :]
            mp_1[n, :]    = multipoles_1[n, ev, :]
            mp_4[n, :]    = multipoles_4[n, ev, :]
            mp_a[n, :]    = multipoles_a[n, ev, :]
            mp_1_res[n, :] = multipoles_1_res[n, ev, :]
            mp_4_res[n, :] = multipoles_4_res[n, ev, :]
            mp_a_res[n, :] = multipoles_a_res[n, ev, :]

            ref_bm[n]     = inter.fp_freq[ev, s_station&s_bm][0]
            ref_tm[n]     = inter.fp_freq[ev, s_station&s_tm][0]
            ref_bm_res[n] = inter.fp_res[s_station&s_bm][0]
            ref_tm_res[n] = inter.fp_res[s_station&s_tm][0]

'''
station = 0
#plt.plot(dipole, '.',              alpha=0.7, label="dipole")
plt.plot(multipoles_1[station, 2:,0],    alpha=0.7, label="dipole fit (1)")
plt.plot(multipoles_4[station, 2:,0],    alpha=0.7, label="dipole fit (4)")
plt.plot(multipoles_a[station, 2:,0],    alpha=0.7, label="dipole fit (all)")
plt.plot(multipoles_1_no[station, 2:,0], alpha=0.7, label="dipole fit (1, no w)")
plt.plot(multipoles_4_no[station, 2:,0], alpha=0.7, label="dipole fit (4, no w)")
plt.plot(multipoles_a_no[station, 2:,0], alpha=0.7, label="dipole fit (all, no w)")
gm2.despine()
plt.legend()
plt.show()


ev = 100
#plt.plot(dipole, '.',              alpha=0.7, label="dipole")
station_nn = np.array(station_nn)
s_a = station_nn==6
station_id = np.array(station_id)
station_phi = np.array(station_phi)
plt.plot((tr_phi[10:,0]/np.pi*180 +360.) % 360., tr_freq[10:,0], '.k', markersize=1)
plt.errorbar(station_phi, multipoles_1[:, ev, 0], yerr=multipoles_1_res[:, ev, 0],    fmt='--', alpha=0.7, label="dipole fit (1)")
plt.errorbar(station_phi, multipoles_4[:, ev, 0], yerr=multipoles_4_res[:, ev, 0],   fmt='--', alpha=0.7, label="dipole fit (4)")
plt.errorbar(station_phi[s_a], multipoles_a[s_a, ev, 0], yerr=multipoles_a_res[s_a, ev, 0],   fmt='--', alpha=0.7, label="dipole fit (all)")
plt.plot(station_phi, multipoles_1_no[:, ev, 0], '.-', alpha=0.7, label="dipole fit (1, no w)")
plt.plot(station_phi, multipoles_4_no[:, ev, 0], '.-', alpha=0.7, label="dipole fit (4, no w)")
plt.plot(station_phi[s_a], multipoles_a_no[s_a, ev, 0], '.-', alpha=0.7, label="dipole fit (all, no w)")
gm2.despine()
plt.legend()
plt.show()
'''


ev = 100
#plt.plot(dipole, '.',              alpha=0.7, label="dipole")
station_nn = np.array(station_nn)
s_a = station_nn==6
station_id = np.array(station_id)
station_phi = np.array(station_phi)
plt.plot((tr_phi[10:,0]/np.pi*180 +360.) % 360., tr_freq[10:,0], '.g', markersize=1)
plt.plot((tr_phi[10:,:].mean(axis=1)/np.pi*180 +360.) % 360., tr_freq[10:,:].mean(axis=1), '.k', markersize=1)
plt.errorbar(station_phi,      mp_1[:,   0],   yerr=mp_1_res[:, 0],   fmt='x', alpha=0.7, label="dipole fit (1)")
plt.errorbar(station_phi,      mp_4[:,   0],   yerr=mp_4_res[:, 0],   fmt='x', alpha=0.7, label="dipole fit (4)")
#plt.errorbar(station_phi[s_a], mp_a[s_a, 0], yerr=mp_a_res[s_a, 0], fmt='--', alpha=0.7, label="dipole fit (all)")
#plt.plot(station_phi,          mp_1_no[:,   0],                               '.-', alpha=0.7, label="dipole fit (1, no w)")
#plt.plot(station_phi,          mp_4_no[:,   0],                               '.-', alpha=0.7, label="dipole fit (4, no w)")
#plt.plot(station_phi[s_a],     mp_a_no[s_a, 0],                             '.-', alpha=0.7, label="dipole fit (all, no w)")

plt.errorbar(station_phi, ref_tm, yerr=ref_tm_res, fmt='v', alpha=0.7, label="tm")
plt.errorbar(station_phi, ref_bm, yerr=ref_bm_res, fmt='v', alpha=0.7, label="bm")

gm2.despine()
plt.legend()
plt.show()
