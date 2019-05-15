import gm2
from gm2 import plt, np
from gm2.plotutil import plot_ts
import ROOT

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



'''
Accordiing to db
# 3928 Yoke G, Azi 1 
# 3930 Yoke G, Azi 2
# 3932 Yoke G, Azi 3
# 2410-2417, 187.1
  -> active at 185.1 -> yoke G, Azi 5

BUT!
Azi 1: 1524643000000 + 520000, peak 172.7 -> active 170.2
Azi 2: 1525653000000 + 480000, peak 176.0 -> active 173.5
Azi 3: 1524643000000 + 400000, peak 182.7 -> active 180.2

File 3928, trolley at 172.8, closest probe is Azi 2

File 2411, trolley at 187.1 -> yoke G, Azi 5

File 3751 to 3760, trolley at 176.929099 -> yoke G, Azi 2
'''


runs = [3928]# 3830, 3832
yoke = 'G'
azi = 2

runs = [3930]# 3830, 3832
yoke = 'G'
azi = 3

runs = np.arange(3751+3, 3760+1) # 5h
yoke = 'G'
azi = 3

inter = Interpolation(runs)

inter.loadFpRes()


s_ti = (inter.fp.id['layer'] == ord('T'))&(inter.fp.id['rad']==ord('I'))
s_tm = (inter.fp.id['layer'] == ord('T'))&(inter.fp.id['rad']==ord('M'))
s_to = (inter.fp.id['layer'] == ord('T'))&(inter.fp.id['rad']==ord('O'))
s_bi = (inter.fp.id['layer'] == ord('B'))&(inter.fp.id['rad']==ord('I'))
s_bm = (inter.fp.id['layer'] == ord('B'))&(inter.fp.id['rad']==ord('M'))
s_bo = (inter.fp.id['layer'] == ord('B'))&(inter.fp.id['rad']==ord('O'))


##### Multipole Fits #####
from gm2.util import getFpMultipole


#mp_1_no  = np.full([12*6, 5], np.nan)
#mp_4_no  = np.full([12*6, 5], np.nan)
#mp_a_no  = np.full([12*6, 5], np.nan)
#mp_1     = np.full([12*6, 5], np.nan)
#mp_4     = np.full([12*6, 5], np.nan)
#mp_a     = np.full([12*6, 5], np.nan)
#mp_1_res = np.full([12*6, 5], np.nan)
#mp_4_res = np.full([12*6, 5], np.nan)
#mp_a_res = np.full([12*6, 5], np.nan)
#ref_bm   = np.full([12*6], np.nan)
#ref_tm   = np.full([12*6], np.nan)
#ref_bm_res   = np.full([12*6], np.nan)
#ref_tm_res   = np.full([12*6], np.nan)

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
    s_station = (inter.fp.id['yoke'] == yoke)&(inter.fp.id['azi']==azi)
    station_n = np.argwhere(s_station).shape[0]
    pos_   = (inter.fp.pos_r[s_station], inter.fp.pos_theta[s_station])
    freq_  = inter.fp_freq[ev, s_station] 
    sigma_ = inter.fp_res[s_station]
    multipoles_1_no[n,ev,:1]                                = getFpMultipole(pos_, freq_, n= 2) # only dipole
    multipoles_4_no[n,ev,:3]                                = getFpMultipole(pos_, freq_, n= 4) # only dipole and quad
    multipoles_a_no[n,ev,:station_n-1]                      = getFpMultipole(pos_, freq_, n=-1)
    multipoles_1[n,ev,:1], multipoles_1_res[n, ev, :1]                     = getFpMultipole(pos_, freq_, sigma=sigma_, n= 2) # only dipole
    multipoles_4[n,ev,:3], multipoles_4_res[n, ev, :3]                     = getFpMultipole(pos_, freq_, sigma=sigma_, n= 4)
    multipoles_a[n,ev,:station_n-1], multipoles_a_res[n, ev, :station_n-1] = getFpMultipole(pos_, freq_, sigma=sigma_, n=-1)
        #ma[ev,:station_n-1], mar[ev,:station_n-1] = getFpMultipole(pos_, freq_, sigma=sigma_, n=-1)
    #return multipoles_1_no, multipoles_4_no, multipoles_a_no, multipoles_1, multipoles_4, multipoles_a, multipoles_1_res, multipoles_4_res, multipoles_a_res


#station_id  = []
#station_phi = []
#station_nn  = []
#station_event = []
#import threading
for ev in np.arange(inter.fp_freq.shape[0]):
    if ev%500==0:
        print(ev)
    calcStationMultipoles(ord(yoke), azi, ev, 0)


s_tr = (inter.tr_time[:,0] > 1e18)&(inter.tr_freq[:,0]>52500)&(inter.tr_freq[:,0]<52900)
tr_dipole = gm2.util.interp1d(inter.tr_time[s_tr,0], inter.tr_freq[s_tr,0])
s_station = (inter.fp.id['yoke'] == ord(yoke))&(inter.fp.id['azi']==azi)

s_fp = inter.fp_time[10:-10,0] > 1e18

plt.subplot(331)
plt.plot(tr_dipole(inter.fp_time[10:-10,s_station][:,0]), multipoles_1[0,10:-10,0], '.', label="(1)")
c = np.corrcoef(tr_dipole(inter.fp_time[10:-10,s_station][:,0]), multipoles_1[0,10:-10,0])
plt.title("(1)")
plt.text(0.3, 0.85, ("%.5f" % c[1, 0]),
                 horizontalalignment='center',
                 verticalalignment='center',
                 transform = plt.gca().transAxes)
plt.gca().set_xticklabels([])
plt.gca().set_yticklabels([])

plt.subplot(332)
plt.plot(tr_dipole(inter.fp_time[10:-10,s_station][:,0]), multipoles_4[0,10:-10,0], '.', label="(4)")
c = np.corrcoef(tr_dipole(inter.fp_time[10:-10,s_station][:,0]), multipoles_4[0,10:-10,0])
plt.title("(2)")
plt.text(0.3, 0.85, ("%.5f" % c[1, 0]),
                 horizontalalignment='center',
                 verticalalignment='center',
                 transform = plt.gca().transAxes)
plt.gca().set_xticklabels([])
plt.gca().set_yticklabels([])

plt.subplot(333)
plt.plot(tr_dipole(inter.fp_time[10:-10,s_station][:,0]), multipoles_a[0,10:-10,0], '.', label="(1, no w)")
c = np.corrcoef(tr_dipole(inter.fp_time[10:-10,s_station][:,0]), multipoles_a[0,10:-10,0])
plt.title("(6)")
plt.text(0.3, 0.85, ("%.5f" % c[1, 0]),
                 horizontalalignment='center',
                 verticalalignment='center',
                 transform = plt.gca().transAxes)
plt.gca().set_xticklabels([])
plt.gca().set_yticklabels([])

plt.subplot(334)
plt.plot(tr_dipole(inter.fp_time[10:-10,s_station][:,0]), multipoles_1_no[0,10:-10,0], '.', label="(1)")
c = np.corrcoef(tr_dipole(inter.fp_time[10:-10,s_station][:,0]), multipoles_1_no[0,10:-10,0])
plt.title("(1, no w)")
plt.text(0.3, 0.85, ("%.5f" % c[1, 0]),
                 horizontalalignment='center',
                 verticalalignment='center',
                 transform = plt.gca().transAxes)
plt.gca().set_xticklabels([])
plt.gca().set_yticklabels([])

plt.subplot(335)
plt.plot(tr_dipole(inter.fp_time[10:-10,s_station][:,0]), multipoles_4_no[0,10:-10,0], '.', label="(4)")
c = np.corrcoef(tr_dipole(inter.fp_time[10:-10,s_station][:,0]), multipoles_4_no[0,10:-10,0])
plt.title("(2, no w)")
plt.text(0.3, 0.85, ("%.5f" % c[1, 0]),
                 horizontalalignment='center',
                 verticalalignment='center',
                 transform = plt.gca().transAxes)
plt.gca().set_xticklabels([])
plt.gca().set_yticklabels([])

plt.subplot(336)
plt.plot(tr_dipole(inter.fp_time[10:-10,s_station][:,0]), multipoles_a_no[0,10:-10,0], '.', label="(1, no w)")
c = np.corrcoef(tr_dipole(inter.fp_time[10:-10,s_station][:,0]), multipoles_a_no[0,10:-10,0])
plt.title("(6, no w)")
plt.text(0.3, 0.85, ("%.5f" % c[1, 0]),
                 horizontalalignment='center',
                 verticalalignment='center',
                 transform = plt.gca().transAxes)
plt.gca().set_xticklabels([])
plt.gca().set_yticklabels([])

plt.subplot(337)
plt.plot(tr_dipole(inter.fp_time[10:-10,s_station&s_tm][:,0]), inter.fp_freq[10:-10,s_station&s_tm], '.', label="(4, no w)")
c = np.corrcoef(tr_dipole(inter.fp_time[10:-10,s_station&s_tm][:,0]), inter.fp_freq[10:-10,s_station&s_tm][:,0])
plt.title("(TM")
plt.text(0.3, 0.85, ("%.5f" % c[1, 0]),
                 horizontalalignment='center',
                 verticalalignment='center',
                 transform = plt.gca().transAxes)
plt.gca().set_xticklabels([])
plt.gca().set_yticklabels([])

plt.subplot(338)
plt.plot(tr_dipole(inter.fp_time[10:-10,s_station&s_tm][:,0]), inter.fp_freq[10:-10,s_station&s_bm], '.', label="(1, no w)")
c = np.corrcoef(tr_dipole(inter.fp_time[10:-10,s_station&s_tm][:,0]), inter.fp_freq[10:-10,s_station&s_bm][:,0])
plt.title("BM")
plt.text(0.3, 0.85, ("%.5f" % c[1, 0]),
                 horizontalalignment='center',
                 verticalalignment='center',
                 transform = plt.gca().transAxes)
#plt.gca().set_xticklabels([])
plt.gca().set_yticklabels([])

plt.subplot(339)
plt.plot(tr_dipole(inter.fp_time[10:-10,s_station&s_tm][:,0]), (inter.fp_freq[10:-10,s_station&s_tm] + inter.fp_freq[10:-10,s_station&s_bm])/2., '.', label="(4, no w)")
c = np.corrcoef(tr_dipole(inter.fp_time[10:-10,s_station&s_tm][:,0]), (inter.fp_freq[10:-10,s_station&s_tm] + inter.fp_freq[10:-10,s_station&s_bm])[:,0]/2.)
plt.title("TM+BM")
plt.text(0.3, 0.85, ("%.5f" % c[1, 0]),
                 horizontalalignment='center',
                 verticalalignment='center',
                 transform = plt.gca().transAxes)
plt.gca().set_xticklabels([])
plt.gca().set_yticklabels([])

gm2.sns.despine()
plt.show()
