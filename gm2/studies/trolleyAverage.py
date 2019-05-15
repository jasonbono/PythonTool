import gm2
import numpy as np


def getMeans(runs, cw = False):
    fp = gm2.FixedProbe(runs)
    def fp_callback():
        return [np.array([fp.getYoke(), fp.getAziId(), fp.getLayer(), fp.getRadId()]), fp.getTimeGPS(), fp.getFrequency(0)]
    fp_id_, fp_time, fp_freq = fp.loop(fp_callback)

    tr = gm2.Trolley(runs)
    def tr_callback():
        return [tr.getTimeGPS(), tr.getPhi(0), tr.getFrequency(0)]
    tr_time, tr_pos, tr_freq = tr.loop(tr_callback)

    ## Settings
    tr_pos_offset = 3.06/180.*np.pi # 380mm?
    tr_azi_avg    = 0.5/180.*np.pi


    fp_probe_n = fp_freq.shape[1] 
    skip = 1

    fp_id       = np.full(fp_probe_n, np.nan)
    rise_time   = np.full(fp_probe_n, np.nan)
    rise_tr_pos = np.full(fp_probe_n, np.nan)
    tr_mean     = np.full(fp_probe_n, np.nan)
    for probe in np.arange(fp_probe_n):
        fp_id[probe] = (fp_id_[3,0,probe] - 65) * 10 + fp_id_[3,1,probe]
        if cw:
            rise_time[probe]   = gm2.util.cf(np.flip(fp_time[skip:, probe]), np.flip(fp_freq[skip:, probe]))
        else:
            rise_time[probe]   = gm2.util.cf(fp_time[skip:, probe], fp_freq[skip:, probe])
        if not np.isnan(rise_time[probe]):
            rise_tr_pos[probe] = tr_pos[gm2.util.nearestIndex(tr_time[:,17//2], rise_time[probe]),17//2] 
            tr_pos_sel         = (tr_pos[:,17//2] > rise_tr_pos[probe] + tr_pos_offset - tr_azi_avg/2.)&(tr_pos[:,17//2] < rise_tr_pos[probe]  + tr_pos_offset + tr_azi_avg/2.)
            tr_mean[probe]     = tr_freq[tr_pos_sel,:].mean()

    return fp_id, rise_time, rise_tr_pos, tr_mean


ccw = getMeans([5217])
cw  = getMeans([5216, 5218], True)

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
