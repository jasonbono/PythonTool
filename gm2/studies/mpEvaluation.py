import gm2
from gm2 import plt, np

runs = [3997]

mps = 3

ip = gm2.Interpolation(runs)
ip.loadTrySpikes()
ip.loadTrMultipoles(n=mps, probe_=8, freqs = ip.trFreqSmooth + ip.tr.getCalibration())


tr_aziAt = gm2.util.interp1d(ip.tr.time.reshape([-1]), (ip.tr.azi.reshape([-1]) + (2*np.pi))%(2*np.pi), fill_value="interpolate")
tr_mpAt  = [gm2.util.interp1d((ip.tr.azi[:,8] + (2*np.pi))%(2*np.pi), ip.tr_mp[:,mp], fill_value="interpolate") for mp in range(mps)]

fp_t_rise = ip.getFpTrlyTimes(plot=[])
fps_t_rise = []
fps_m1 = []
tr_azis = []
tr_m1  = []
for yoke in range(ord('A'), ord('A')+12):
    for azi in range(1,6):
        s_station = ip.fp.select(yokes=[chr(yoke)], aziIds=[azi])
        #print(yoke, azi, fp_t_rise[s_station], np.nanmean(fp_t_rise[s_station]))
        fps_t_rise.append(np.nanmean(fp_t_rise[s_station]))
        s_time = (ip.fp.time[:,s_station][:,0] > fps_t_rise[-1] - fp_t_offset1 - fp_t_range)&(ip.fp.time[:,s_station][:,0] < fps_t_rise[-1] - fp_t_offset1 + fp_t_range)
        fps_m1.append(np.nanmean(ip.fp.freq[s_time,:][:,s_station]))
        if np.isnan(fps_t_rise[-1]):
            tr_m1.append(np.nan)
            azis_ = []
        else:
            azi = tr_aziAt(fps_t_rise[-1])
            azis_ = np.arange(azi-tr_azi_range/2., azi+tr_azi_range/2., tr_azi_range/10.)
            #print(tr_aziAt((fps_t_rise[-1]))) 
            tr_m1.append(np.nanmean(tr_mpAt[0](azis_)))
        tr_azis.append(azis_)

fps_m1 = np.array(fps_m1)
tr_m1  = np.array(tr_m1)


runs2 = [3956]

ip2 = gm2.Interpolation(runs2)
ip2.loadTrySpikes()
ip2.loadTrMultipoles(n=mps, probe_=8, freqs = ip2.trFreqSmooth + ip2.tr.getCalibration())


tr_aziAt2 = gm2.util.interp1d(ip2.tr.time.reshape([-1]), (ip2.tr.azi.reshape([-1]) + (2*np.pi))%(2*np.pi), fill_value="interpolate")
tr_mpAt2  = [gm2.util.interp1d((ip2.tr.azi[:,8] + (2*np.pi))%(2*np.pi), ip2.tr_mp[:,mp], fill_value="interpolate") for mp in range(mps)]

fp2_t_rise = ip2.getFpTrlyTimes(plot=[])
fps2_t_rise = []
fps2_m1 = []
tr2_azis = []
tr2_m1  = []
i = 0
for yoke in range(ord('A'), ord('A')+12):
    for azi in range(1,6):
        s_station = ip2.fp.select(yokes=[chr(yoke)], aziIds=[azi])
        #print(yoke, azi, fp_t_rise[s_station], np.nanmean(fp_t_rise[s_station]))
        fps2_t_rise.append(np.nanmean(fp2_t_rise[s_station]))
        s_time = (ip2.fp.time[:,s_station][:,0] > fps2_t_rise[-1] - fp_t_offset1 - fp_t_range)&(ip2.fp.time[:,s_station][:,0] < fps2_t_rise[-1] - fp_t_offset1 + fp_t_range)
        fps2_m1.append(np.nanmean(ip2.fp.freq[s_time,:][:,s_station]))
        if np.isnan(fps_t_rise[-1]):
            tr_m1.append(np.nan)
            #azis_ = []
        else:
            #azi = tr_aziAt(fps_t_rise[-1])
            #azis_ = tr_azis[i]
            #print(tr_aziAt((fps_t_rise[-1]))) 
            tr2_m1.append(np.nanmean(tr_mpAt2[0](tr_azis[i])))
            #tr_azis.append(azis_)
        i = i + 1


fps2_m1 = np.array(fps2_m1)
tr2_m1  = np.array(tr2_m1)


d = tr2_m1 - (fps2_m1 + (tr_m1 - fps_m1))

runs1 = [3996, 3998]
ip1 = gm2.Interpolation(runs1)
ip1.loadTrySpikes()
ip1.loadTrMultipoles(n=mps, probe_=8, freqs = ip1.trFreqSmooth + ip1.tr.getCalibration())



