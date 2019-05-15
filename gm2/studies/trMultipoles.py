import gm2

runs = [3997]

ip = gm2.Interpolation(runs)
ip.loadTrySpikes()
mp_max = 13
# raw multipoles
mp_raw = ip.loadTrMultipoles(n=mp_max, probe_=8, freqs = ip.tr.freq)
# spike corrected multipoles
mp_smooth = ip.loadTrMultipoles(n=mp_max, probe_=8, freqs = ip.trFreqSmooth)
# calibrated multipoles
mp_raw_c =  ip.loadTrMultipoles(n=mp_max, probe_=8, freqs = ip.tr.freq + ip.tr.getCalibration())
# spike corrected multipoles
mp_smooth_c = ip.loadTrMultipoles(n=mp_max, probe_=8, freqs = ip.trFreqSmooth + ip.tr.getCalibration())

from gm2 import plt, np

fs = gm2.plotutil.figsize()
for mp in range(0,mp_max):
    plt.figure(figsize=[fs[0]*2, fs[1]])
    plt.plot(ip.tr.azi[:,8]/np.pi*180., mp_raw[:,     mp] * gm2.HZ2PPM  , '.', markersize=2, label="raw")
    plt.plot(ip.tr.azi[:,8]/np.pi*180., mp_smooth[:,  mp] * gm2.HZ2PPM, '.', markersize=2, label="smooth")
    plt.plot(ip.tr.azi[:,8]/np.pi*180., mp_raw_c[:,   mp] * gm2.HZ2PPM, '.', markersize=2, label="raw calib.")
    plt.plot(ip.tr.azi[:,8]/np.pi*180., mp_smooth_c[:,mp] * gm2.HZ2PPM, '.', markersize=2, label="smooth calib.")
    plt.xlabel("azimuth [deg]")
    if mp > 0:
        plt.ylabel("[ppm] @ 4.5cm")
        mean = 0
        r = 20
    else:
        plt.ylabel("[ppm]")
        mean = mp_smooth_c[:,mp].mean()* gm2.HZ2PPM
        r = 100
    plt.ylim([mean-r, mean+r])
    plt.title("run %i, m = %i, n = %i" % (runs[0], ((mp+1)//2), mp % 2))
    plt.legend(markerscale=6)
    gm2.despine()
    plt.savefig("plots/mp_%i_m%i_n%i.png" % (runs[0], ((mp+1)//2), mp % 2))
    plt.clf()


runs2 = [3956]

ip2 = gm2.Interpolation(runs2)
ip2.loadTrySpikes()
mp_max = 13
# raw multipoles
#mp_raw2 = ip2.loadTrMultipoles(n=mp_max, probe_=8, freqs = ip2.tr.freq)
# spike corrected multipoles
#mp_smooth2 = ip2.loadTrMultipoles(n=mp_max, probe_=8, freqs = ip2.trFreqSmooth)
# calibrated multipoles
mp_raw_c2 =  ip2.loadTrMultipoles(n=mp_max, probe_=8, freqs = ip2.tr.freq + ip2.tr.getCalibration())
# spike corrected multipoles
mp_smooth_c2 = ip2.loadTrMultipoles(n=mp_max, probe_=8, freqs = ip2.trFreqSmooth + ip2.tr.getCalibration())


fs = gm2.plotutil.figsize()
for mp in range(0, mp_max):
    plt.figure(figsize=[fs[0]*2, fs[1]])
    #mp_raw2_      = gm2.util.interp1d(ip2.tr.azi[:,8], mp_raw2[:,mp], fill_value="extrapolate")(ip.tr.azi[:,8])
    #mp_smooth2_   = gm2.util.interp1d(ip2.tr.azi[:,8], mp_smooth2[:,mp],   fill_value="extrapolate")(ip.tr.azi[:,8])
    mp_raw_c2_    = gm2.util.interp1d(ip2.tr.azi[:,8], mp_raw_c2[:,mp],    fill_value="extrapolate")(ip.tr.azi[:,8])
    mp_smooth_c2_ = gm2.util.interp1d(ip2.tr.azi[:,8], mp_smooth_c2[:,mp], fill_value="extrapolate")(ip.tr.azi[:,8])
    #plt.plot(ip.tr.azi[:,8]/np.pi*180., (mp_raw[:,     mp] - mp_raw2_)      * gm2.HZ2PPM, '.', markersize=2, label="raw")
    #plt.plot(ip.tr.azi[:,8]/np.pi*180., (mp_smooth[:,  mp] - mp_smooth2_)   * gm2.HZ2PPM, '.', markersize=2, label="smooth")
    plt.plot(ip.tr.azi[:,8]/np.pi*180., (mp_raw_c[:,   mp] - mp_raw_c2_)    * gm2.HZ2PPM, '.', markersize=2, label="raw")
    plt.plot(ip.tr.azi[:,8]/np.pi*180., (mp_smooth_c[:,mp] - mp_smooth_c2_) * gm2.HZ2PPM, '.', markersize=2, label="smooth")
    plt.xlabel("azimuth [deg]")
    if mp > 0:
        plt.ylabel("difference [ppm] @ 4.5cm")
    else:
        plt.ylabel("difference [ppm]")
    mean = 0
    r = 20
    plt.ylim([mean-r, mean+r])
    plt.title("runs %i and %i, m = %i, n = %i" % (runs[0], runs2[0], ((mp+1)//2), mp % 2))
    plt.legend(markerscale=6)
    gm2.despine()
    #plt.show()
    plt.savefig("plots/mp_difference_%i_%i_m%i_n%i.png" % (runs2[0], runs[0], ((mp+1)//2), mp % 2))
    plt.clf()



runsS = np.arange(3751+3, 3760+1) # 5h
yoke = 'G'
azi = 3


ipS = gm2.Interpolation(runsS)
mp_s   =  ipS.loadTrMultipoles(n=mp_max, probe_=8, freqs = ipS.tr.freq)
mp_s_c =  ipS.loadTrMultipoles(n=mp_max, probe_=8, freqs = ipS.tr.freq + ipS.tr.getCalibration())

fs = gm2.plotutil.figsize()
t0 = ipS.tr.time.min()
for mp in range(0, mp_max):
    plt.figure(figsize=[fs[0]*2, fs[1]])
    plt.plot((ipS.tr.time[:,8]-t0)/(3600*1e9), mp_s[:,mp]   * gm2.HZ2PPM, '.', markersize=2, label="raw")
    plt.plot((ipS.tr.time[:,8]-t0)/(3600*1e9), mp_s_c[:,mp] * gm2.HZ2PPM, '.', markersize=2, label="calib.")
    plt.xlabel("time [h]")
    if mp > 0:
        plt.ylabel("[ppm] @ 4.5cm")
        r = 1
    else:
        plt.ylabel("[ppm]")
        r = 5
    mean = mp_s[:,mp].mean() * gm2.HZ2PPM
    plt.ylim([mean-r, mean+r])
    plt.title("runs %i - %i, m = %i, n = %i" % (runsS[0], runsS[-1], ((mp+1)//2), mp % 2))
    plt.legend(markerscale=6)
    gm2.despine()
    plt.savefig("plots/mp_time_%i_m%i_n%i.png" % (runsS[0], ((mp+1)//2), mp % 2))
    #plt.show()
    plt.clf()


