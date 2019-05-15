import gm2
from gm2 import np, plt

runs = np.arange(3751+3, 3760+1) # 5h
yoke = 'G'
azi = 3

ip = gm2.Interpolation(runs)
mp_max = 3
mp_raw_c =  ip.loadTrMultipoles(n=mp_max, probe_=8, freqs = ip.tr.freq + ip.tr.getCalibration())

skip = 100

fs = gm2.plotutil.figsize()
plt.figure(figsize=[fs[0]*2, fs[1]])
t0 = ip.fp.time[ip.fp.time>1e18].min()
for layer in ["T","B"]:
    for rad in ["I","M","O"]:
            s = ip.fp.select(yokes=[yoke], aziIds=[azi], layers=[layer], radIds=[rad])
            data_ = ip.fp.freq[skip:-skip,s] - ip.tr_mpAt[0](ip.fp.time[skip:-skip,s])
            plt.plot((ip.fp.time[skip:-skip,s] - t0)/(3600*1e9), (data_ - data_.mean()) * gm2.HZ2PPM, '.', markersize=2, label=layer+rad, alpha=0.5)


s = ip.fp.select(yokes=[yoke], aziIds=[azi], layers=["T","B"], radIds=["I","M","O"])
data_ = ip.fp.freq[skip:-skip,s].mean(axis=1) - ip.tr_mpAt[0](ip.fp.time[skip:-skip,s][:,0])
plt.plot((ip.fp.time[skip:-skip,s][:,0] - t0)/(3600*1e9), (data_ - data_.mean()) * gm2.HZ2PPM, 'k.', markersize=1, label=r"$m_1^{\rm{fp}}$", alpha=0.3)


plt.legend(markerscale=6)
plt.xlabel("time [h]")
plt.ylabel(r"$f_{\rm{fp}} - m_1^{\rm{trolley}}$ [ppm]")
gm2.despine()
plt.show()



