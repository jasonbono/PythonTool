import numpy as np
import matplotlib.pyplot as plt
import Trolley

from util import *

t    = Trolley.Trolley([3614])
t_hr = Trolley.Trolley([3687])
freq, phi, time = t.getBasics()
freq_hr, phi_hr, time_hr = t_hr.getBasics()


freq_hr_mean = []
freq_hr_std  = []
phi_hr_mean  = []
freq_ = [freq[1,:]]
phi_  = [phi[1,:]]
for i in range(1,freq_hr.shape[0]):
    #if i > 100:
    #    break
    #print i, phi_hr[i,0], len(phi_), np.array(phi_)[:,0].mean(), phi[i,0] - np.array(phi_)[:,0].mean()
    if phi_hr[i,0] - np.array(phi_)[:,0].mean() < -0.0001:
        #print "ok"
        # new step
        freq_hr_mean.append(np.array(freq_).mean(axis=0))
        freq_hr_std.append(np.array(freq_).std(axis=0))
        phi_hr_mean.append( np.array(phi_ ).mean(axis=0))
        freq_ = [freq_hr[i,:]]
        phi_  = [phi_hr[i,:]]
    else:
        freq_.append(freq_hr[i,:])
        phi_.append(phi_hr[i,:])

freq_hr_mean = np.array(freq_hr_mean)
freq_hr_std  = np.array(freq_hr_std)
phi_hr_mean  = np.array(phi_hr_mean)


for probe in [0,1]:
    plt.errorbar(phi_hr_mean[:, probe], freq_hr_mean[:, probe], yerr=freq_hr_std[:, probe], fmt=' ', label="run 3687: high resolution")
    plt.plot(phi[:, probe], freq[:, probe],'.', label="run 2614: normal trolley run")
    #plt.plot(smooth(phi[:, probe], 0.5, 'gauss'), smooth(freq[:, probe],  0.5, 'gauss'), '--', label="0.5")
    #plt.plot(smooth(phi[:, probe], 1,   'gauss'), smooth(freq[:, probe],  1,   'gauss'), '--', label="1.0")
    #plt.plot(smooth(phi[:, probe], 1.5, 'gauss'), smooth(freq[:, probe],  1.5, 'gauss'), '--', label="1.5")
    #plt.plot(smooth(phi[:, probe], 2,   'gauss'), smooth(freq[:, probe],  2,   'gauss'), '--', label="2.0")
    #plt.plot(smooth(phi[:, probe], 3,   'gauss'), smooth(freq[:, probe],  3,   'gauss'), '--', label="3.0")

    #plt.errorbar(phi_hr_mean[:, probe], freq_hr_mean[:, probe], yerr=freq_hr_std[:, probe], fmt=' ', label="run 3687: high resolution")

    #plt.plot(phi[:, probe], freq[:, probe],'.', label="run 2614: normal trolley run")
    #plt.plot(smooth(phi[:, probe],  3, 'lr'), smooth(freq[:, probe],  3, 'lr'), '--', label=" 3")
    plt.plot(smooth(phi[:, probe],  5, 'lr'), smooth(freq[:, probe],  5, 'lr'), '--', label="run 264: filter")
    #plt.plot(smooth(phi[:, probe],  7, 'lr'), smooth(freq[:, probe],  7, 'lr'), '--', label=" 7")
    #plt.plot(smooth(phi[:, probe],  9, 'lr'), smooth(freq[:, probe],  9, 'lr'), '--', label=" 9")
    #plt.plot(smooth(phi[:, probe], 11, 'lr'), smooth(freq[:, probe], 11, 'lr'), '--', label="11")
    plt.xlabel("azimuth [rad]")
    plt.ylabel("frequency [Hz]")
    plt.legend()
    plt.xlim([4.42,4.58])
    plt.ylim([51400,53700])
    plt.show()

from scipy import interpolate

freq_hr_f = interpolate.interp1d(phi_hr_mean[:, probe], freq_hr_mean[:, probe])
s  = (phi[:, probe] > phi_hr_mean[:, probe].min())&(phi[:, probe] < phi_hr_mean[:, probe].max())
#s2 = (phi[:, probe] > phi_hr_mean[:, probe].min())&(phi[:, probe] < phi_hr_mean[:, probe].max())

#for probe in range(17):
plt.plot(phi[s, probe], freq[s, probe]-freq_hr_f(phi[s, probe]),'.', label="run 2614: normal trolley run")
plt.plot(smooth(phi[s, probe], 5,'lr'), smooth(freq[s, probe],  5, 'lr') - freq_hr_f(smooth(phi[s, probe], 5,'lr')), '.', label="run 2614: normal trolley run")
plt.xlabel("azimuth [rad]")
plt.ylabel("frequency residual [Hz]")
plt.xlim([4.42,4.58])
plt.show()

'''
plt.plot(smooth(phi[:, probe], 11, 'ref'), smooth(freq[:, probe],  0.5, 'gauss') - smooth(freq[:, probe],  11, 'ref'), '.', label="0.5")
plt.plot(smooth(phi[:, probe], 11, 'ref'), smooth(freq[:, probe],  1.0, 'gauss') - smooth(freq[:, probe],  11, 'ref'), '.', label="1.0")
plt.plot(smooth(phi[:, probe], 11, 'ref'), smooth(freq[:, probe],  1.5, 'gauss') - smooth(freq[:, probe],  11, 'ref'), '.', label="1.5")
plt.plot(smooth(phi[:, probe], 11, 'ref'), smooth(freq[:, probe],  2.0, 'gauss') - smooth(freq[:, probe],  11, 'ref'), '.', label="2.0")
plt.plot(smooth(phi[:, probe], 11, 'ref'), smooth(freq[:, probe],  3.0, 'gauss') - smooth(freq[:, probe],  11, 'ref'), '.', label="3.0")
plt.xlabel("azimuth [rad]")
plt.ylabel("frequency residual [Hz]")
plt.xlim([4.42,4.58])
plt.show()


plt.plot(smooth(phi[:, probe],  3+2, 'ref'), smooth(freq[:, probe],  3, 'lr') - smooth(freq[:, probe],  3+2, 'ref'), '.', label="3")
plt.plot(smooth(phi[:, probe],  5+2, 'ref'), smooth(freq[:, probe],  5, 'lr') - smooth(freq[:, probe],  5+2, 'ref'), '.', label="3")
plt.plot(smooth(phi[:, probe],  7+2, 'ref'), smooth(freq[:, probe],  7, 'lr') - smooth(freq[:, probe],  7+2, 'ref'), '.', label="3")
plt.plot(smooth(phi[:, probe],  9+2, 'ref'), smooth(freq[:, probe],  9, 'lr') - smooth(freq[:, probe],  9+2, 'ref'), '.', label="3")
plt.plot(smooth(phi[:, probe], 11+2, 'ref'), smooth(freq[:, probe], 11, 'lr') - smooth(freq[:, probe], 11+2, 'ref'), '.', label="3")
plt.xlabel("azimuth [rad]")
plt.ylabel("frequency residual [Hz]")
plt.xlim([4.42,4.58])
plt.show()
'''
from scipy.optimize import curve_fit
def gaussian(x, a, m, s):
        return a * np.exp( - ((x - m) / s) ** 2)

for probe in range(17):
    #print probe
    v,b,_, = plt.hist(smooth(freq[:, probe],  5, 'lr') - smooth(freq[:, probe],  5+2, 'ref'), bins=np.arange(-100,100,1))
    bc = b[:-1] + np.diff(b) / 2.0
    popt, _ = curve_fit(gaussian, bc, v, p0=[1.0, 0., 1.])
    xx = np.arange(-100,100)
    plt.plot(xx, gaussian(xx,*popt),'-')
    plt.text(50,v.max()*0.7,"mean: "+("%.2f" %popt[1])+"\nstd:   "+("%.1f" % popt[2]))
    plt.xlabel("frequency [Hz]")
    print probe, popt[2]
plt.show()

'''
res = []
res_bin = []
res_n = []
for lim in np.arange(-100,100,5.0):
     s = (dff>lim)&(dff<lim+10.0)
     v,b,_, = plt.hist(df[probe,:-1][s], bins=np.arange(-100,100,2))
     bc = b[:-1] + np.diff(b) / 2.0
     popt, _ = curve_fit(gaussian, bc, v, p0=[v.max(), 0., 15.])
     xx = np.arange(-100,100)
     plt.plot(xx, gaussian(xx,*popt),'-')
     plt.text(50,v.max()*0.7,"mean: "+("%.2f" %popt[1])+"\nstd:   "+("%.1f" % popt[2]))
     plt.xlabel("frequency [Hz]")
     print lim, lim+1, popt[2]
     res.append( popt[2])
     res_bin.append(lim + 2.5)
     res_n.append(v.sum())
     print lim+2.5, v.sum(), popt[1], popt[2]
     #plt.show()

res     = np.array(res)
res_bin = np.array(res_bin)
res_n   = np.array(res_n)
s = res_n>200
plt.plot(res_bin[s], res[s],'o')
plt.show()
'''


df = np.array([smooth(freq[:, a],  5, 'lr') - smooth(freq[:, a],  5+2, 'ref') for a in range(17)])

res = []
res_err = []
res_bin = []
res_n = []
for probe in range(16):
    res.append([])
    res_err.append([])
    res_bin.append([])
    res_n.append([])
    sf = smooth(freq[:, probe],  5, 'lr')
    sf_std = np.full(len(sf), -100)
    sf_amp = np.full(len(sf), -100)
    for i in range(5, len(sf)-6):
        sf_std[i] = sf[i-2:i+3].std()
        sf_amp[i] = (sf[i-2:i+3].max() - sf[i-2:i+3].min())/np.abs(np.argmax(sf[i-2:i+3])-np.argmin(sf[i-2:i+3]))

    binsize = 5
    for lim in np.arange(0,100, binsize):
         #s = (sf_std>lim)&(sf_std<lim + binsize)
         s = (sf_amp>lim)&(sf_amp<lim + binsize)
         v,b,_, = plt.hist(df[probe,:][s], bins=np.arange(-100,100,2))
         bc = b[:-1] + np.diff(b) / 2.0
         if(v.sum()>50):
             popt, pcov  = curve_fit(gaussian, bc, v, p0=[v.max(), 0., 15.])
         else:
             popt = [-100,-100,-100]
             pcov = np.zeros([3,3])
         xx = np.arange(-100,100)
         plt.plot(xx, gaussian(xx,*popt),'-')
         plt.text(50,v.max()*0.7,"mean: "+("%.2f" %popt[1])+"\nstd:   "+("%.1f" % popt[2]))
         plt.xlabel("frequency [Hz]")
         #print lim, lim+1, popt[2]
         res[probe].append( popt[2])
         res_err[probe].append(pcov[2,2])
         res_bin[probe].append(lim + binsize/2.0)
         res_n[probe].append(v.sum())
         print lim, v.sum(), popt[1], popt[2]
         #plt.show()
    plt.clf()

res     = np.array(res)
res_err = np.array(res_err)
res_bin = np.array(res_bin)
res_n   = np.array(res_n)
for probe in [0,1,4]:
    s = res_n[probe]>200
    plt.errorbar(res_bin[probe,s]*3.55, res[probe,s], xerr=binsize/np.sqrt(12), yerr=res_err[probe,s], fmt=' ', label="probe "+str(probe))
plt.xlabel('largest gradient over 15mm [Hz/mm]')
plt.ylabel('resolution [Hz]')
plt.legend()
plt.show()


# Correlation
df = np.array([smooth(freq[:, a],  5, 'lr') - smooth(freq[:, a],  5+2, 'ref') for a in range(17)])

window = 8000
c = np.zeros([17, df.shape[-1]-window])
for i in range(df.shape[-1]-window):
    c[:,i] = np.corrcoef(df[:, i:i + window])[7,:]

f, ax = plt.subplots(2, 1, sharex=True)
for p in range(17):
    ax[0].plot(smooth(freq[:, p],  5, 'lr')[window/2:-window/2])
    ax[1].plot(c[p,:],'.')
f.show()


# 
