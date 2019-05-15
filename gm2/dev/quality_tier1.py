# coding: utf-8
import gm2
import numpy as np
import matplotlib.pyplot as plt

tr = gm2.Trolley([5303], True)

def callback():
    return tr.getTimeGPS(), tr.getPhi(0), tr.getFrequency(0), tr.getOutlier(0), tr.getFrequencySmooth(0)

time, phi, freq, d, freqSmooth = tr.loop(callback)

probe = 3
skip = 10
plt.plot(time[skip:-skip, 0], freqSmooth[skip:-skip, 0], '.', alpha=0.3)
plt.errorbar(time[skip:-skip, 0], freq[skip:-skip, 0], yerr=d[skip:-skip, 0], fmt='.')
plt.xlabel(r'azimuth [rad]')
plt.ylabel("frequency [Hz]")
plt.show()

'''
def run(runs):
    tr = gm2.Trolley(runs)
    def callback():
        return [tr.getTimeGPS(), tr.getPhi(), tr.getFrequency(), tr.getFidLength(), tr.getAmplitude()]
    time, phi, freq, length, amp = tr.loop(callback)

    skip = 0
    freq_ = freq[skip:,:,0].copy()
    cor = np.full(freq_.shape, False)

    for p in range(17):
        dmax = 1000
        while np.abs(dmax) > 61*2:
            s = 1
            d = freq_[s:-s, p] - (freq_[:-s*2,p] + freq_[s*2:,p])/2.
            n = np.argmax(np.abs(d)) + s
            dmax = d[n-s]
            freq_[n,p] = (freq_[n-s,p] + freq_[n+s,p])/2.
            cor[n+skip, p] = True

    sel = np.array([0,1,2,3,4,5,6,7,8,9,11,12,13,14,15,])
    bins = np.arange(-1.7,4.5,0.01)
    v = np.zeros([bins.shape[0]-1, 17])
    for probe in np.arange(17):
        v[:, probe], _, _ = plt.hist(phi[:,probe,0][cor[:,probe]], bins=bins, histtype='stepfilled')
    plt.close('all')
    bc = np.array(0.5*(bins[1:]+bins[:-1]))
    return v, bc


v1, bc1 = run([3997])
v2, bc2 = run([3996, 3998])


plt.step(bc1/np.pi*180,  v1[:,0],where='mid')
plt.step(bc2/np.pi*180, -v2[:,0], where='mid')
plt.xlabel("azimuth [deg]")
plt.xlim([-4,4])
plt.ylabel("#spikes/ %.1f deg" % ((bc1[3]-bc1[2])/np.pi*180))
plt.show()

plt.subplots_adjust(hspace=0.0, wspace=0.1)
for p in np.arange(9):
  for m in np.arange(2):
    i = p*2 + m
    plt.subplot(9,2, 1+i)
    if i < 17:
        plt.step(bc1/np.pi*180,  v1[:,i],where='mid')
        plt.step(bc2/np.pi*180, -v2[:,i], where='mid')
        plt.ylim([-4,4])
#plt.subplot(212)
#plt.step(bc1/np.pi*180,  v1[:,1],where='mid')
#plt.step(bc2/np.pi*180, -v2[:,1], where='mid')

plt.subplot(9,2,9*2-3)
plt.xlabel("azimuth [deg]")
#plt.subplot(9,2,9*2-2)
#plt.xlabel("azimuth [deg]")
plt.subplot(9,2,9*2/2)
plt.ylabel("#spikes/ %.1f deg" % ((bc1[3]-bc1[2])/np.pi*180))
plt.show()


sel = np.array([0,1,2,3,4,5,6,7,8,9,11,12,13,14,15,])
plt.step(bc1/np.pi*180,  v1[:,sel].sum(axis=1), where='mid')
plt.step(bc2/np.pi*180, -v2[:,sel].sum(axis=1), where='mid')
plt.xlabel("azimuth [deg]")
plt.ylabel("#spikes/ %.1f deg" % ((bc1[3]-bc1[2])/np.pi*180))
plt.show()


v1_2, bc1_2 = run([4058])
v2_2, bc2_2 = run([4057, 4059])


plt.subplots_adjust(hspace=0.0, wspace=0.1)
plt.subplot(211)
plt.step(bc1/np.pi*180,  v1[:,sel].sum(axis=1), where='mid')
plt.step(bc2/np.pi*180, -v2[:,sel].sum(axis=1), where='mid')
plt.subplot(212)
plt.step(bc1_2/np.pi*180,  v1_2[:,sel].sum(axis=1), where='mid')
plt.step(bc2_2/np.pi*180, -v2_2[:,sel].sum(axis=1), where='mid')
plt.xlabel("azimuth [deg]")
plt.ylabel("#spikes/ %.1f deg" % ((bc1[3]-bc1[2])/np.pi*180))
plt.show()



'''
'''
for p in range(17):
    plt.plot(phi[:,p,0], freq[:,p,0], '.')
    plt.plot(phi[cor[:,p],p,0], freq[cor[:,p],p,0], '.')
    plt.show()

for p in range(17):
    plt.plot(phi[:,p,0], freq[:,p,0], '.')
    plt.plot(phi[cor[:,p],p,0], freq[cor[:,p],p,0], '.')
plt.show()

for p in range(17):
    plt.plot(phi[cor[:,p],p,0], freq_[cor[:,p],p] - freq[cor[:,p],p,0], '.')
plt.show()
'''

'''
sel = np.array([0,1,2,3,4,5,6,7,8,9,11,12,13,14,15,])
plt.subplot(211)
plt.hist(phi[:,sel,0][cor[:,sel]], bins=np.arange(-1.7,4.5,0.01), histtype='stepfilled')
plt.subplot(212)
plt.hist(phi2[:,sel,0][cor2[:,sel]], bins=np.arange(-1.7,4.5,0.01), histtype='stepfilled')
plt.show()
'''









#d = freq[1:-1,:,0] - (freq[:-2,:,0] + freq[2:,:,0])/2.
#d2 = freq[2:-2,:,0] - (freq[:-4,:,0] + freq[1:-3,:,0] + freq[3:-1,:,0] + freq[4:,:,0] )/4.
#h1 = (phi[2:,:,0] - phi[:-2,:,0])/2.
