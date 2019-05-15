# Example how to use the Spike Module

import gm2
from gm2 import plt, np

# settings
ms = 2 # marker size
skip = 6
ylim = 500
binw = 5


run = 3997
tr = gm2.Trolley([run])
_, tr_phi, tr_freq = tr.getBasics()



probe = 0

tr_phi_  = tr_phi[skip:-skip, probe]
tr_freq_ = tr_freq[skip:-skip, probe]

spk = gm2.Spikes(tr_phi_, tr_freq_, gm2.PPM_HZ)

figsize = [plt.rcParams['figure.figsize'][0] * 2.0, plt.rcParams['figure.figsize'][1] * 2.0]
fig = plt.figure(figsize=figsize)
ax0_0 = plt.subplot2grid((2,2), (0, 0))
ax1_0 = plt.subplot2grid((2,2), (1, 0), sharex=ax0_0)
ax1_1 = plt.subplot2grid((2,2), (1, 1), sharey=ax1_0)

ax0_0.set_title("run "+str(run)+", probe "+str(probe + 1))

ax0_0.errorbar(tr_phi_/np.pi*180., tr_freq_/1e3, yerr=spk.outl/1e3, fmt='x', markersize=ms)
s = spk.isOutl()
ax0_0.errorbar(tr_phi_[s]/np.pi*180., tr_freq_[s]/1e3, yerr=spk.outl[s]/1e3, fmt='x', markersize=ms)
ax0_0.set_xlabel("azimuth [degree]")
ax0_0.set_ylabel("frequency [kHz]")


ax1_0.plot(tr_phi_/np.pi*180., spk.outl, '.', markersize=ms)
ax1_0.set_xlabel("azimuth [degree]")
ax1_0.set_ylabel("outlier [Hz]")


gm2.plotutil.histWithGauss(ax1_1, spk.outl, bins=np.arange(-ylim, ylim, binw), RMS=True)
ax1_1.semilogx()
ax1_1.set_ylim(ax1_0.get_ylim())
ax1_1.set_ylabel("counts")



gm2.despine()
plt.show()

