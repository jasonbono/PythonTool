import gm2

run = 3997
tr = gm2.Trolley([run])
_, tr_phi, tr_freq = tr.getBasics()

probe = 0

tr_phi_  = tr_phi[9:-4, probe]
tr_freq_ = tr_freq[9:-4, probe]
spk = gm2.Spikes(tr_phi_, tr_freq_, gm2.PPM_HZ)
s = spk.isOk()
tr_phi_  = tr_phi_[s]
tr_freq_ = tr_freq_[s]

print("Data points: ", tr_phi[10:-5,0].shape[-1], "\t", tr_phi_.shape[-1])

N = 500
fr_raw = gm2.Fourier(tr_phi[10:-5, probe],  tr_freq[10:-5, probe], N)
fr_rm  = gm2.Fourier(tr_phi_, tr_freq_, N)


from gm2 import plt, np
phi = tr_phi[10:-5, probe]
ax1 = plt.subplot(211)
ax1.plot(phi/np.pi*180., fr_raw.B(phi)/1e3, label="full data")
ax1.plot(phi/np.pi*180., fr_rm.B(phi)/1e3,  label="removed spikes")
ax1.plot(tr_phi[9:-4, probe]/np.pi*180.,  tr_freq[9:-4, probe]/1e3, '.', markersize=2)
ax1.plot(tr_phi[9:-4, probe][s == False]/np.pi*180.,  tr_freq[9:-4, probe][s == False]/1e3, '.', markersize=3)
ax1.set_xlabel("azimuth [degree]")
ax1.set_ylabel("frequency [kHz]")
plt.legend(ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.1))

plt.subplot(212, sharex=ax1)
plt.plot(phi/np.pi*180., fr_raw.B(phi)-fr_rm.B(phi))
plt.xlabel("azimuth [degree]")
plt.ylabel("difference [Hz]")
plt.gca().text(0.5,0.8,"RMS %.0f / %.0f ppb" % (fr_raw.getChi(tr_phi[10:-5, probe],  tr_freq[10:-5, probe])*1e3, fr_rm.getChi(tr_phi_, tr_freq_)*1e3),\
        horizontalalignment='center',\
        transform = plt.gca().transAxes)
gm2.despine()
plt.show()



######################### Test convergence #########################
fr = gm2.Fourier()
ns,  chi2s,  chi2bars = fr.convTest(tr_phi[10:-5, probe], tr_freq[10:-5, probe], dN=[50, 100], Nstep=10) 
ns_, chi2s_, chi2bars_ = fr.convTest(tr_phi_, tr_freq_, dN=[50, 100], Nstep=10) 

from gm2 import plt
plt.semilogy(ns,  chi2s,  '.', label="full data", markersize=2)
#plt.semilogy(ns,  chi2bars[:,0],  'x', label=r'$\bar{\chi}^2(N, \Delta N), \Delta N =  50$', markersize=4)
#plt.semilogy(ns,  chi2bars[:,1],  'x', label=r'$\bar{\chi}^2(N, \Delta N), \Delta N = 100$', markersize=4)
plt.semilogy(ns_, chi2s_, '.', label="removed spikes", markersize=2)
#plt.semilogy(ns_, chi2bars_, 'x', label="removed spikes", markersize=4)
plt.xlabel("N")
plt.ylabel(r'$\chi^2$ [ppm]')
plt.title("run "+str(run)+", probe "+str(probe+1))
plt.legend()
gm2.despine()
plt.show()



