import gm2
from gm2 import plt, np

runs = [3997]
ip = gm2.Interpolation(runs)

spk = [gm2.Spikes(ip.tr.azi[:,probe], ip.tr.freq[:,probe], gm2.PPM_HZ/2.) for probe in range(17)] 

freq_smooth = np.array([spk_.freq for spk_ in spk]).T

ip.loadTrMultipoles(freqs=freq_smooth) 


freqAtPos = [gm2.util.interp1d(ip.tr.azi[:,probe], freq_smooth[:,probe]) for probe in range(17)]
azis =      [np.arange(ip.tr.azi[:,probe].min(), ip.tr.azi[:,probe].max(), (ip.tr.azi[:,probe].max()-ip.tr.azi[:,probe].min())/ip.tr.azi.shape[0]*4) for  probe in range(17)]
freqMean = np.array([freqAtPos[probe](azis[probe]).mean() for probe in range(17)])



gm2.plotutil.plotTrFieldMap(freqMean, nmax=6)
plt.savefig("plots/fieldMapsMean.png")

'''
for azi_ in np.arange(-90,0,0.1):
    freq_ = np.array([freqAtPos[probe](azi_/180.*np.pi).mean() for probe in range(17)])
    gm2.plotutil.plotTrFieldMap(freq_, nmax=6)
    plt.title(r"at % 3.0f$^{\degree}$" % azi_)
    plt.savefig("plots/fieldMaps%03.1f.png" % azi_)
    plt.gcf().clear()
    #plt.show()
'''

plt.figure(figsize=[gm2.plotutil.figsize()[0]*2,gm2.plotutil.figsize()[1]*1.2])
ax0 = plt.subplot(211)
data_ = ip.tr_mp[:,0]
center_ = (data_.max() + data_.min())/2.
plt.plot(ip.tr.azi[:,8]/np.pi*180, (data_-center_)/gm2.PPM_HZ, '.', markersize=2)
plt.ylabel("Dipole Change\n"+r"$\Delta B_{0}$ [ppm]")
plt.setp(ax0.get_xticklabels(), visible=False)

plt.subplot(212, sharex=ax0)
plt.plot(ip.tr.azi[:,8]/np.pi*180, (ip.tr_mp[:,1])/gm2.PPM_HZ, '.', markersize=2, label="a1: normal", color=gm2.colors[1])
plt.plot(ip.tr.azi[:,8]/np.pi*180, (ip.tr_mp[:,2])/gm2.PPM_HZ, '.', markersize=2, label="b1: skew",   color=gm2.colors[2])
plt.legend(markerscale=4.)

plt.ylabel("Quadrupole \n[ppm@45mm]")
plt.xlabel(r"azimuth [degree]")
gm2.despine()
plt.show()





