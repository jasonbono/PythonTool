import gm2
from gm2 import np, plt

runs = np.arange(3751+3, 3760+1) # 5h
yoke = 'G'
azi = 3

ip = gm2.Interpolation(runs)
mp_max = 3
freq_c = ip.tr.freq + ip.tr.getCalibration()
mp_raw_c = []
for mp_max in range(1,18,2):
    mp_raw_c.append(ip.loadTrMultipoles(n=mp_max, probe_ = 8, freqs = freq_c))

at = 45.
r = np.array([at**0, at**1, at**1, at**2, at**2, at**3, at**3, at**4, at**4, at**5, at**5, at**6, at**6, at**7, at**7, at**8, at**8])


fs = gm2.plotutil.figsize() 
plt.figure(figsize=[fs[0]*1.5,fs[1]])
plt.plot((ip.tr.time[:,8]-t0)/(3600*1e9), mp_raw_c[1][:,0]*gm2.HZ2PPM, '.', markersize=2)
plt.xlabel("time [h]")
plt.ylabel("dipole ppm")
gm2.despine()
plt.show()

t0 = ip.tr.time[ ip.tr.time>1e8].min()
for i,mp_max in enumerate(range(1,4,2)):
    res = np.zeros([mp_raw_c[0].shape[0], 17])
    for ev in range(mp_raw_c[0].shape[0]):
        res[ev, :] = gm2.util.multipole((gm2.TR.probes.position.r, gm2.TR.probes.position.theta), *(mp_raw_c[i][ev,:]/r[:mp_max])) - freq_c[ev,:]
    plt.figure(figsize=[fs[0]*1.5,fs[1]])
    ax = plt.subplot(211) 
    plt.plot((ip.tr.time-t0)/(3600*1e9), res*gm2.HZ2PPM, '.', markersize=2)
    #plt.xlabel("time [h]")
    plt.ylabel("residual [ppm]")
    plt.title("n <= %i" % (mp_max//2))
    plt.ylim([-3,2])
    plt.subplot(212, sharex=ax)
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.plot((ip.tr.time-t0)/(3600*1e9), (res-res.mean(axis=0)) *gm2.HZ2PPM, '.', markersize=2, alpha=0.2)
    plt.xlabel("time [h]")
    plt.ylabel("residual \n variations [ppm]")
    plt.plot([0, (ip.tr.time.max()-t0)/(3600*1e9)],[0.0, 0.0],'k--')
    plt.ylim([-0.2,0.2])
    gm2.despine()
    plt.savefig("plots/residuals_n%i.png" % (mp_max//2))
    plt.show()



mp = 2
res = np.zeros([mp_raw_c[0].shape[0], 17])
for ev in range(mp_raw_c[0].shape[0]):
    res[ev, :] = gm2.util.multipole((gm2.TR.probes.position.r, gm2.TR.probes.position.theta), *(mp_raw_c[mp][ev,:]/r[:mp*2+1])) - freq_c[ev,:]

plt.figure(figsize=[fs[0]*2,fs[1]*1.5])
for probe in range(17):
    plt.subplot(6, 3, probe+1)
    plt.plot((ip.tr.time[:,0]-t0)/(3600*1e9), (res[:,probe]-res[3000:4000,probe].mean()) *gm2.HZ2PPB, '.', markersize=2, alpha=0.2)
    plt.plot([0, (ip.tr.time.max()-t0)/(3600*1e9)],[0.0, 0.0],'k--')
    plt.ylim([-80, 80])
    #plt.title("#%i" % (probe + 1))
    plt.text((ip.tr.time.max()-t0)/(3600*1e9)-1, 40,"#%i" % (probe + 1))
plt.xlabel("time [h]")
plt.subplots_adjust(hspace=0)
gm2.sns.despine()
plt.show()


chi2 = []
res_mean = []
res_std  = []
for mp in range(9):
    for ev in range(mp_raw_c[0].shape[0]):
        res[ev, :] = gm2.util.multipole((gm2.TR.probes.position.r, gm2.TR.probes.position.theta), *(mp_raw_c[mp][ev,:]/r[:mp*2+1])) - freq_c[ev,:]
    res_mean.append(res[abs(res.max(axis=1))<500,:].mean(axis=0))
    res_std.append(res[abs(res.max(axis=1)<500),:].std(axis=0))
    chi2.append(np.sqrt((res**2).sum(axis=1)))

plt.figure(figsize=[fs[0]*2,fs[1]*1.5])
ax = plt.subplot(311)
plt.errorbar(np.arange(8), [res_mean[i][4]*gm2.HZ2PPM for i in range(8)], [res_std[i][0]*gm2.HZ2PPM for i in range(8)],label="#1", fmt='x-')
plt.plot([0,8],[0,0],'--k',linewidth=0.5)
plt.legend()
plt.ylabel("residual [ppm]")
plt.setp(ax.get_xticklabels(), visible=False)
plt.subplot(312, sharex=ax)
for j in range(1,5,1):
    plt.errorbar(np.arange(8), [res_mean[i][j]*gm2.HZ2PPM for i in range(8)], [res_std[i][j]*gm2.HZ2PPM for i in range(8)],label="#%i" % (j+1), fmt='x-')
plt.plot([0,8],[0,0],'--k',linewidth=0.5)
plt.legend(ncol=4)
plt.setp(ax.get_xticklabels(), visible=False)
plt.ylabel("residual [ppm]")
plt.subplot(313, sharex=ax)
for j in range(5,17,1):
    plt.errorbar(np.arange(8), [res_mean[i][j]*gm2.HZ2PPM for i in range(8)], [res_std[i][j]*gm2.HZ2PPM for i in range(8)], label="#%i" % (j+1),fmt='x-')
plt.plot([0,8],[0,0],'--k',linewidth=0.5)
plt.legend(ncol=6)
plt.subplots_adjust(hspace=0)
plt.xlabel("n")
plt.ylabel("residual [ppm]")
gm2.sns.despine()
plt.show()



plt.figure(figsize=[fs[0]*2,fs[1]*1.5])
for mp in range(8):
    plt.plot((ip.tr.time[:,0]-t0)/(3600*1e9), chi2[mp] * gm2.HZ2PPM, '.',  markersize=2, label="n <= %i" % mp)
plt.legend()
plt.xlabel("time [h]")
plt.ylabel(r"$\chi^2 [ppm]$")
gm2.despine()
plt.show()

mp = 8
cc = np.array([c.mean() for c in chi2])
plt.plot(np.arange(mp), cc[:mp] * gm2.HZ2PPM, 'x')
plt.xlabel("n >= ")
plt.ylabel(r"$\chi^2$ [ppm]")
plt.gca().set_xticklabels(np.arange(mp))
plt.gca().set_xticks(np.arange(mp))
gm2.despine()
plt.show()




