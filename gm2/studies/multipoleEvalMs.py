import gm2
from gm2 import np, plt

runs = np.arange(3751+3, 3760+1) # 5h
yoke = 'G'
azi = 3


pos_fp = (gm2.FP.probes.position.r, gm2.FP.probes.position.theta)
pos_tr = (gm2.TR.probes.position.r, gm2.TR.probes.position.theta)

trb_ = np.array([gm2.util.multipole(pos_tr, 1,     0,    0),
                 gm2.util.multipole(pos_tr, 0, 1./45,    0),
                 gm2.util.multipole(pos_tr, 0,    0, 1./45),
                 gm2.util.multipole(pos_tr, 0,    0,     0, 1./45**2,     0),
                 gm2.util.multipole(pos_tr, 0,    0,     0,     0, 1./45**2),
                 gm2.util.multipole(pos_tr, 0,    0,     0,     0,      0, 1./45**3,     0),
                 gm2.util.multipole(pos_tr, 0,    0,     0,     0,      0,     0, 1./45**3),
                 gm2.util.multipole(pos_tr, 0,    0,     0,     0,      0,     0,     0, 1./45**4,     0),
                 gm2.util.multipole(pos_tr, 0,    0,     0,     0,      0,     0,     0,     0, 1./45**4),
                 gm2.util.multipole(pos_tr, 0,    0,     0,     0,      0,     0,     0,     0,     0, 1./45**5,     0),
                 gm2.util.multipole(pos_tr, 0,    0,     0,     0,      0,     0,     0,     0,     0,     0, 1./45**5),
                 gm2.util.multipole(pos_tr, 0,    0,     0,     0,      0,     0,     0,     0,     0,      0,     0, 1./45**6,     0),
                 gm2.util.multipole(pos_tr, 0,    0,     0,     0,      0,     0,     0,     0,     0,      0,     0,     0, 1./45**6),
                 gm2.util.multipole(pos_tr, 0,    0,     0,     0,      0,     0,     0,     0,     0,      0,     0,     0,     0, 1./45**7,     0),
                 gm2.util.multipole(pos_tr, 0,    0,     0,     0,      0,     0,     0,     0,     0,      0,     0,     0,     0,     0, 1./45**7),
                 gm2.util.multipole(pos_tr, 0,    0,     0,     0,      0,     0,     0,     0,     0,      0,     0,     0,     0,     0,     0, 1./45**8,     0),
                 gm2.util.multipole(pos_tr, 0,    0,     0,     0,      0,     0,     0,     0,     0,      0,     0,     0,     0,     0,     0,     0,  1./45**8),
                 gm2.util.multipole(pos_tr, 0,    0,     0,     0,      0,     0,     0,     0,     0,      0,     0,     0,     0,     0,     0,     0,    0, 1./45**8, 0),
                 gm2.util.multipole(pos_tr, 0,    0,     0,     0,      0,     0,     0,     0,     0,      0,     0,     0,     0,     0,     0,     0,    0,        0, 1./45**8)]).T

mtr0 = np.linalg.pinv(trb_[:,np.array([0])])
mtr1 = np.linalg.pinv(trb_[:,np.array([0,1,2])])
mtr2 = np.linalg.pinv(trb_[:,np.array([0,1,2,3,4])])
mtr3 = np.linalg.pinv(trb_[:,np.array([0,1,2,3,4,5,6])])
mtr4 = np.linalg.pinv(trb_[:,np.array([0,1,2,3,4,5,6,7,8])])
mtr5 = np.linalg.pinv(trb_[:,np.array([0,1,2,3,4,5,6,7,8,9,10])])
mtr6 = np.linalg.pinv(trb_[:,np.array([0,1,2,3,4,5,6,7,8,9,10,11,11])])
mtr6[12,:] = 0.0
mtr7 = np.linalg.pinv(trb_[:,np.array([0,1,2,3,4,5,6,7,8,9,10,11,11,13,14])])
mtr7[12,:] = 0.0
mtr8 = np.linalg.pinv(trb_[:,np.array([0,1,2,3,4,5,6,7,8,9,10,11,11,13,14,15,16])])
mtr8[12,:] = 0.0
mtr9 = np.linalg.pinv(trb_[:,np.array([0,1,2,3,4,5,6,7,8,9,10,11,11,13,14,15,16,17,18])])
mtr9[12,:] = 0.0
mtr = [mtr0, mtr1, mtr2, mtr3, mtr4, mtr5, mtr6, mtr7, mtr8, mtr9]


ip = gm2.Interpolation(runs)
freq_c = ip.tr.freq + ip.tr.getCalibration()
s = ((freq_c[:,0]>52000)&(freq_c[:,0]<53000))
freq_c = freq_c[s,:]
n = freq_c.shape[0]
mp_raw_c = []
for mp_max in range(10):
    mp_raw_c.append(np.zeros([n, mp_max*2 + 1]))
    for ev in range(n):
        mp_raw_c[-1][ev,:] = mtr[mp_max] @ freq_c[ev,:]  



at = 45.
r = np.array([at**0, at**1, at**1, at**2, at**2, at**3, at**3, at**4, at**4, at**5, at**5, at**6, at**6, at**7, at**7, at**8, at**8, at**9, at**9])


fs = gm2.plotutil.figsize() 
plt.figure(figsize=[fs[0]*1.5,fs[1]])
plt.plot((ip.tr.time[s,8]-t0)/(3600*1e9), mp_raw_c[1][:,0]*gm2.HZ2PPM, '.', markersize=2)
plt.xlabel("time [h]")
plt.ylabel("dipole ppm")
gm2.despine()
plt.show()

t0 = ip.tr.time[ ip.tr.time>1e8].min()
for i,mp_max in enumerate(range(1,18,2)):
    res = np.zeros([mp_raw_c[0].shape[0], 17])
    for ev in range(mp_raw_c[0].shape[0]):
        res[ev, :] = gm2.util.multipole((gm2.TR.probes.position.r, gm2.TR.probes.position.theta), *(mp_raw_c[i][ev,:]/r[:mp_max])) - freq_c[ev,:]
    plt.figure(figsize=[fs[0]*1.5,fs[1]])
    ax = plt.subplot(211) 
    plt.plot((ip.tr.time[s,:]-t0)/(3600*1e9), res*gm2.HZ2PPM, '.', markersize=2)
    #plt.xlabel("time [h]")
    plt.ylabel("residual [ppm]")
    plt.title("n <= %i" % (mp_max//2))
    #plt.ylim([-3,2])
    plt.subplot(212, sharex=ax)
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.plot((ip.tr.time[s,:]-t0)/(3600*1e9), (res-res.mean(axis=0)) *gm2.HZ2PPM, '.', markersize=2, alpha=0.2)
    plt.xlabel("time [h]")
    plt.ylabel("residual \n variations [ppm]")
    plt.plot([0, (ip.tr.time.max()-t0)/(3600*1e9)],[0.0, 0.0],'k--')
    #plt.ylim([-0.2,0.2])
    gm2.despine()
    plt.savefig("plots/residuals_m_n%i.png" % (mp_max//2))
    plt.show()



mp = 2
res = np.zeros([mp_raw_c[0].shape[0], 17])
for ev in range(mp_raw_c[0].shape[0]):
    res[ev, :] = gm2.util.multipole((gm2.TR.probes.position.r, gm2.TR.probes.position.theta), *(mp_raw_c[mp][ev,:]/r[:mp*2+1])) - freq_c[ev,:]

plt.figure(figsize=[fs[0]*2,fs[1]*1.5])
for probe in range(17):
    plt.subplot(6, 3, probe+1)
    plt.plot((ip.tr.time[s,0]-t0)/(3600*1e9), (res[:,probe]-res[3000:4000,probe].mean()) *gm2.HZ2PPB, '.', markersize=2, alpha=0.2)
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
mp_max = 9
for mp in range(mp_max):
    for ev in range(mp_raw_c[0].shape[0]):
        res[ev, :] = gm2.util.multipole((gm2.TR.probes.position.r, gm2.TR.probes.position.theta), *(mp_raw_c[mp][ev,:]/r[:mp*2+1])) - freq_c[ev,:]
    res_mean.append(np.nanmean(res, axis=0))
    res_std.append(np.nanstd(res, axis=0))
    chi2.append(np.sqrt((res**2).sum(axis=1)))

plt.figure(figsize=[fs[0]*2,fs[1]*1.5])
ax = plt.subplot(311)
plt.errorbar(np.arange(mp_max), [res_mean[i][0]*gm2.HZ2PPM for i in range(mp_max)], [res_std[i][0]*gm2.HZ2PPM for i in range(mp_max)],label="#1", fmt='x-')
plt.plot([0,mp_max-1],[0,0],'--k',linewidth=0.5)
plt.legend()
plt.ylabel("residual [ppm]")
plt.setp(ax.get_xticklabels(), visible=False)
plt.subplot(312, sharex=ax)
for j in range(1,5,1):
    plt.errorbar(np.arange(mp_max), [res_mean[i][j]*gm2.HZ2PPM for i in range(mp_max)], [res_std[i][j]*gm2.HZ2PPM for i in range(mp_max)],label="#%i" % (j+1), fmt='x-')
plt.plot([0,mp_max-1],[0,0],'--k',linewidth=0.5)
plt.legend(ncol=4)
plt.setp(ax.get_xticklabels(), visible=False)
plt.ylabel("residual [ppm]")
plt.subplot(313, sharex=ax)
for j in range(5,17,1):
    plt.errorbar(np.arange(mp_max), [res_mean[i][j]*gm2.HZ2PPM for i in range(mp_max)], [res_std[i][j]*gm2.HZ2PPM for i in range(mp_max)], label="#%i" % (j+1),fmt='x-')
plt.plot([0,mp_max-1],[0,0],'--k',linewidth=0.5)
plt.legend(ncol=6)
plt.subplots_adjust(hspace=0)
plt.xlabel("n")
plt.ylabel("residual [ppm]")
gm2.sns.despine()
plt.show()


plt.figure(figsize=[fs[0]*2,fs[1]*1.5])
ax = plt.subplot(211)
plt.errorbar(np.arange(mp_max), [mp_raw_c[i].mean(axis=0)[0] for i in range(mp_max)], [mp_raw_c[i].std(axis=0)[0] for i in range(mp_max)])
plt.setp(ax.get_xticklabels(), visible=False)
plt.subplot(212, sharex=ax)
for mp in range(1,mp_max):
    plt.errorbar(np.arange(mp,mp_max), [mp_raw_c[i].mean(axis=0)[mp*2] for i in range(mp,9)], [mp_raw_c[i].std(axis=0)[mp*2] for i in range(mp,mp_max)])
    plt.errorbar(np.arange(mp,mp_max), [mp_raw_c[i].mean(axis=0)[mp*2-1] for i in range(mp,9)], [mp_raw_c[i].std(axis=0)[mp*2-1] for i in range(mp,mp_max)])
plt.show()



plt.figure(figsize=[fs[0]*2,fs[1]*1.5])
for mp in range(9):
    plt.plot((ip.tr.time[s,0]-t0)/(3600*1e9), chi2[mp] * gm2.HZ2PPM, '.',  markersize=2, label="n <= %i" % mp)
plt.legend()
plt.xlabel("time [h]")
plt.ylabel(r"$\chi^2 [ppm]$")
gm2.despine()
plt.show()

mp = 9
cc = np.array([c.mean() for c in chi2])
plt.plot(np.arange(mp), cc[:mp] * gm2.HZ2PPM, 'x')
plt.xlabel("n >= ")
plt.ylabel(r"$\chi^2$ [ppm]")
plt.gca().set_xticklabels(np.arange(mp))
plt.gca().set_xticks(np.arange(mp))
gm2.despine()
plt.show()



ip2 = gm2.Interpolation([3997])
ip2.loadTrySpikes()
freq2_c = ip2.trFreqSmooth + ip2.tr.getCalibration()



mp_max=10
n2 = freq2_c.shape[0]
mp2_raw_c = []
for mp_max in range(10):
    mp2_raw_c.append(np.zeros([n2, mp_max*2 + 1]))
    for ev in range(n2):
        mp2_raw_c[-1][ev,:] = mtr[mp_max] @ freq2_c[ev,:]


isOutl = ip2.spk[0].isOutl()|ip2.spk[1].isOutl()|ip2.spk[2].isOutl()|ip2.spk[3].isOutl()|ip2.spk[4].isOutl()|ip2.spk[5].isOutl()|ip2.spk[6].isOutl()|ip2.spk[7].isOutl()|ip2.spk[8].isOutl()|ip2.spk[9].isOutl()|ip2.spk[10].isOutl()|ip2.spk[11].isOutl()|ip2.spk[12].isOutl()|ip2.spk[13].isOutl()|ip2.spk[14].isOutl()|ip2.spk[15].isOutl()|ip2.spk[16].isOutl()

chi2 = []
res_ = []
for i, mp_max in enumerate(range(1,18,2)):
    res = np.zeros([mp2_raw_c[0].shape[0], 17])
    for ev in range(mp2_raw_c[0].shape[0]):
        res[ev, :] = gm2.util.multipole((gm2.TR.probes.position.r, gm2.TR.probes.position.theta), *(mp2_raw_c[i][ev,:]/r[:mp_max])) - freq2_c[ev,:]
    res_.append(res)
    chi2.append(np.sqrt((res**2).sum(axis=1)))
    plt.figure(figsize=[fs[0]*1.5,fs[1]])
    ax = plt.subplot(311)
    plt.plot(ip2.tr.azi[10:,0][~isOutl[10:]]/np.pi*180., res[10:,0][~isOutl[10:]]*gm2.HZ2PPM, '.', markersize=2)
    #plt.xlabel("time [h]")
    #plt.ylabel("residual [ppm]")
    plt.ylim([-2,2])
    plt.title("n <= %i" % (mp_max//2))
    plt.setp(ax.get_xticklabels(), visible=False)
    #plt.ylim([-3,2])
    plt.subplot(312, sharex=ax)
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.plot(ip2.tr.azi[10:,0][~isOutl[10:]]/np.pi*180, (res[10:,1:5][~isOutl[10:]]) *gm2.HZ2PPM, '.', markersize=2, alpha=0.2)
    plt.ylabel("residual [ppm]")
    plt.ylim([-1,1])
    plt.subplot(313, sharex=ax)
    plt.plot(ip2.tr.azi[10:,0][~isOutl[10:]]/np.pi*180, (res[10:,5:17][~isOutl[10:]]) *gm2.HZ2PPM, '.', markersize=2, alpha=0.2)
    plt.ylim([-1,1])
    plt.xlabel("azimuth [deg]")
    #plt.ylabel("residual [ppm]")
#plt.plot([0, (ip.tr.time.max()-t0)/(3600*1e9)],[0.0, 0.0],'k--')
    #plt.ylim([-0.2,0.2])
    plt.subplots_adjust(hspace=0)
    gm2.sns.despine()
    #plt.savefig("plots/residuals_m_fieldmap_n%i.png" % (mp_max//2))
    plt.show()


for mp in range(9):
    plt.hist(res_[mp][:,0]*gm2.HZ2PPM, bins=np.arange(-1.5,1.5,1/62.), histtype='step', label="n=%i" % mp)
plt.title("#1")
plt.legend()
plt.ylim([0,200])
gm2.despine()
plt.savefig("plots/residuals_m_fieldmap_hist_ch1.png")
plt.show()


for p in range(1,5):
    plt.subplot(2,2,p)
    for mp in range(9):
        plt.hist(res_[mp][:,1]*gm2.HZ2PPM, bins=np.arange(-1.5,1.5,1/62.), histtype='step', label="n=%i" % mp)
    plt.title("#%i" % (p+1))
    #plt.legend()
    plt.ylim([0,200])
gm2.despine()
plt.savefig("plots/residuals_m_fieldmap_hist_ch2to5.png")
plt.show()

plt.figure(figsize=[fs[0]*1.5,fs[1]*1.5])
for p in range(5,17):
    plt.subplot(3,4,p-4)
    for mp in range(9):
        plt.hist(res_[mp][:,1]*gm2.HZ2PPM, bins=np.arange(-1.5,1.5,1/62.), histtype='step', label="n=%i" % mp)
    plt.title("#%i" % (p+1))
    #plt.legend()
    plt.ylim([0,200])
gm2.despine()
plt.savefig("plots/residuals_m_fieldmap_hist_ch6to17.png")
plt.show()


plt.figure(figsize=[fs[0]*1.5,fs[1]*1.0])
for i in range(9):
    plt.plot(ip2.tr.azi[10:,0][~isOutl[10:]]/np.pi*180, chi2[i][10:][~isOutl[10:]]*gm2.HZ2PPM,'.',markersize=2, label="n=%i" % i)
plt.xlabel("azimuth [deg]")
plt.ylabel(r"$\chi^2$ [ppm]")
plt.legend(ncol=2)
gm2.despine()
plt.savefig("plots/chi2_m_fieldmap.png")
plt.show()

