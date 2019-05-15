import gm2
from gm2 import plt, np

runs = [3997]
#plt.errorbar(phi2_, [res2_[i][0] for i in range(len(phi2_))], xerr=np.pi*2./m2, fmt=' ')
#plt.plot(phi3_, [res3_[i][0] for i in range(len(phi3_))])
tr = gm2.Trolley(runs, True)
spk = [gm2.Spikes(tr.azi[:,probe], tr.freq[:,probe], gm2.PPM_HZ/2.) for probe in range(17)]

skip = 7



''' Study Cut Dependency '''
probe = 0
res_ = []
ths = np.array([1./10, 1./8, 1./4, 3./8,  1./2, 5./8, 3./4, 7./8., 1, 1.25, 1.5, 1.75, 2, 4, 6, 8])*gm2.PPM_HZ
Ns = [500, 600, 700]
for i, th in enumerate(ths):
    res_.append([])
    for j, N in enumerate(Ns):
        print(i,j)
        s_  = np.abs(spk[probe].outl) < th 
        s_[:skip] = False
        fr_ = gm2.Fourier(tr.azi[s_, probe], tr.freq[s_, probe], N)
        data_ = (fr_.B(tr.azi[s_, probe]) - tr.freq[s_, probe])
        popt_, _ = gm2.plotutil.histWithGauss(plt.gca(), data_, bins=np.arange(-100,100,2), orientation='vertical', nsigma=nsigma, alpha=1.0) 
        #plt.show()
        res_[i].append(popt_[2])

res_ = np.array(res_)
plt.gcf()
for i,n in enumerate(Ns):
    plt.plot(ths, res_[:,i], '.', label="N=%i"%n, color=gm2.sns.color_palette()[i])
    plt.plot(ths, res_[:,i], '-', alpha=0.5, color=gm2.sns.color_palette()[i])
#plt.plot([gm2.PPM_HZ*1.5, gm2.PPM_HZ*1.5],[10,17],'--', color=gm2.sns.color_palette()[i+1])
plt.plot([ths[8]], [res_[8][0]], 'o', color=gm2.sns.color_palette()[i+1], label="used cuts")
plt.xlabel(r"$th_{\rm{spikes}}$ [Hz]")
plt.ylabel("resolution probe #1 [Hz]")
plt.legend()
gm2.despine()
plt.show()


N = 500
nsigma = 3.0
th = 1.0*gm2.PPM_HZ # Hz

s = [np.abs(spk[probe].outl) < th for probe in range(17)]
sall = s[0]
for p in range(17):
    s[p][:skip] = False
    sall = sall&s[p]


#fr = [gm2.Fourier(tr.azi[s[probe], probe], tr.freq[s[probe], probe], N) for probe in range(17)]
#data = [fr[probe].B(tr.azi[s[probe], probe]) - tr.freq[s[probe],probe] for probe in range(17)]

fr = [gm2.Fourier(tr.azi[sall, probe], tr.freq[sall, probe], N) for probe in range(17)]
data = [fr[probe].B(tr.azi[sall, probe]) - tr.freq[sall, probe] for probe in range(17)]

popt = []
ss = np.full(data[0].shape, True)
for probe in range(17):
    popt_, _ = gm2.plotutil.histWithGauss(plt.gca(), data[probe], bins=np.arange(-100,100,2), orientation='vertical', nsigma=nsigma, alpha=1.0) 
    popt.append(popt_)
    plt.title("# %i" % (probe+1))
    if probe in [0]:
        plt.xlabel("residuals [Hz]")
        plt.plot([popt_[1] - nsigma * np.abs(popt_[2]), popt_[1] - nsigma * np.abs(popt_[2])], [0, 300], '--', color=gm2.sns.color_palette()[2])
        plt.plot([popt_[1] + nsigma * np.abs(popt_[2]), popt_[1] + nsigma * np.abs(popt_[2])], [0, 300], '--', color=gm2.sns.color_palette()[2])
        gm2.despine()
        plt.show()
    plt.clf()
    s_ = (data[probe] > popt_[1] - nsigma * np.abs(popt_[2]) )&( data[probe] < popt_[1] + nsigma * np.abs(popt_[2]) )
    ss = ss&s_
    #print("DEBUG", data[probe][s_].std(), popt_[2])

phi_cov  = tr.azi[sall, 8][ss]
data     = np.array(data)
data_cov = data[:,ss]

phi_ = []
res_ = []
cov_ = []
m = 12*4 #size of mocing window
n = m*4 # moving window steps
for i in range(n):
    l = phi_cov.min()+np.pi*2./n * i
    u = phi_cov.min()+np.pi*2./n * i + np.pi*2./m
    sss = (phi_cov > l)&(phi_cov < u)
    phi_.append((l+u)/2)
    res_.append(np.sqrt(np.diag((np.cov(data_cov[:,sss])))))
    cov_.append(np.corrcoef(data_cov[:,sss])[:,0])


fs = gm2.plotutil.figsize()    

plt.figure(figsize=[fs[0]*1.5, fs[1]])
r = np.sqrt(np.diag((np.cov(data_cov))))
for i in range(17):
    plt.plot(phi_,  [res_[j][i] for j in range(len(phi_))], color=gm2.sns.color_palette()[i%10])
    plt.plot([np.max(phi_), np.max(phi_)+0.3], [r[i], r[i]], '--', color=gm2.sns.color_palette()[i%10])
    plt.plot([np.min(phi_)-0.3, np.min(phi_)], [r[i], r[i]], '--', color=gm2.sns.color_palette()[i%10])
plt.xlabel("azimuth [rad]")
plt.ylabel("probe resolution [Hz]")
gm2.despine()
plt.show()


plt.figure(figsize=[fs[0]*1.5, fs[1]*1.5])
ax = []
ax.append(plt.subplot2grid((17, 17), (0, 0)))
ax[-1].xaxis.set_ticklabels([]) 
ax[-1].yaxis.set_ticklabels([]) 
ax[-1].xaxis.set_ticks([])      
ax[-1].yaxis.set_ticks([])
plt.title("1")
ax[-1].set_ylabel("1")

ax.append(plt.subplot2grid((17, 17), (16, 16)))
ax[-1].xaxis.set_ticklabels([]) 
ax[-1].yaxis.set_ticklabels([]) 
ax[-1].xaxis.set_ticks([])      
ax[-1].yaxis.set_ticks([])

cs = gm2.sns.color_palette('Blues',10)
cs = gm2.sns.color_palette('Spectral_r',40)
cov = np.corrcoef(data_cov)

for p1 in range(0,17):
    for p2 in range(p1+1,17):
       ax.append(plt.subplot2grid((17, 17), (p1, p2)))
       ax[-1].plot(data_cov[p1,:], data_cov[p2,:], '.', markersize=2, alpha=0.1)
       ax[-1].set_xlim([-70,70])
       ax[-1].set_ylim([-70,70])
       ax[-1].xaxis.set_ticklabels([]) 
       ax[-1].yaxis.set_ticklabels([]) 
       ax[-1].xaxis.set_ticks([]) 
       ax[-1].yaxis.set_ticks([]) 
       if p1 in [0]:
           plt.title("%i" % (p2+1))
    for p2 in range(0, p1):
       ax.append(plt.subplot2grid((17, 17), (p1, p2)))
       if p2 in [0]:
          ax[-1].set_ylabel("%i" % (p1+1))
       ax[-1].xaxis.set_ticklabels([]) 
       ax[-1].yaxis.set_ticklabels([]) 
       ax[-1].xaxis.set_ticks([])      
       ax[-1].yaxis.set_ticks([])   
       ax[-1].text(0.5,0.5,"%.2f" % cov[p1,p2], horizontalalignment='center', verticalalignment='center', fontsize=8)
       ax[-1].set_facecolor(cs[int(np.floor(cov[p1,p2]*40)) ])

plt.subplots_adjust(wspace=0, hspace=0)
plt.show()


pos_fp = (gm2.FP.probes.position.r, gm2.FP.probes.position.theta)
pos_tr = (gm2.TR.probes.position.r, gm2.TR.probes.position.theta)

trb  = np.array([gm2.util.multipole(pos_tr, 1,     0,    0),
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
                 gm2.util.multipole(pos_tr, 0,    0,     0,     0,      0,     0,     0,     0,     0,      0,     0,     0,     0,     0,     0,     0, 1./45**8)]).T

trb[:,12] = 0.0

e   = np.cov(data_cov)
e_d = np.diag(np.diag(np.cov(data_cov)))
e_a = np.diag(np.ones([17])*(275.0*gm2.PPB2HZ)**2)


for n in range(1,18,2):
    cov   = (np.linalg.pinv((trb[:,:n].T @ np.linalg.pinv(e) @ trb[:, :n])))
    cov_d = (np.linalg.pinv((trb[:,:n].T @ np.linalg.pinv(e_d) @ trb[:, :n])))
    cov_a = (np.linalg.pinv((trb[:,:n].T @ np.linalg.pinv(e_a) @ trb[:, :n])))
    print(np.array_str(np.sqrt(np.diag(cov)),  precision=3, max_line_width=10000).replace(". ", "& ").replace("nan","-&").replace(".]","\\\\").replace("[","cor & %i &" % (n//2)))
    print(np.array_str(np.sqrt(np.diag(cov_d)),precision=3, max_line_width=10000).replace(". ", "& ").replace("nan","-&").replace(".]","\\\\").replace("[","dia & %i &" % (n//2)))
    print(np.array_str(np.sqrt(np.diag(cov_a)),precision=3, max_line_width=10000).replace(". ", "& ").replace("nan","-&").replace(".]","\\\\").replace("[","uni & %i &" % (n//2)))

    ''' plot the fit error and correlations '''
    allTypes = True
    cs = gm2.sns.color_palette('Spectral_r',40)
    plt.figure(figsize=[fs[0]*1.5, fs[1]*1.5])
    ax = []
    cov_ = cov
    cor_ = cov_ / np.sqrt(np.diag(cov_)[:,None] * np.diag(cov_))
    t = np.arange(0, 2 * np.pi, 0.01)
    for i in range(n):
        for j in range(n):
            ax.append(plt.subplot2grid((17, 17), (i, j)))
            ax[-1].xaxis.set_ticklabels([]) 
            ax[-1].yaxis.set_ticklabels([]) 
            ax[-1].xaxis.set_ticks([])      
            ax[-1].yaxis.set_ticks([])   
            if i == j:
                 ax[-1].text(0.5,0.5,"%.2f" % np.sqrt(cov_[i,j]), horizontalalignment='center', verticalalignment='center', fontsize=10)
                 if i in [0]:
                     ax[-1].set_title(r"$B_{0}$")
                     ax[-1].set_ylabel(r"$B_{0}$")
            if j > i:
                w, v = np.linalg.eig(cov_[[i,j]][:,[i,j]])
                xx = []
                for y in (np.array([np.cos(t), np.sin(t)]).T * np.sqrt(w)):
                     xx.append(v @ y)
                xx = np.array(xx)
                ax[-1].plot(xx[:,1], xx[:,0])
                if allTypes:
                    w_d, v_d = np.linalg.eig(cov_d[[i,j]][:,[i,j]])
                    xx_d = []
                    w_a, v_a = np.linalg.eig(cov_a[[i,j]][:,[i,j]])
                    xx_a = []
                    for y in (np.array([np.cos(t), np.sin(t)]).T * np.sqrt(w_d)):
                        xx_d.append(v_d @ y)
                    for y in (np.array([np.cos(t), np.sin(t)]).T * np.sqrt(w_a)):
                        xx_a.append(v_a @ y)
                    xx_d = np.array(xx_d)
                    xx_a = np.array(xx_a)
                    ax[-1].plot(xx_d[:,1], xx_d[:,0])
                    ax[-1].plot(xx_a[:,1], xx_a[:,0])
                ax[-1].set_xlim([-30,30])
                ax[-1].set_ylim([-30,30])
                if i in [0]:
                    if j%2 == 0:
                        ax[-1].set_title(r"$b_{%i}$" % ((j+1)//2) )
                    else:
                        ax[-1].set_title(r"$a_{%i}$" % ((j+1)//2) )
            if j < i:
                ax[-1].text(0.5,0.5,"%.2f" % cor_[i,j], horizontalalignment='center', verticalalignment='center', fontsize=10)
                if ~np.isnan(cor_[i,j]):
                    ax[-1].set_facecolor(cs[int(np.floor(cor_[i,j]*20))+20 ])
                #else:
                #    ax[-1].set_facecolor(cs[int(np.floor(cor_[i,j]*20))+20 ])
                if j in [0]:
                    if i%2 == 0:
                        ax[-1].set_ylabel(r"$b_{%i}$" % ((i+1)//2) )
                    else:
                        ax[-1].set_ylabel(r"$a_{%i}$" % ((i+1)//2) )

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig("plots/mp_coef_cor_%i.png" % (i//2))
    plt.savefig("plots/mp_coef_cor_%i.pdf" % (i//2))
    plt.show()



''' Plot the field map uncertainty '''
def f(r, phi, n, r0=45.): 
    if type(r) is not np.ndarray:  
        r   = np.array([r])
        phi = np.array([phi])
    if r.shape[0] != phi.shape[0]:
        raise Exception('Dimension of r and phi have to be the samne.')
    ii = np.arange(1,n+1)
    return np.concatenate([np.ones([r.shape[0], 1]), 
                           np.insert((((r[:,None]/r0)**ii) * np.sin(ii * phi[:,None])) ,np.arange(n),
                                     (((r[:,None]/r0)**ii) * np.cos(ii * phi[:,None])) , axis=1)], axis=1) 


for N in range(1,18,2):
    cov   = (np.linalg.pinv((trb[:,:N].T @ np.linalg.pinv(e)   @ trb[:, :N])))
    cov_d = (np.linalg.pinv((trb[:,:N].T @ np.linalg.pinv(e_d) @ trb[:, :N])))
    cov_a = (np.linalg.pinv((trb[:,:N].T @ np.linalg.pinv(e_a) @ trb[:, :N])))

    mm = 50.
    xx, yy = np.meshgrid(np.arange(-mm, mm), np.arange(-mm, mm))
    rr  = np.sqrt(xx**2 + yy**2)
    phi = np.arctan2(yy, xx)
    s2 = np.zeros_like(xx)
    s2_d = np.zeros_like(xx)
    s2_a = np.zeros_like(xx)
    for i in range(rr.shape[0]):
        for j in range(rr[0].shape[0]):
            s2[i,j]   = f(rr[i,j], phi[i,j], N//2) @ cov   @ f(rr[i,j], phi[i,j], N//2).T
            s2_d[i,j] = f(rr[i,j], phi[i,j], N//2) @ cov_d @ f(rr[i,j], phi[i,j], N//2).T
            s2_a[i,j] = f(rr[i,j], phi[i,j], N//2) @ cov_a @ f(rr[i,j], phi[i,j], N//2).T


    step = 50 # ppb
    names = ["cor","dia","uni"]
    for ii, s2_ in enumerate([s2, s2_d,s2_a]):
    #for ii, s2_ in enumerate([s2]):
        ff = gm2.plt.figure(figsize=(7, 6))
        from matplotlib.colors import ListedColormap
        s2_[rr>mm] = np.nan
        nmax = 1000
        if nmax is None:
            nmax = ((gm2.np.ceil(np.nanmax(np.sqrt(s2_)))*gm2.HZ2PPB)//step+1)*step
            nmax = np.min([nmax, 1000])
        nmin = 0
        if nmin is None:
            nmin = ((gm2.np.floor(np.nanmin(np.sqrt(s2_)))*gm2.HZ2PPB)//step)*step
        nc = int((nmax-nmin)//step)
        cmap = ListedColormap(gm2.sns.color_palette("Spectral_r", nc))
        #cmap = ListedColormap(gm2.sns.color_palette("Blues", nc))
        levels_ = gm2.np.arange(nmin, nmax+(nmax-nmin)/nc, step)
        cs = gm2.plt.contour(xx, yy, np.sqrt(s2_)*gm2.HZ2PPB,  levels=levels_, colors='k', linewidths=1.0, inline=True, alpha=0.5)
        plt.clabel(cs, inline=1, fontsize=10)
        cf = gm2.plt.contourf(xx, yy, np.sqrt(s2_)*gm2.HZ2PPB, levels=levels_, cmap=cmap, extend='max')
        cb = gm2.plt.colorbar()
        cb.ax.set_title("[ppb]")
        gm2.plt.plot(gm2.TR.probes.position.x, gm2.TR.probes.position.y, 'x', color='black', alpha=0.3)
        t = np.arange(0, np.pi*2,0.01)
        r_ = 45.
        plt.plot(r_*np.sin(t), r_*np.cos(t),color='black', alpha=0.3, linewidth=2)

        gm2.plt.xlim([-mm, mm])
        gm2.plt.ylim([-mm, mm])
        gm2.plt.xlabel("x [mm]")
        gm2.plt.ylabel("y [mm]")
        plt.title(r"$n_{\rm{max}}=%i$, %s" % (N//2, names[ii]))
        gm2.plt.tight_layout()
        plt.savefig("plots/fieldmap_uncertenty_n%i_%s.png" % (N//2, names[ii]))
        plt.savefig("plots/fieldmap_uncertenty_n%i_%s.pdf" % (N//2, names[ii]))
        plt.show()



w_raw = tr.freq + tr.getCalibration()
calib = tr.getCalibration()[0] #np.zeros([17])
freqAt      = [gm2.util.interp1d(tr.azi[10:,probe],  tr.freq[10:,probe]  + calib[probe], fill_value='extrapolate') for probe in range(17)]
freqAt_sall = [gm2.util.interp1d(tr.azi[sall,probe], tr.freq[sall,probe] + calib[probe], fill_value='extrapolate') for probe in range(17)]
#w_interpol       = np.array([freqAt[probe](tr.azi[34:-23,8])      for probe in range(17)]).T # range needs to be adjusted to moving part
#w_interpol_sall  = np.array([freqAt_sall[probe](tr.azi[34:-23,8]) for probe in range(17)]).T # range needs to be adjusted to moving part
w_interpol       = np.array([freqAt[probe](tr.azi[:,8])      for probe in range(17)]).T
w_interpol_sall  = np.array([freqAt_sall[probe](tr.azi[:,8]) for probe in range(17)]).T
w = w_interpol_sall
nev = w.shape[0]

m = []
m_d = []
m_a = []
res   = np.zeros([9, nev, 17], dtype=float)
res_d = np.zeros_like(res)
res_a = np.zeros_like(res)
chi2   = np.zeros([9, nev], dtype='float')
chi2_d = np.zeros_like(chi2)
chi2_a = np.zeros_like(chi2)

peff= np.zeros([9,3],dtype=float)

for n in range(1,18,2):
    print("n=", n)
    M   = (np.linalg.pinv((trb[:,:n].T @ np.linalg.pinv(e)   @ trb[:, :n]))) @ trb[:,:n].T @ np.linalg.pinv(e)
    M_d = (np.linalg.pinv((trb[:,:n].T @ np.linalg.pinv(e_d) @ trb[:, :n]))) @ trb[:,:n].T @ np.linalg.pinv(e_d)
    M_a = (np.linalg.pinv((trb[:,:n].T @ np.linalg.pinv(e_a) @ trb[:, :n]))) @ trb[:,:n].T @ np.linalg.pinv(e_a)
    peff[(n//2),0] = np.trace(trb[:, :n] @ M) 
    peff[(n//2),1] = np.trace(trb[:, :n] @ M_d) 
    peff[(n//2),2] = np.trace(trb[:, :n] @ M_a) 

    m_   = np.zeros([nev, n], dtype=float)
    m_d_ = np.zeros_like(m_)
    m_a_ = np.zeros_like(m_)

    for ev in range(nev):
        m_[ev,:]   = M   @ w[ev,:]  
        m_d_[ev,:] = M_d @ w[ev,:]  
        m_a_[ev,:] = M_a @ w[ev,:]  
        res[(n//2), ev,:]   = (w[ev,:] - trb[:, :n] @ m_[ev,  :])
        res_d[(n//2), ev,:] = (w[ev,:] - trb[:, :n] @ m_d_[ev,:])
        res_a[(n//2), ev,:] = (w[ev,:] - trb[:, :n] @ m_a_[ev,:])
        chi2[(n//2), ev]    = res[(n//2),   ev,:].T @ np.linalg.pinv(e)   @ res[(n//2),   ev,:]
        chi2_d[(n//2), ev]  = res_d[(n//2), ev,:].T @ np.linalg.pinv(e_d) @ res_d[(n//2), ev,:]
        chi2_a[(n//2), ev]  = res_a[(n//2), ev,:].T @ np.linalg.pinv(e_a) @ res_a[(n//2), ev,:]
    m.append(m_)
    m_d.append(m_d_)
    m_a.append(m_a_)


nlim = 200
for n in range(9):
    lim = chi2[n,:].mean()
    v  , _, _ = plt.hist(chi2[n,sall],   bins=np.arange(0,lim, lim/nlim), histtype='step', label="cor")
    v_d, _, _ = plt.hist(chi2_d[n,sall], bins=np.arange(0,lim, lim/nlim), histtype='step', label="diag")
    v_a, _, _ = plt.hist(chi2_a[n,sall], bins=np.arange(0,lim, lim/nlim), histtype='step', label="uniform")
    tt = np.arange(0, lim, lim/nlim/2.)
    chi2_ = stats.chi2.pdf(tt,17.-peff[n,0]) 
    plt.plot(tt, chi2_/chi2_.max()*np.max([v.max(), v_d.max(), v_a.max()])*0.8, '-', label=r"$\chi^2$(d.o.f.= %.1f)" % (17.-peff[n,0]))
    plt.xlabel(r"$\chi^{2}$")
    plt.title(r"$n=%i$" % (n))
    plt.legend()
    gm2.despine()
    plt.savefig("plots/chi2_n%i.png" % n)
    plt.savefig("plots/chi2_n%i.pdf" % n)
    plt.show()


n = 6
fig = plt.figure(figsize=[fs[0]*1.5, fs[1]])
plt.plot(tr.azi[sall,8], chi2[n,sall],  '.', markersize=2, alpha=1,label="cor")
plt.plot(tr.azi[sall,8], chi2_d[n,sall],'.', markersize=2, alpha=1, label="diag")
plt.plot(tr.azi[sall,8], chi2_a[n,sall],'.', markersize=2, alpha=1, label="uniform")
plt.xlabel("azimuth [rad]")
plt.ylabel("$\chi^2$")
plt.title(r"$n=%i$" % n)
plt.legend()
ax2 = fig.add_axes([0.2, 0.55, 0.3, 0.35])
s = (tr.azi[:,8] > 0.1)&(tr.azi[:,8] < 0.22)&sall
plt.plot(tr.azi[s,8], chi2[n,s],  '.', markersize=2, alpha=1,label="cor")
plt.plot(tr.azi[s,8], chi2_d[n,s],'.', markersize=2, alpha=1, label="diag")
plt.plot(tr.azi[s,8], chi2_a[n,s],'.', markersize=2, alpha=1, label="uniform")
ax2.set_ylim([-20,250])
gm2.despine()

plt.savefig("plots/chi2VsAzi_n%i.png" % n)
plt.savefig("plots/chi2VsAzi_n%i.pdf" % n)
plt.show()


i#s_ = azimuth < tr.azi[sall,8].min()
azimuth[azimuth < tr.azi[sall,8].min()] += 2*np.pi

''' plot residuals '''
fp = False
res_  = res
chi2_ = chi2
m_    = m
suffix_ = "cor"
lims = [100,100,100]
#for n in range(9):
for n in [6]:   
    plt.figure(figsize=[fs[0]*1.75,fs[1]*1.5])
    ax1_1 = plt.subplot2grid((5, 6), (0, 0), colspan=5)
    ax2_1 = plt.subplot2grid((5, 6), (1, 0), colspan=5, sharex=ax1_1)
    ax3_1 = plt.subplot2grid((5, 6), (2, 0), colspan=5, sharex=ax1_1)
    ax4_1 = plt.subplot2grid((5, 6), (3, 0), colspan=5, sharex=ax1_1)
    ax5_1 = plt.subplot2grid((5, 6), (4, 0), colspan=5, sharex=ax1_1)
    ax3_2 = plt.subplot2grid((5, 6), (2, 5), sharey=ax3_1)
    ax4_2 = plt.subplot2grid((5, 6), (3, 5), sharey=ax4_1)
    ax5_2 = plt.subplot2grid((5, 6), (4, 5), sharey=ax5_1)
    ax1_1.set_title("n <= %i" % (n))
    ax1_1.plot(tr.azi[sall,8]/np.pi*180., m_[n][sall,0],  '.', markersize=2, alpha=1)
    ax1_1.plot(tr.azi[~sall,8]/np.pi*180., m_[n][~sall,0],  '.', markersize=2, alpha=1)
    ax1_1.set_ylabel("$B_0$ [Hz]")
    ax2_1.plot(tr.azi[sall,8]/np.pi*180., chi2_[n,sall],  '.', markersize=2, alpha=1)
    ax2_1.plot(tr.azi[~sall,8]/np.pi*180., chi2_[n,~sall],  '.', markersize=2, alpha=1)
    if fp:
        ax2_1.plot(azimuth/np.pi*180., np.ones_like(azimuth)*100.,'.')
    ax2_1.set_ylabel("$\chi^2$")
    ax3_1.text(80, 80, "central probe #1", horizontalalignment='center', verticalalignment='center', fontsize=12)
    ax3_1.plot(tr.azi[sall,8]/np.pi*180., res_[n,sall,0], '.', markersize=2)
    ax3_1.set_ylim([-lims[0],lims[0]])
    ax4_1.text(80, 80, "inner probes #2-5", horizontalalignment='center', verticalalignment='center', fontsize=12)
    ax4_1.plot(tr.azi[sall,8]/np.pi*180, (res_[n,sall,1:5]), '.', markersize=2, alpha=0.2)
    ax4_1.set_ylabel("residuals [Hz]")
    ax4_1.set_ylim([-lims[1],lims[1]])
    ax5_1.text(80, 80, "outer probes #6-17", horizontalalignment='center', verticalalignment='center', fontsize=12)
    ax5_1.plot(tr.azi[sall,0]/np.pi*180, (res_[n,sall,5:17]), '.', markersize=2, alpha=0.2)
    ax5_1.set_ylim([-lims[2], lims[2]])
    ax5_1.set_xlabel("azimuth [deg]")
    ax3_2.set_xticklabels("")
    ax4_2.set_xticklabels("")
    ax5_2.set_xticklabels("")
    plt.setp(ax3_2.get_yticklabels(), visible=False)
    plt.setp(ax4_2.get_yticklabels(), visible=False)
    plt.setp(ax5_2.get_yticklabels(), visible=False)
    v,_,_,=ax3_2.hist((res_[n,sall,0]), bins=np.arange(-lims[0],lims[0],1), histtype='step', orientation='horizontal') 
    tt = np.arange(-lims[0], lims[0], 1)
    ax3_2.plot(gm2.util.gauss(tt, res_[n,sall,0].shape[0], 0.0, np.sqrt(e[0,0])), tt, '--', color=gm2.sns.color_palette()[0] )                   
    for i in range(1,5):
         ax4_2.hist((res_[n,sall,i]), bins=np.arange(-lims[1],lims[1],1), histtype='step', orientation='horizontal')
    ax4_2.plot(gm2.util.gauss(tt, res_[n,sall,1].shape[0], 0.0, np.sqrt(e[1,1])), tt, '--', color=gm2.sns.color_palette()[0] )                   
    ax4_2.plot(gm2.util.gauss(tt, res_[n,sall,4].shape[0], 0.0, np.sqrt(e[4,4])), tt, '--', color=gm2.sns.color_palette()[3] )                   
    for i in range(5,17):
         ax5_2.hist((res_[n,sall,i]), bins=np.arange(-lims[2],lims[2],1), histtype='step', orientation='horizontal')
 
    ax5_2.plot(gm2.util.gauss(tt, res_[n,sall,5].shape[0], 0.0, np.sqrt(e[5,5])), tt, '--', color=gm2.sns.color_palette()[0] )                   
    ax5_2.plot(gm2.util.gauss(tt, res_[n,sall,14].shape[0], 0.0, np.sqrt(e[14,14])), tt, '--', color=gm2.sns.color_palette()[14-5] )                   
    plt.subplots_adjust(hspace=0)
    gm2.sns.despine()
    #plt.savefig("plots/residuals_n%i_%s.png" % (n, suffix_))
    #plt.savefig("plots/residuals_n%i_%s.pdf" % (n, suffix_))
    plt.show()



lims = 100.0
plt.figure(figsize=[fs[0]*1.75,fs[1]*1.5])
ax = []
tt = np.arange(-lims, lims, 1)
res_ = res_a
suffix_ = "uni"
for p in range(17):
    for n in range(9):
        ax.append(plt.subplot2grid((9, 17), (n,  p)))
        ax[-1].hist(res_[n,sall,p], bins=np.arange(-lims, lims,1), histtype='step', orientation='vertical')
        ax[-1].plot(tt, gm2.util.gauss(tt, res[n,sall,p].shape[0], 0.0, np.sqrt(e[p,p])),  '--' )
        #ax[-1].hist(res_d[n,sall,p], bins=np.arange(-lims, lims,1), histtype='step', orientation='vertical')
        #ax[-1].hist(res_a[n,sall,p], bins=np.arange(-lims, lims,1), histtype='step', orientation='vertical')
        ax[-1].set_xlim([-lims, lims])
        if n < 9:
             plt.setp(ax[-1].get_xticklabels(), visible=False)
        plt.setp(ax[-1].get_yticklabels(), visible=False)
        ax[-1].set_yticks([])
        if n in [0]:
            if p in [8]:
                 ax[-1].set_title("probe\n%i" % (p+1))
            else:
                 ax[-1].set_title("\n%i" % (p+1))
        if p in [0]:
            if n in [4]:
                ax[-1].set_ylabel("$n_{{max}}$ \n $%i$" % n)
            else:
                ax[-1].set_ylabel("\n$%i$" % n)

plt.subplots_adjust(hspace=0, wspace=0)
plt.savefig("plots/residuals_%s.png" % suffix_)
plt.savefig("plots/residuals_%s.pdf" % suffix_)
plt.show()


s = np.abs(((tr.azi[:,8]/np.pi*180.)%10.))<1
plt.plot(tr.azi[s,8]/np.pi*180., chi2[6,s],'.', markersize=2)
plt.plot(tr.azi[~s,8]/np.pi*180., chi2[6,~s],'.', markersize=2)
plt.show()
v,_,_ = plt.hist(chi2[6,s],bins=np.arange(0,25,0.25), histtype='step')
v,_,_ = plt.hist(chi2[6,~s],bins=np.arange(0,25,0.25), histtype='step')
tt = np.arange(0, 25, 0.25)
chi2_ = stats.chi2.pdf(tt, 17.-peff[6,0])
plt.plot(tt, chi2_/np.max(chi2_)*v.max())
plt.show()

plt.plot(tr.azi[sall,8]/np.pi*180., m_d[n][sall,0]-m[n][sall,0],  '.', markersize=2, alpha=1)
plt.plot(tr.azi[sall,8]/np.pi*180., m_a[n][sall,0]-m[n][sall,0],  '.', markersize=2, alpha=1)
plt.plot(tr.azi[~sall,8]/np.pi*180., m_d[n][~sall,0]-m[n][~sall,0],  '.', markersize=2, alpha=1)
plt.plot(tr.azi[~sall,8]/np.pi*180., m_a[n][~sall,0]-m[n][~sall,0],  '.', markersize=2, alpha=1)
gm2.despine()
plt.show()



m_ = m
plt.plot(tr.azi[sall,8]/np.pi*180., m_[n-6][sall,0]-m_[n][sall,0],  '.', markersize=2, alpha=1)
plt.plot(tr.azi[sall,8]/np.pi*180., m_[n-5][sall,0]-m_[n][sall,0],  '.', markersize=2, alpha=1)
plt.plot(tr.azi[sall,8]/np.pi*180., m_[n-4][sall,0]-m_[n][sall,0],  '.', markersize=2, alpha=1)
plt.plot(tr.azi[sall,8]/np.pi*180., m_[n-3][sall,0]-m_[n][sall,0],  '.', markersize=2, alpha=1)
plt.plot(tr.azi[sall,8]/np.pi*180., m_[n-2][sall,0]-m_[n][sall,0],  '.', markersize=2, alpha=1)
plt.plot(tr.azi[sall,8]/np.pi*180., m_[n-1][sall,0]-m_[n][sall,0],  '.', markersize=2, alpha=1)
plt.show()
