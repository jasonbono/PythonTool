import gm2

class FieldMap:
    """Class for Trolley Field Maps.
    
    Attributes:
        azi[sevents] (array(float), optional): azimuth of multipole fits
        m[sevents, N] ( optional): N multipole coefficients 
        res[sevents, 17] ( optional): residuals from multipole fit
        chi2[sevents] ( optional): chi2 from multipole fit
        tr (gm2.Trolley, optional): trolley run data.
        raw_azi[nevents, 17]: the raw azimuths.
        raw_freq[nevents, 17]: the raw frequencies.
        freqAt[probe] (optional): interpolated frequencies from the selected trolley readout cycles (s).
        s[neverns] (bool):  selection of used trolley readout cycles, defaults to all True.
        spk[17] (list(gm2.Spikes), optional): trolley spikes class.
        spk_s[nevents, 17]: selection of good frequencies, per probe.
        fr[17] (list(gm2.Fourier), optional): probe wise fourier fit in phi.
    """
    def __init__(self, runs = None, azi = None, freq = None, load = "", prefix=None):
        """Trolley Field Map Constructor.
        
        Args:
            runs (list[int], optional): run numbers. Specified trolley run is loaded and azi and freq are used.
            azi[neventsx17]  (array, optional): array with raw azimuth. If specified also freq needs to be provided.
            freq[neventsx17]  (array, optional): array with raw frequencies. If specified also azi needs to be provided.
            load (bool): 'simple' load multipoles with uniform weight
                         'cov' load multipoles with resolution and correlations from data
        """
        self.spk_th = gm2.PPM_HZ/2. 
        
        if azi is None:
            if not(runs is None):
                self.tr = gm2.Trolley(runs, False, prefix=prefix)
                _, self.raw_azi, self.raw_freq = self.tr.getBasics()
        else:
            if freq is None:
                raise Exception("Azi und freq need to be provided.")
            else:
                if azi.shape != freq.shape:
                    raise Exception("azi %s und freq %s need to have the same dimensions." % (str(azi.shape), str(freq.shape)))
                self.raw_azi  = azi
                self.raw_freq = freq
        self.s = gm2.np.full(self.raw_azi.shape[0], True)
        
        self.spk = None
        self.spk_s = None
        self.fr = None
        self.freqAt = None
        if load == 'simple':
            self.loadSpikes()
            self.multipoles(cov = gm2.np.diag(gm2.np.array([17.0]*17)**2))
        if load == 'cov':
            self.loadSpikes()
            cov = self.dataCov(nsigma=3.0)
            self.multipoles(cov=cov)

    def loadSpikes(self, th = gm2.PPM_HZ, skip = None):
        """Loads Trolley Spikes.

        Loads the trolley spikes for all channels as well as probe wise and a global selection for good events.

        Args:
            th (float): spike threshold in Hz. Defaults to 1ppm.  
            skip (int): numnber of events which should be exluded at a run start.

        Returns:
            array(bool): selection of cycles with only good probes.
        """
        self.spk = [gm2.Spikes(self.raw_azi[:,probe], self.raw_freq[:,probe], self.spk_th) for probe in range(17)]
        self.spk_s = [gm2.np.abs(self.spk[probe].outl) < th for probe in range(17)]
        sall = self.spk_s[0]
        for p in range(17):
            if not(skip is None):
                if skip > 0:
                    self.spk_s[p][:skip] = False
            sall = sall&self.spk_s[p]
        self.s = sall
        return sall

    def loadFourier(self, N = 500, probes = None):
        """ Loads probe wise fourier fit in phi.

        Args: 
            N (int): number of coefficents, default 500.
            probes (list(int), optional): list with probes which should be loaded. If not specified all 17 probes are loaded. 
        """
        if probes is None:
            self.fr = [gm2.Fourier(self.raw_azi[self.s, probe]/180.*gm2.np.pi, self.raw_freq[self.s, probe], N) for probe in range(17)]
        else:
            self.fr = [None]*17
            for probe in probes:
                self.fr[probe] = gm2.Fourier(self.raw_azi[self.s, probe]/180.*gm2.np.pi, self.raw_freq[self.s, probe], N)

    def loadFreqAt(self, s = None, calib = None):
        """Loads the interpolation of the frequencies in selection.
        
        Args:
            s[nevents] (array(bool), optional): selection, if None the self.s the overall selection is used.
            calib[17] (array(float), optional): if provided it is used as calibration. 
                Otherwise standard calibration is used.
        """
        if calib is None:
            calib = self.tr.getCalibration()[0]
        s_ = [self.s&(self.raw_azi[:,probe] != 0)&(self.raw_freq[:, probe] > 0) for probe in range(17)]
        self.freqAt = [gm2.util.interp1d(self.raw_azi[s_[probe], probe], self.raw_freq[s_[probe], probe] + calib[probe], fill_value='extrapolate') for probe in range(17)]

    def multipoles(self, n_ = 6, cov = None, s = None):
        """Calculates the multipoles with the provided covariance matrix up to order n.
        
        Args:
            n (int): multipole expansion order. Defaults to 6.
            cov[17x17] (optional): covariance matrix
            s[nevents] (bool, optional): special selection of events.

        Returns:
            m[nevents x n]: multipoles in Hz.
            res[nevents x 17]: residuals in Hz.
            chi2[nevent]: chi2s in Hz.
        """
        n = n_*2+1
        if cov is None:
            cov = gm2.np.diag([1.0]*17)
        if self.freqAt is None:
            self.loadFreqAt()
        if s is None:
            s = self.s
        self.azi = self.raw_azi[s,8] 
        w = gm2.np.array([self.freqAt[probe](self.azi) for probe in range(17)]).T
        nev = w.shape[0]
        trb = trBase()
        e   = cov
        M   = (gm2.np.linalg.pinv((trb[:,:n].T.dot(gm2.np.linalg.pinv(e)).dot( trb[:, :n])))).dot( trb[:,:n].T).dot(  gm2.np.linalg.pinv(e))
        peff = gm2.np.trace(trb[:, :n].dot(M))
        self.m    = gm2.np.zeros([nev, n],  dtype=float)
        self.res  = gm2.np.zeros([nev, 17], dtype=float)
        self.chi2 = gm2.np.zeros([nev],     dtype=float)
        for ev in range(nev):
            self.m[ev,:]   = M.dot(w[ev,:])
            self.res[ev,:] = (w[ev,:] - trb[:, :n].dot(self.m[ev,  :]))
            self.chi2[ev]  = self.res[ev,:].T.dot(gm2.np.linalg.pinv(e)).dot(self.res[ev,:])

        self.mAt = [gm2.util.interp1d(self.azi, self.m[:,i], fill_value='extrapolate') for i in range(self.m.shape[1])]
        return self.m, self.res, self.chi2

    def dataResiduals(self, nsigma = None, N = None, th = None, skip = 7, plots = []):
        """Estimate the Residuals of Measuerements to the Fourier Fit. Optioanly truncated to data within nsigma.
        
        Args:
            nsigma (float, optional): if set only residuals within nsigma are returned. defaults None.
            N (int, optional): number of fourier coefficents. See loadFourier(), defaults to None.
            th (float, optional): spike threshold in Hz. See loadSpikes(), defaults to None. 
            plots (list(int)): list of probe numbers which should be ploted. Default [].

        Requires the load the fourier expansion which is done if it is not yet present.
        """
        if ((self.fr is None)|(not(N is None))|(not(th is None))):
            if th is None:
                self.loadSpikes(skip = skip)
            else:
                self.loadSpikes(th, skip = skip)
            if N is None: 
                self.loadFourier()
            else:
                self.loadFourier(N)

        data = [self.fr[probe].B(self.raw_azi[self.s, probe]/180*gm2.np.pi) - self.raw_freq[self.s, probe] for probe in range(17)]
        if nsigma is None:
            return data

        popt = []
        ss = gm2.np.full(data[0].shape, True)
        for probe in range(17):
            popt_, _ = gm2.plotutil.histWithGauss(gm2.plt.gca(), data[probe], bins=gm2.np.arange(-100, 100, 2), orientation='vertical', nsigma=nsigma, alpha=1.0)
            popt.append(popt_)
            if probe in plots:
                gm2.plt.xlabel("residuals [Hz]")
                gm2.plt.plot([popt_[1] - nsigma * gm2.np.abs(popt_[2]), popt_[1] - nsigma * gm2.np.abs(popt_[2])], [0, 300], '--', color=gm2.sns.color_palette()[2])
                gm2.plt.plot([popt_[1] + nsigma * gm2.np.abs(popt_[2]), popt_[1] + nsigma * gm2.np.abs(popt_[2])], [0, 300], '--', color=gm2.sns.color_palette()[2])
                gm2.despine()
                gm2.plt.show()
            gm2.plt.clf()
            s_ = (data[probe] > popt_[1] - nsigma * gm2.np.abs(popt_[2]) )&( data[probe] < popt_[1] + nsigma * gm2.np.abs(popt_[2]) )
            ss = ss&s_

        #cov_phi  = self.azi[self.s, 8][ss]
        data     = gm2.np.array(data)
        return data[:,ss]

    def plotFourier(self, probes):
        """Plots the fourier fit of probes (start at 0).

        Args:
            probes (array(int)): probe numbers to plot.

        Returns:
            f (figure): figure with plot.
        """
        if not(self.fr is None):
            f = gm2.plt.figure()
            for probe in probes:
                if not(self.fr[probe] is None):
                    gm2.plt.plot(self.raw_azi[self.s, probe], self.raw_freq[self.s, probe],'.',label="data #%i" % (probe+1))
                    gm2.plt.plot(gm2.np.sort(self.raw_azi[self.s, probe]), self.fr[probe].B(gm2.np.sort(self.raw_azi[self.s, probe])/180*gm2.np.pi), label="fit #%i" % (probe+1))
            gm2.plt.xlabel("azi [deg]")
            gm2.plt.ylabel("frequency [Hz]")
            gm2.despine()
            return f
        else:
            print("Please first loadFourier.")

    def dataCov(self, nsigma = None, N = None, th = None, skip = 7):
        """Data covariance matrix.
        
        Args:
            nsigma (float, optional): if set only residuals within nsigma are returned. defaults None.
            N (int, optional): number of fourier coefficents. See loadFourier(), defaults to None.
            th (float, optional): spike threshold in Hz. See loadSpikes(), defaults to None. 

        Requires the load the fourier expansion which is done if it is not yet present.
        """
        return gm2.np.cov(self.dataResiduals(nsigma=nsigma, N=N, th=th, skip=skip))

    def plotChi2(self):
        """Plot chi2 as a function of azimuth."""
        plotChi2(self.azi, self.chi2)


def printMultipoleResolution(cov, modes = ['cor', 'dia', 'uni']):
    """Print latex table of multipole resolutions of the specified modes.
    
    Args:
        cov (array(17x17)): data covariance matrix.
        modes (list['cor', 'dia', 'uni']): specify used modes.
            'cor'  full corelations, 'dia' only different probe resolutions, 'uni' uniform resolution.
    """
    e   = cov
    e_d = gm2.np.diag(gm2.np.diag(cov))
    e_a = gm2.np.diag(gm2.np.ones([17])*(gm2.np.sqrt(gm2.np.diag(cov)).mean()**2))
    trb = trBase()
    for n in range(1,18,2):
        if 'cor' in modes:
            cov_   = (gm2.np.linalg.pinv((trb[:,:n].T.dot(gm2.np.linalg.pinv(e)).dot(trb[:, :n]))))
            print(gm2.np.array_str(gm2.np.sqrt(gm2.np.diag(cov_)), precision=3, max_line_width=10000).replace(". ", "& ").replace("nan","-&").replace(".]","\\\\").replace("[","cor & %i &" % (n//2)))
        if 'dia' in modes:
            cov_d = (gm2.np.linalg.pinv((trb[:,:n].T.dot(gm2.np.linalg.pinv(e_d)).dot(trb[:, :n]))))
            print(gm2.np.array_str(gm2.np.sqrt(gm2.np.diag(cov_d)),precision=3, max_line_width=10000).replace(". ", "& ").replace("nan","-&").replace(".]","\\\\").replace("[","dia & %i &" % (n//2)))
        if 'uni' in modes:
            cov_a = (gm2.np.linalg.pinv((trb[:,:n].T.dot(gm2.np.linalg.pinv(e_a)).dot(trb[:, :n]))))
            print(gm2.np.array_str(gm2.np.sqrt(gm2.np.diag(cov_a)),precision=3, max_line_width=10000).replace(". ", "& ").replace("nan","-&").replace(".]","\\\\").replace("[","uni & %i &" % (n//2)))

def printMultipoleResolution(cov, n):
    """ Plot multipole resolutions and correlations of the specified modes.
    
    Args:
        cov (array(17x17)): data covariance matrix.
        n (int): expansion, needs to be in range(1,18,2).
    """
    trb = trBase()
    e   = cov
    e_d = gm2.np.diag(gm2.np.diag(cov))
    e_a = gm2.np.diag(gm2.np.ones([17])*(gm2.np.sqrt(gm2.np.diag(cov)).mean()**2))
    cov   = (gm2.np.linalg.pinv((trb[:,:n].T.dot(gm2.np.linalg.pinv(e)).dot(trb[:, :n]))))
    cov_d = (gm2.np.linalg.pinv((trb[:,:n].T.dot(gm2.np.linalg.pinv(e_d)).dot(trb[:, :n]))))
    cov_a = (gm2.np.linalg.pinv((trb[:,:n].T.dot(gm2.np.linalg.pinv(e_a)).dot(trb[:, :n]))))

    ''' plot the fit error and correlations '''
    allTypes = True
    fs = gm2.plotutil.figsize()
    cs = gm2.sns.color_palette('Spectral_r',40)
    f = gm2.plt.figure(figsize=[fs[0]*1.5, fs[1]*1.5])
    ax = []
    cov_ = cov
    cor_ = cov_ / gm2.np.sqrt(gm2.np.diag(cov_)[:,None] * gm2.np.diag(cov_))
    t = gm2.np.arange(0, 2 * gm2.np.pi, 0.01)
    for i in range(n):
        for j in range(n):
            ax.append(gm2.plt.subplot2grid((17, 17), (i, j)))
            ax[-1].xaxis.set_ticklabels([])
            ax[-1].yaxis.set_ticklabels([])
            ax[-1].xaxis.set_ticks([])     
            ax[-1].yaxis.set_ticks([])
            if i == j:
                 ax[-1].text(0.5,0.5,"%.2f" % gm2.np.sqrt(cov_[i,j]), horizontalalignment='center', verticalalignment='center', fontsize=10)
                 if i in [0]:
                     ax[-1].set_title(r"$B_{0}$")
                     ax[-1].set_ylabel(r"$B_{0}$")
            if j > i:
                w, v = gm2.np.linalg.eig(cov_[[i,j]][:,[i,j]])
                xx = []
                for y in (gm2.np.array([gm2.np.cos(t), gm2.np.sin(t)]).T * gm2.np.sqrt(w)):
                     xx.append(v.dot(y))
                xx = gm2.np.array(xx)
                ax[-1].plot(xx[:,1], xx[:,0])
                if allTypes:
                    w_d, v_d = gm2.np.linalg.eig(cov_d[[i,j]][:,[i,j]])
                    xx_d = []
                    w_a, v_a = gm2.np.linalg.eig(cov_a[[i,j]][:,[i,j]])
                    xx_a = []
                    for y in (gm2.np.array([gm2.np.cos(t), gm2.np.sin(t)]).T * gm2.np.sqrt(w_d)):
                        xx_d.append(v_d.dot(y))
                    for y in (gm2.np.array([gm2.np.cos(t), gm2.np.sin(t)]).T * gm2.np.sqrt(w_a)):
                        xx_a.append(v_a.dot(y))
                    xx_d = gm2.np.array(xx_d)
                    xx_a = gm2.np.array(xx_a)
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
                if ~gm2.np.isnan(cor_[i,j]):
                    ax[-1].set_facecolor(cs[int(gm2.np.floor(cor_[i,j]*20))+20 ])
                #else:
                #    ax[-1].set_facecolor(cs[int(np.floor(cor_[i,j]*20))+20 ])
                if j in [0]:
                    if i%2 == 0:
                        ax[-1].set_ylabel(r"$b_{%i}$" % ((i+1)//2) )
                    else:
                        ax[-1].set_ylabel(r"$a_{%i}$" % ((i+1)//2) )

    gm2.plt.subplots_adjust(wspace=0, hspace=0)
    #plt.savefig("plots/mp_coef_cor_%i.png" % (i//2))
    #plt.savefig("plots/mp_coef_cor_%i.pdf" % (i//2))
    return f


def trBase(at=45.0):
    """Get Trolley probe value in multipole base.
    
    Args:
        at (float): multipole evaluation distance in mm. Defaults to 45mm."""
    pos_fp = (gm2.FP.probes.position.r, gm2.FP.probes.position.theta)
    pos_tr = (gm2.TR.probes.position.r, gm2.TR.probes.position.theta)

    trb  = gm2.np.array([gm2.util.multipole(pos_tr, 1,     0,    0),
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
    return trb 

def plotFieldMapUncertainty(cov, N):
    """Plot field map uncertenty for expansion up to N.
    
    Args:
        cov (array(17x17)): data covariance matrix.
        N (int): multipole expansion up to N. Should be in range(1,18,2).
    """
    trb = trBase()
    e   = cov
    cov   = (gm2.np.linalg.pinv((trb[:,:N].T.dot(gm2.np.linalg.pinv(e)).dot(trb[:, :N]))))

    mm = 50.
    xx, yy = gm2.np.meshgrid(gm2.np.arange(-mm, mm), gm2.np.arange(-mm, mm))
    rr  = gm2.np.sqrt(xx**2 + yy**2)
    phi = gm2.np.arctan2(yy, xx)
    s2 = gm2.np.zeros_like(xx)
    for i in range(rr.shape[0]):
        for j in range(rr[0].shape[0]):
            s2[i,j]   = f(rr[i,j], phi[i,j], N//2).dot(cov).dot(f(rr[i,j], phi[i,j], N//2).T)

    step = 50 # ppb
    names = ["cor","dia","uni"]
    for ii, s2_ in enumerate([s2]):
        ff = gm2.plt.figure(figsize=(7, 6))
        from matplotlib.colors import ListedColormap
        s2_[rr>mm] = gm2.np.nan
        nmax = 1000
        if nmax is None:
            nmax = ((gm2.np.ceil(gm2.np.nanmax(gm2.np.sqrt(s2_)))*gm2.HZ2PPB)//step+1)*step
            nmax = gm2.np.min([nmax, 1000])
        nmin = 0
        if nmin is None:
            nmin = ((gm2.np.floor(gm2.np.nanmin(gm2.np.sqrt(s2_)))*gm2.HZ2PPB)//step)*step
        nc = int((nmax-nmin)//step)
        cmap = ListedColormap(gm2.sns.color_palette("Spectral_r", nc))
        #cmap = ListedColormap(gm2.sns.color_palette("Blues", nc))
        levels_ = gm2.np.arange(nmin, nmax+(nmax-nmin)/nc, step)
        cs = gm2.plt.contour(xx, yy, gm2.np.sqrt(s2_)*gm2.HZ2PPB,  levels=levels_, colors='k', linewidths=1.0, inline=True, alpha=0.5)
        gm2.plt.clabel(cs, inline=1, fontsize=10)
        cf = gm2.plt.contourf(xx, yy, gm2.np.sqrt(s2_)*gm2.HZ2PPB, levels=levels_, cmap=cmap, extend='max')
        cb = gm2.plt.colorbar()
        cb.ax.set_title("[ppb]")
        gm2.plt.plot(gm2.TR.probes.position.x, gm2.TR.probes.position.y, 'x', color='black', alpha=0.3)
        t = gm2.np.arange(0, gm2.np.pi*2,0.01)
        r_ = 45.
        gm2.plt.plot(r_*gm2.np.sin(t), r_*gm2.np.cos(t),color='black', alpha=0.3, linewidth=2)

        gm2.plt.xlim([-mm, mm])
        gm2.plt.ylim([-mm, mm])
        gm2.plt.xlabel("x [mm]")
        gm2.plt.ylabel("y [mm]")
        gm2.plt.title(r"$n_{\rm{max}}=%i$, %s" % (N//2, names[ii]))
        gm2.plt.tight_layout()
    return ff

def f(r, phi, n, r0=45.):
    """Calculate multipole term n at positions r(array), phi(array).
    
    Args:
        r (floar ot array(float)): radius of point.
        phi (floar ot array(float)): phi of point, same dimension as r.
        n (array(float)): number of multipole term. 0:Dipole, 1: skew qud, ...
        """
    if type(r) is not gm2.np.ndarray:
        r   = gm2.np.array([r])
        phi = gm2.np.array([phi])
    if r.shape[0] != phi.shape[0]:
        raise Exception('Dimension of r and phi have to be the samne.')
    ii = gm2.np.arange(1,n+1)
    return gm2.np.concatenate([gm2.np.ones([r.shape[0], 1]),
                           gm2.np.insert((((r[:,None]/r0)**ii) * gm2.np.sin(ii * phi[:,None])) ,gm2.np.arange(n),
                                     (((r[:,None]/r0)**ii) * gm2.np.cos(ii * phi[:,None])) , axis=1)], axis=1)

def plotDataCov(residuals):
    cor = gm2.np.corrcoef(residuals)
    """Helper Function to plot data covariance and correlations
    
    Args:
        residuals (array(17x..)): residual data.
    """
    fs = gm2.plotutil.figsize()
    f = gm2.plt.figure(figsize=[fs[0]*1.5, fs[1]*1.5])
    ax = []
    ax.append(gm2.plt.subplot2grid((17, 17), (0, 0)))
    ax[-1].xaxis.set_ticklabels([])
    ax[-1].yaxis.set_ticklabels([])
    ax[-1].xaxis.set_ticks([])
    ax[-1].yaxis.set_ticks([])
    gm2.plt.title("1")
    ax[-1].set_ylabel("1")

    ax.append(gm2.plt.subplot2grid((17, 17), (16, 16)))
    ax[-1].xaxis.set_ticklabels([])
    ax[-1].yaxis.set_ticklabels([])
    ax[-1].xaxis.set_ticks([])
    ax[-1].yaxis.set_ticks([])

    cs = gm2.sns.color_palette('Blues',10)
    cs = gm2.sns.color_palette('Spectral_r',40)
    for p1 in range(0,17):
        for p2 in range(p1+1,17):
           ax.append(gm2.plt.subplot2grid((17, 17), (p1, p2)))
           ax[-1].plot(residuals[p1,:], residuals[p2,:], '.', markersize=2, alpha=0.1)
           ax[-1].set_xlim([-70,70])
           ax[-1].set_ylim([-70,70])
           ax[-1].xaxis.set_ticklabels([])
           ax[-1].yaxis.set_ticklabels([])
           ax[-1].xaxis.set_ticks([])
           ax[-1].yaxis.set_ticks([])
           if p1 in [0]:
               gm2.plt.title("%i" % (p2+1))
        for p2 in range(0, p1):
           ax.append(gm2.plt.subplot2grid((17, 17), (p1, p2)))
           if p2 in [0]:
              ax[-1].set_ylabel("%i" % (p1+1))
           ax[-1].xaxis.set_ticklabels([])
           ax[-1].yaxis.set_ticklabels([])
           ax[-1].xaxis.set_ticks([])
           ax[-1].yaxis.set_ticks([])
           ax[-1].text(0.5,0.5,"%.2f" % cor[p1,p2], horizontalalignment='center', verticalalignment='center', fontsize=8)
           ax[-1].set_facecolor(cs[int(gm2.np.floor(cor[p1,p2]*40)) ])

    gm2.plt.subplots_adjust(wspace=0, hspace=0)
    return f

from scipy import stats
def plotChi2Dist(chi2, peff = None, cov = None, label = ""):
    """Plot chi2 distribution.
    
    Args:
        chi2 (array): hi2s array.
        peff (float, optional): degree of freedoms of M matrix.
        cov[17x17] (optional): covariance matrix to determine d.o.f.
        label (str, optional): plot lable.
    """
    if peff is None:
        if not(cov is None):
            e = cov
            trb = trBase()
            M   = (gm2.np.linalg.pinv((trb[:,:n].T.dot(gm2.np.linalg.pinv(e)).dot( trb[:, :n])))).dot(trb[:,:n].T).dot(gm2.np.linalg.pinv(e))
            peff = gm2.np.trace(trb[:, :n].dot(M))
    nlim = 200
    lim = chi2.mean()
    v  , _, _ = gm2.plt.hist(chi2, bins=gm2.np.arange(0,lim, lim/nlim), histtype='step', label=label)
    tt = gm2.np.arange(0, lim, lim/nlim/2.)
    if not(peff is None):
        chi2_ = stats.chi2.pdf(tt,17.-peff)
        gm2.plt.plot(tt, chi2_/chi2_.max()*v.max()*0.8, '-', label=r"$\chi^2$(d.o.f.= %.1f)" % (17.-peff))
    gm2.plt.xlabel(r"$\chi^{2}$")
    #plt.title(r"$n=%i$" % (n))
    gm2.plt.legend()
    gm2.despine()
    gm2.plt.show()

def plotChi2(azi, chi2):
    """Plot chi2 as a function of azimuth.
    
    Args:
        azi[nevents] (array): azimuthal position.
        chi2[nevents] (array): chi2s.

    Returns:
        fig (figure): figure.
    """
    fs = gm2.plotutil.figsize()
    fig = gm2.plt.figure(figsize=[fs[0]*1.5, fs[1]])
    gm2.plt.plot(azi, chi2,  '.', markersize=2, alpha=1,label="cor")
    gm2.plt.xlabel("azimuth [rad]")
    gm2.plt.ylabel("$\chi^2$")
    gm2.despine()
    return fig

def study(self):
    fp = gm2.FixedProbe(self.runs,load='simple')
    s_phi = fp.getStationPhi()
    s_phi = np.sort(s_phi)
    az = np.arange(0,360,0.001)
    d_s_boxcar = []
    d_s_linear = []
    for n, sphi in enumerate(s_phi):
        if n in [0]: 
           lower = s_phi[-1]  
        else:
           lower = s_phi[n-1]
        if n in [s_phi.shape[0]-1]:
           upper = s_phi[0]
        else:
           upper = s_phi[n+1]
        s = (az>lower)&(az<upper)
        if lower > upper:
            s = (az>lower)|(az<upper)
       
        az_    = ((az[s] - sphi+180.)%360)-180.
        lower_ = ((lower - sphi+180.)%360)-180.
        upper_ = ((upper - sphi+180.)%360)-180.
        
        w_boxcare = (az_>lower_/2.)&(az_<upper_/2.)
        print(az_.min(), az_.max(), lower_, upper_)

        w_linear = np.ones_like(az_)
        w_linear[az_<0] = w_linear[az_<0] - az_[az_<0]/az_.min() 
        w_linear[az_>0] = w_linear[az_>0] - az_[az_>0]/az_.max()
        #gm2.plt.plot(az[s], w_boxcare)
        #gm2.plt.plot(az[s], w_linear)
        #gm2.plt.show()
        
        if n == 1:
            gm2.plt.plot(az[s], w_boxcare, '.', markersize=2)
            gm2.plt.plot(az[s], w_linear, '.', markersize=2)
            gm2.plt.show()



        w_boxcar =((az[s]>(sphi+lower)/2.)&(az[s]<(upper+sphi)/2.))
        if lower > upper:
            w_boxcar =((az[s]>(sphi+lower)/2.)|(az[s]<(upper+sphi)/2.))

        if n < 2:
          gm2.plt.plot(az[s], w_boxcar,'.', markersize=2)
    gm2.plt.show()

    #      print "%03.1f %03.1f %02.0f %.1f " % (lower, upper, (upper-lower), az[s].shape[0]/1000.)


