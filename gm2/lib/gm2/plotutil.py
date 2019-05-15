import gm2

def figsize():
    return gm2.plt.rcParams['figure.figsize']

def plot_ts(*args, **kwargs):
    from matplotlib.dates import DateFormatter
    if 'dformat' in kwargs:
        dformat = kwargs['dformat']
    else:
        dformat = '%m/%d\n%H:%M'
    formatter = DateFormatter(dformat)
    ax = gm2.plt.plot_date(gm2.util.ts2datetime(args[0]), *args[1:], **kwargs)
    gm2.plt.gca().xaxis.set_major_formatter(formatter)
    return ax

def histWithGauss(ax, data, bins, nsigma=3, orientation='horizontal', alpha=0.5, RMS=False):
    s = gm2.np.isnan(data) == False
    v_, b_, _ = ax.hist(data[s], bins=bins, histtype='stepfilled', orientation=orientation)  
    bc = gm2.util.getBinCenter(b_)
    popt, pcov = gm2.util.fit_gauss(v_, bc,p0=[gm2.np.max(data[s]), data[s].mean(),  data[s].std()])
    popt[2] = gm2.np.abs(popt[2])
    s_= (bc> popt[1] - popt[2]*nsigma)&(bc< popt[1] + popt[2]*nsigma)
    if gm2.np.argwhere(s_).shape[0] < 3:
        print("fit failed")
        return popt, pcov
        s_= (bc> popt[1] - popt[2]*nsigma)&(bc< popt[1] + popt[2]*nsigma)
        s_= (bc> popt[1] - popt[2]*nsigma*3)&(bc< popt[1] + popt[2]*nsigma*3)
    popt, _ = gm2.util.fit_gauss(v_[s_], bc[s_], p0=popt)
    popt[2] = gm2.np.abs(popt[2])
    s_= (bc> popt[1] - popt[2]*nsigma)&(bc< popt[1] + popt[2]*nsigma)
    popt, pcov = gm2.util.fit_gauss(v_[s_], bc[s_], p0=popt)
    popt[2] = gm2.np.abs(popt[2])
    if orientation is 'horizontal':
        ax.plot(gm2.util.gauss(bc[s_], *popt), bc[s_], '-', alpha=alpha)
    else:
        ax.plot(bc[s_], gm2.util.gauss(bc[s_], *popt), '-', alpha=alpha)

    if RMS:
        s = gm2.np.isnan(data) == False
        rms = data[s].std()
        if rms/gm2.PPM_HZ < 1:
            ax.text(0.75, 0.85,("RMS %.0f" % (rms*gm2.HZ2PPB))+" ppb",\
                                        horizontalalignment='center',\
                                        verticalalignment='center',\
                                        transform = ax.transAxes)
        else:
            ax.text(0.75, 0.85,("RMS %.2f" % (rms*gm2.HZ2PPM))+" ppm",\
                                        horizontalalignment='center',\
                                        verticalalignment='center',\
                                        transform = ax.transAxes)
        if popt[2]/gm2.PPM_HZ < 1:
            ax.text(0.75, 0.75,(r'$\sigma$ '+"%.0f" % (popt[2]*gm2.HZ2PPB))+"ppb",\
                                        horizontalalignment='center',\
                                        verticalalignment='center',\
                                        transform = ax.transAxes)
        else:
            ax.text(0.75, 0.75,(r'$\sigma '+"%.2f" % (popt[2]*gm2.HZ2PPM))+"ppm",\
                                        horizontalalignment='center',\
                                        verticalalignment='center',\
                                        transform = ax.transAxes)
    else:
        if popt[2]/gm2.PPM_HZ < 1:
            ax.text(0.75, 0.75,("%.0f" % (popt[2]*gm2.HZ2PPB))+"\nppb",\
                                        horizontalalignment='center',\
                                        verticalalignment='center',\
                                        transform = ax.transAxes)
        else:
            ax.text(0.75, 0.75,("%.2f" % (popt[2]*gm2.HZ2PPM))+"\nppm",\
                                        horizontalalignment='center',\
                                        verticalalignment='center',\
                                        transform = ax.transAxes)
    return popt, pcov

'''
def trlyDist(ax, probe, data, bins, nsigma):
    s = gm2.np.isnan(data[probe]) == False
    v_, b_, _ = ax.hist(data[probe][s], bins=bins, histtype='stepfilled')
    bc = gm2.util.getBinCenter(b_)
    popt, _ = gm2.util.fit_gauss(v_, bc)
    popt[2] = gm2.np.abs(popt[2])
    s_= (bc> popt[1] - popt[2]*nsigma)&(bc< popt[1] + popt[2]*nsigma)
    popt, _ = gm2.util.fit_gauss(v_[s_], bc[s_], p0=popt)
    popt[2] = gm2.np.abs(popt[2])
    ax.semilogy()
    ax.plot(bc[s_], gm2.util.gauss(bc[s_], *popt),'-', alpha=0.5)
    ax.text(0.75, 0.75,("%.1f" % (popt[2]/61.78))+"\nppm",\
                                    horizontalalignment='center',\
                                    verticalalignment='center',\
                                    transform = ax.transAxes)
'''


from gm2.constants import TR, FP
def trlyPlot(usrPlt, hr=60, xlabel="xlabel", title="title", *args):
        """Genrates a canvas with 17 subplots one at each trolley probe position.
        
        Args:
            usrPlt (function): the function which is called for each subplot
                the format is usrPlt(ax, probe_no, \*args), typically \*args contains the plotting data.
            \*args (var): arguments which are forwarded to usrPlt, typically the plotting data.
            hr (float, optional): dexlim of pplots. Default 60.0.
            xlabel (string, optional): x-label title. Default ``xlabel``.
            title (string, optional):  canvas title. Default ``title``.
            
        Returns:
            figure: the canvas including the 17 subplots.
            
        Examples:
            Simple example to plot ``data`` with the format data[plot_data,probe_no]

            >>> import gm2
            >>> from gm2 import plt, np
            >>> def usrPlt(ax, probe, data):
            >>>     ax.plot(data[:,probe])
            >>> data = (np.arange(0,10,0.1)[:,None] ** np.arange(17))
            >>> gm2.plotutil.trlyPlot(usrPlt, data, hr=110)
            >>> plt.show() 

            Let's assume ``data`` stores the residuals for each trolley probe to a model in the format data[residuals, probe_no]:
            
            >>> xlim = 100
            >>> binw = 5
            >>> def trlyHistWithGauss(ax, probe, data):
            >>>    gm2.plotutil.histWithGauss(ax, data[:,probe], bins=np.arange(-xlim, xlim, binw), orientation='vertical')
            >>>
            >>> f = gm2.plotutil.trlyPlot(trlyHistWithGauss, data, hr=xlim, xlabel="residuals [Hz]", title="title")
            >>> f.show()
        """
        #from matplotlib.backends.backend_pdf import PdfPages
        #with PdfPages(fname) as pdf:
        ax = []
        figsize = [gm2.plt.rcParams['figure.figsize'][1] * 2.0, gm2.plt.rcParams['figure.figsize'][1] * 2.0]
        f = gm2.plt.figure(figsize=figsize)
        gm2.plt.subplots_adjust(wspace=0.0, hspace=0.0)
        for probe in range(17):
            ax.append(gm2.plt.subplot2grid((5,5), TR.probes.grid[probe]))
            usrPlt(ax[probe], probe, *args);

            ax[probe].text(0.15, 0.9,"#"+str(probe+1),\
                        horizontalalignment='center',\
                        verticalalignment='center',\
                        transform = ax[probe].transAxes)

            ax[probe].yaxis.set_ticklabels([])
            if TR.probes.grid[probe][0] != 4:
                ax[probe].xaxis.set_ticklabels([])
            if hr:
                ax[probe].set_xlim([-hr,hr])
            if TR.probes.grid[probe] == (4,2):
                ax[probe].set_xlabel(xlabel)

            if TR.probes.grid[probe] == (0,2):
                gm2.plt.title(title)
        return f


from scipy.interpolate import griddata
def plotTrFieldMap(probes, method='cubic', at=45.0, nmax=None, lines=True):
    stepSize = 1
    print(len(probes), TR.probes.n)
    if len(probes) == TR.probes.n:
        xx = gm2.np.arange(TR.probes.position.x.min(), TR.probes.position.x.max() + stepSize, stepSize)
        yy = gm2.np.arange(TR.probes.position.y.min(), TR.probes.position.y.max() + stepSize, stepSize)
        zz = griddata((TR.probes.position.x, TR.probes.position.y), probes, (xx[None,:], yy[:,None]), method=method)
    else:
        for i in range(1, len(probes)):
            probes[i] = probes[i] / at**((i+1)//2)
        #probes = probes / np.array([at**0, at**1, at**1, at**2, at**2, at**3, at**3, at**4, at**4])
        xx = gm2.np.arange(-45, 45 + stepSize, stepSize)
        yy = gm2.np.arange(-45, 45 + stepSize, stepSize)
        zz = gm2.np.zeros([xx.shape[0], yy.shape[0]])
        for i in range(xx.shape[0]):
            for j in range(yy.shape[0]):
                #print i, j
                r_     = gm2.np.sqrt(yy[j]**2 + xx[i]**2)
                theta_ = gm2.np.arctan2(yy[j],  xx[i]) 
                #print(probes)
                zz[i,j] = gm2.util.multipole((gm2.np.array([r_]), gm2.np.array([theta_])), *probes)
    zz = (zz-gm2.np.nanmean(zz))/gm2.PPM_HZ
    f = gm2.plt.figure(figsize=(10, 8))
    from matplotlib.colors import ListedColormap
    #cmap = ListedColormap(gm2.sns.color_palette("nipy_spectral",32))
    if nmax is None:
        nmax = gm2.np.ceil(gm2.np.max([abs(gm2.np.nanmin(zz)), gm2.np.nanmax(zz)]))
    cmap = ListedColormap(gm2.sns.color_palette("Spectral_r",32))
    #gm2.plt.contourf(xx, yy, zz, levels=gm2.np.arange(gm2.np.nanmin(zz), gm2.np.nanmax(zz)+gm2.PPM_HZ, gm2.PPM_HZ), cmap=cmap)
    #print(gm2.np.nanmin(zz), gm2.np.nanmax(zz))
    levels_ = gm2.np.arange(-nmax, nmax+0.1, 0.1)
    if lines:
        cs = gm2.plt.contour(xx, yy, zz, levels=levels_, colors='k', linewidths=1.0, inline=True, alpha=0.5)
        gm2.plt.gca().clabel(cs, inline=1, fontsize=10)
        cf = gm2.plt.contourf(xx, yy, zz, levels=levels_, cmap=cmap, extend='both')
    else:
        cmap = ListedColormap(gm2.sns.color_palette("Spectral_r",128))
        levels_ = gm2.np.arange(-nmax, nmax+2.0*nmax/128, 2.0*nmax/128)
        cf = gm2.plt.contourf(xx, yy, zz.T, levels=levels_, cmap=cmap, extend='both')
    #gm2.plt.contourf(xx, yy, zz)
    #gm2.plt.clabel("[ppm]")
    
    gm2.plt.plot(TR.probes.position.x, TR.probes.position.y, 'x', color='black')
    cb = gm2.plt.colorbar()
    cb.ax.set_title("[ppm]")

    gm2.plt.xlim([-45, 45])
    gm2.plt.ylim([-45, 45])
    gm2.plt.xlabel("x [mm]")
    gm2.plt.ylabel("y [mm]")
    return f

def Multipole(m, method='cubic', at=45.0, nmax=None): 
    stepSize = 1
    for i in range(1, len(m)):
        m[i] = m[i] / at**((i+1)//2)
    #probes = probes / np.array([at**0, at**1, at**1, at**2, at**2, at**3, at**3, at**4, at**4])
    xx = gm2.np.arange(-45, 45 + stepSize, stepSize)
    yy = gm2.np.arange(-80, 80 + stepSize, stepSize)
    zz = gm2.np.zeros([xx.shape[0], yy.shape[0]])
    for i in range(xx.shape[0]):
        for j in range(yy.shape[0]):
            #print i, j
            r_     = gm2.np.sqrt(yy[j]**2 + xx[i]**2)
            theta_ = gm2.np.arctan2(yy[j],  xx[i]) 
            #print(probes)
            zz[i,j] = gm2.util.multipole((gm2.np.array([r_]), gm2.np.array([theta_])), *m)
    zz = (zz-gm2.np.nanmean(zz))
    f = gm2.plt.figure(figsize=(6, 8))
    from matplotlib.colors import ListedColormap
    #cmap = ListedColormap(gm2.sns.color_palette("nipy_spectral",32))
    if nmax is None:
        nmax = gm2.np.ceil(gm2.np.max([abs(gm2.np.nanmin(zz)), gm2.np.nanmax(zz)]))
    #cmap = ListedColormap(gm2.sns.color_palette("Spectral_r",32))
    #gm2.plt.contourf(xx, yy, zz, levels=gm2.np.arange(gm2.np.nanmin(zz), gm2.np.nanmax(zz)+gm2.PPM_HZ, gm2.PPM_HZ), cmap=cmap)
    #print(gm2.np.nanmin(zz), gm2.np.nanmax(zz))
    cmap = ListedColormap(gm2.sns.color_palette("Spectral_r",128))
    levels_ = gm2.np.arange(-nmax, nmax+2.0*nmax/128, 2.0*nmax/128)
    cf = gm2.plt.contourf(xx, yy, zz.T, levels=levels_, cmap=cmap, extend='both')
    #gm2.plt.contourf(xx, yy, zz)
    #gm2.plt.clabel("[ppm]")
    
    gm2.plt.plot(TR.probes.position.x, TR.probes.position.y, 'x', color='black', alpha=0.3)
    gm2.plt.plot(FP.probes.position.x, FP.probes.position.y, 'x', color='black', alpha=0.3)
    
    for x, y, r, theta in zip(TR.probes.position.x, TR.probes.position.y, TR.probes.position.r, TR.probes.position.theta ):
        gm2.plt.text(x,y,"%.2f" % gm2.util.multipole((gm2.np.array([r]), gm2.np.array([theta])), *m) , 
            horizontalalignment='center',
            verticalalignment='center')

    for x, y, r, theta in zip(FP.probes.position.x, FP.probes.position.y, FP.probes.position.r, FP.probes.position.theta ):
        gm2.plt.text(x,y,"%.2f" % gm2.util.multipole((gm2.np.array([r]), gm2.np.array([theta])), *m) , 
            horizontalalignment='center',
            verticalalignment='center')

    cb = gm2.plt.colorbar()
    cb.ax.set_title("[ppm]")

    gm2.plt.xlim([-45, 45])
    gm2.plt.ylim([-80, 80])
    gm2.plt.xlabel("x [mm]")
    gm2.plt.ylabel("y [mm]")
    gm2.plt.tight_layout()
    return f
