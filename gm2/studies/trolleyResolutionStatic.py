import Trolley
from util import *

details = np.arange(16)
show    = []#np.arange(16)

runs = {3932:"180.844 deg, Yoke G Azi 3",
        3930:"176.08 deg, Yoke G azi 2",
        3928:"172.85 deg, Yoke G Azi 1",
        3754:"feedback off",
        3425:"PP region 189.241 deg",
        3419:"PP region 189.241 deg",
        3365:"inflector"}

for run in runs:
    #run = 3365
    title = "run "+str(run)+" ("+runs.get(run)+")"
    t = Trolley.Trolley([run])



    freq, phi, time = t.getBasics()
    s = (time[:,0]>0)&(freq[:,0]>40000.0)

    #import matplotlib.pyplot as plt
    from gm2plotsettings import *
    PPB_HZ = PPM_HZ/1000.


    sl = 31
    nsigma = 3.0

    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages('plots/trolleyResolutionStatic_run'+str(run)+'.pdf') as pdf:
        ax = []
        hr = 60
        figsize = [plt.rcParams['figure.figsize'][1] * 2.0, plt.rcParams['figure.figsize'][1] * 2.0]
        f = plt.figure(figsize=figsize)
        plt.subplots_adjust(wspace=0.0, hspace=0.0)
        for probe in range(17):
            ax.append(plt.subplot2grid((5,5), TR.probes.grid[probe]))
            df = smooth(freq[s, probe], sl+2, 'ref') - smooth(freq[s, probe], sl, 'lr')
            v_, b_, _ = ax[probe].hist(df/PPB_HZ, bins=np.arange(-hr, hr, 1.0), alpha=0.8, histtype='stepfilled') 
            ax[probe].yaxis.set_ticklabels([])
            if TR.probes.grid[probe][0] != 4:
                ax[probe].xaxis.set_ticklabels([])
            ax[probe].set_xlim([-hr,hr])
            from scipy.optimize import curve_fit
            bc_ = b_[:-1] + np.diff(b_) / 2.0
            s_  = v_>0
            popt, pcov = curve_fit(gauss, bc_[s_], v_[s_], p0=[v_.sum(), 0, 10.0], sigma=np.sqrt(v_[s_]), absolute_sigma=True)
            popt[2] = np.abs(popt[2])
            s_  = (bc_> popt[1] - popt[2]*nsigma)&(bc_< popt[1] + popt[2]*nsigma)&s_
            popt, pcov = curve_fit(gauss, bc_[s_], v_[s_], p0=popt,                sigma=np.sqrt(v_[s_]), absolute_sigma=True)
            popt[2] = np.abs(popt[2])
            ax[probe].plot(bc_[s_], gauss(bc_[s_], *popt)) 
            ax[probe].text(0.15, 0.9,"#"+str(probe),\
                                    horizontalalignment='center',\
                                    verticalalignment='center',\
                                    transform = ax[probe].transAxes)
            ax[probe].text(0.75, 0.75,("%.1f" % popt[2])+"\nppb",\
                                    horizontalalignment='center',\
                                    verticalalignment='center',\
                                    transform = ax[probe].transAxes)
            if TR.probes.grid[probe] == (4,2):
                ax[probe].set_xlabel("resolution [ppb]")

            if TR.probes.grid[probe] == (0,2):
                plt.title(title)
        pdf.savefig(f)
        if len(show)>0:
            plt.show()


        for probe in range(17):
            s = (time[:,probe]>0)&(freq[:,probe]>40000.0)
            t0 = time[s, probe].min()
            tm = time[s, probe].max()
            df = smooth(freq[s, probe], sl+2, 'ref') - smooth(freq[s, probe], sl, 'lr')

            if probe in details:
                figsize = [plt.rcParams['figure.figsize'][0] * 2.0, plt.rcParams['figure.figsize'][1] * 2.0]
                f = plt.figure(figsize=figsize)
                ax0_0 = plt.subplot2grid((2,2), (0, 0))
                ax0_1 = plt.subplot2grid((2,2), (1, 0))
                ax1_0 = plt.subplot2grid((2,2), (1, 1))
                ax1_1 = plt.subplot2grid((2,2), (1, 1))

                ax0_0.plot((time[s, probe]-t0)/1e9, freq[s, probe],'.', label="raw data", color=sns.color_palette()[0])
                ax0_0.plot((smooth(time[s, probe], sl+2,'ref')-t0)/1e9, smooth(freq[s, probe], sl, 'lr'), '-', label="filter", color=sns.color_palette()[1])
                ax0_0.set_ylim(np.percentile(smooth(freq[s, probe], sl, 'lr'),[0.5,99.5]))
                ax0_1.set_xlabel("time [s]")
                ax0_0.set_ylabel("frequency [Hz]")
                ax0_0.set_title(title+" :: probe "+str(probe))
                ax0_0.legend()
                ax0_0.set_xlim([0,(tm-t0)/1e9])

                #df = smooth(freq[s, probe], 31+2, 'ref') - smooth(freq[s, probe], 31, 'lr')
                ax0_1.plot((smooth(time[s, probe], sl+2,'ref')-t0)/1e9, df,'.') 
                ax0_1.set_ylim(np.percentile(df, [0.5,99.5])*1.2)
                ax0_1.set_ylabel("residual\nfrequency [Hz]")
                ax0_1.set_xlim([0, (tm-t0)/1e9])
                ax0_1.plot([0,(tm-t0)/1e9],[0.0, 0.0], '--', color='grey')

                lim = ax0_1.get_ylim()
                v_, b_, _ = ax1_1.hist(df/PPB_HZ, bins=np.arange(lim[0], lim[1], 1.0*PPB_HZ)/PPB_HZ, alpha=0.8, orientation='horizontal', histtype='stepfilled')
                ax1_1.xaxis.set_ticklabels([])
                ax1_1.set_ylabel("[ppb]", rotation=0)
                ax1_1.yaxis.set_label_coords(0.015, 1.05)

                from scipy.optimize import curve_fit
                bc_ = b_[:-1] + np.diff(b_) / 2.0
                s_  = v_>0
                popt, pcov = curve_fit(gauss, bc_[s_], v_[s_], p0=[v_.sum(), 0, 10.0], sigma=np.sqrt(v_[s_]), absolute_sigma=True)
                popt[2] = np.abs(popt[2])
                s_  = (bc_> popt[1] - popt[2]*nsigma)&(bc_< popt[1] + popt[2]*nsigma)&s_
                popt, pcov = curve_fit(gauss, bc_[s_], v_[s_], p0=popt,                sigma=np.sqrt(v_[s_]), absolute_sigma=True)
                popt[2] = np.abs(popt[2])
                ax1_1.plot(gauss(bc_[s_], *popt), bc_[s_])
                ax1_1.text(0.7, 0.8,("mean: % 4.1f" % popt[1])+" ppb\n"+\
                                    ("std:   % 4.1f" % popt[2])+" ppb",\
                                    horizontalalignment='center',\
                                    verticalalignment='center',\
                                    transform = ax1_1.transAxes)

                #polish()
                sns.despine()
                pdf.savefig(f)
                if probe in show:
                    plt.show()





