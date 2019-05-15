import gm2
from gm2 import util

#details = np.arange(16)
show    = [0]#np.arange(16)


run = 3687
title = "run "+str(run)+" (slow 253.629 - 262.435 deg)"
t = gm2.Trolley([run])

time_hr, phi_hr, freq_hr = t.getBasics()
s = (time_hr[:,0]>0)&(freq_hr[:,0]>40000.0)


freq_hr_residuals = []
freq_hr_mean = []
freq_hr_std  = []
phi_hr_mean  = []
freq_ = [freq_hr[1,:]]
phi_  = [phi_hr[1,:]]
for i in range(1,freq_hr.shape[0]):
    if phi_hr[i,0] - np.array(phi_)[:,0].mean() < -0.0001:
        freq_hr_mean.append(np.array(freq_).mean(axis=0))
        freq_hr_std.append(np.array(freq_).std(axis=0))
        phi_hr_mean.append( np.array(phi_ ).mean(axis=0))
        freq_hr_residuals.append(np.array(freq_))
        freq_ = [freq_hr[i,:]]
        phi_  = [phi_hr[i,:]]
    else:
        freq_.append(freq_hr[i,:])
        phi_.append(phi_hr[i,:])

freq_hr_mean = np.array(freq_hr_mean)
freq_hr_std  = np.array(freq_hr_std)
phi_hr_mean  = np.array(phi_hr_mean)
freq_hr_residuals = np.array(freq_hr_residuals)

R = 7112
freq_hr_ds    = (phi_hr_mean[2:,:] - phi_hr_mean[:-2,:])/2. * R
#freq_hr_df    = 
freq_hr_grad1 = ( 1./12 * freq_hr_mean[4:,:] - 2./3*freq_hr_mean[3:-1,:] + 0.0  * freq_hr_mean[2:-2,:]  + 2./3 * freq_hr_mean[1:-3,:] - 1./12 * freq_hr_mean[:-4,:])/freq_hr_ds[1:-1,:]**1
freq_hr_grad2 = (-1./12 * freq_hr_mean[4:,:] + 4./3*freq_hr_mean[3:-1,:] - 5./2 * freq_hr_mean[2:-2,:]  + 4./3 * freq_hr_mean[1:-3,:] - 1./12 * freq_hr_mean[:-4,:])/freq_hr_ds[1:-1,:]**2
freq_hr_grad3 = (-1./12 * freq_hr_mean[4:,:] + 1./1*freq_hr_mean[3:-1,:] - 0./2 * freq_hr_mean[2:-2,:]  - 1./1 * freq_hr_mean[1:-3,:] + 1./12 * freq_hr_mean[:-4,:])/freq_hr_ds[1:-1,:]**3
freq_hr_grad4 = ( 1./1 * freq_hr_mean[4:,:]  - 4./1*freq_hr_mean[3:-1,:] + 6./1 * freq_hr_mean[2:-2,:]  - 4./1 * freq_hr_mean[1:-3,:] + 1./1  * freq_hr_mean[:-4,:])/freq_hr_ds[1:-1,:]**4




#import matplotlib.pyplot as plt
from gm2plotsettings import *
PPB_HZ = PPM_HZ/1000.


sl = 31
nsigma = 3.0

gradMin = -100
gradMax =  100

step = 2
grads = [[-100,100]] + [[i,i+step*2] for i in np.arange(-10,8, step)]

rr = np.zeros([len(grads), 17])


from matplotlib.backends.backend_pdf import PdfPages
with PdfPages('plots/trolleyResolutionStatic_run'+str(run)+'.pdf') as pdf:
    ii = 0
    for gradMin, gradMax in grads:
        ax = []
        hr = 60
        figsize = [plt.rcParams['figure.figsize'][1] * 2.0, plt.rcParams['figure.figsize'][1] * 2.0]
        f = plt.figure(figsize=figsize)
        plt.subplots_adjust(wspace=0.0, hspace=0.0)
        for probe in range(17):
            ax.append(plt.subplot2grid((5,5), TR.probes.grid[probe]))
            #df = np.concatenate([freq_hr_residuals[i][:, probe]-freq_hr_mean[i, probe] for i in range(len(freq_hr_residuals))])
            sel = np.argwhere((freq_hr_grad1[:,0] >= gradMin)&(freq_hr_grad1[:,0] < gradMax))
            df = np.concatenate([freq_hr_residuals[i[0]][:, probe]-freq_hr_mean[i[0], probe] for i in sel])
            if df.shape[0] < 100:
                continue
            #freq_hr_residual#smooth(freq[s, probe], sl+2, 'ref') - smooth(freq[s, probe], sl, 'lr')
            if ii in [0]:
                v_, b_, _ = ax[probe].hist(df/PPB_HZ, bins=np.arange(-hr, hr, 1.0), alpha=0.8, histtype='stepfilled') 
            else:
                v_, b_, _ = ax[probe].hist(df/PPB_HZ, bins=np.arange(-hr, hr, 5.0), alpha=0.8, histtype='stepfilled')
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
            
            rr[ii, probe] = popt[2]
            
            if TR.probes.grid[probe] == (4,2):
                ax[probe].set_xlabel("resolution [ppb]")

            if TR.probes.grid[probe] == (0,2):
                title_ = title+" :: gradients ["+str(gradMin)+","+str(gradMax)+"] Hz/mm"
                plt.title(title_)
        ii += 1
        pdf.savefig(f)
        if len(show)>0:
            plt.show()
        #else:
        #    plt.clf()

    figsize = [plt.rcParams['figure.figsize'][1] * 2.0, plt.rcParams['figure.figsize'][1] * 2.0]
    f = plt.figure(figsize=figsize)
    plt.subplots_adjust(wspace=0.0, hspace=0.0)
    ax = []
    for probe in range(17):
        ax.append(plt.subplot2grid((5,5), TR.probes.grid[probe]))
        plt.errorbar(np.arange(-10+step, 10, step), rr[1:,probe], xerr=step, fmt=' ')
        plt.plot([-10+step/2.-1, 10-step/2.+1],[rr[0, probe], rr[0, probe]], '--')
        #ax[probe].yaxis.set_ticklabels([])
        #if TR.probes.grid[probe][0] != 4:
        #    ax[probe].xaxis.set_ticklabels([])
        if TR.probes.grid[probe] == (4,2):
            ax[probe].set_xlabel("gradient [Hz/mm]")
        else:
            ax[probe].xaxis.set_ticklabels([])
        if TR.probes.grid[probe] == (2,0):
            ax[probe].set_ylabel("resolution [Hz]")
        else:
            ax[probe].yaxis.set_ticklabels([])
        ax[probe].set_ylim([0,40])
        ax[probe].text(0.15, 0.9,"#"+str(probe),\
                       horizontalalignment='center',\
                       verticalalignment='center',\
                       transform = ax[probe].transAxes)
    pdf.savefig(f)
    plt.show()

from scipy.optimize import curve_fit
def func(x, a, b):                    
    return a + b * x

slopes  = []
for p in range(17):
     popt, pcov = curve_fit(func, np.abs(np.arange(-10+step, 10, step)), rr[1:,p])
     slopes.append(popt)



'''
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

'''



