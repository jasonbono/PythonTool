import numpy as np
import matplotlib.pyplot as plt

import gm2

# look at static runs
'''
Accordiing to db
# 3928 Yoke G, Azi 1 
# 3930 Yoke G, Azi 2
# 3932 Yoke G, Azi 3
# 2410-2417, 187.1
  -> active at 185.1 -> yoke G, Azi 5

BUT!
Azi 1: 1524643000000 + 520000, peak 172.7 -> active 170.2
Azi 2: 1525653000000 + 480000, peak 176.0 -> active 173.5
Azi 3: 1524643000000 + 400000, peak 182.7 -> active 180.2

File 3928, trolley at 172.8, closest probe is Azi 2

File 2411, trolley at 187.1 -> yoke G, Azi 5

File 3751 to 3760, trolley at 176.929099 -> yoke G, Azi 2
'''


runs = [3928]# 3830, 3832
yokes = 'G'
aziIds = 2

runs = [3930]# 3830, 3832
yokes = 'G'
aziIds = 3
exclude = []
selection = {205:0.000265}


runs = np.arange(2411,2417+1) # 5h
yokes = 'G'
aziIds = 4
exclude = []
selection = {}


runs = np.arange(3751, 3760+1) # 5h
yokes = 'G'
aziIds = 3
exclude = []
selection = {}

plots = [True, True]


tr = gm2.Trolley(runs)
fp = gm2.FixedProbe(runs)

tr_time, tr_phi,  tr_freq = tr.getBasics()
tr_freq[np.isinf(tr_freq)] = np.nan
if tr_time[10,0] < 10e10:
    tr_time *= 1e9
#class Interpolation:
#    def __init__(self):
#        pass
#        #iself.tr = tr
#        #self.fp = fp
#


tr_sigma = 0.6
fp_sigma = 0.7

## Settings
fitrange=500
single = False
forceZero = 1 # intverted


def getM0(f):
    return f.mean(axis=1)

if True:
        #    def calibration(self, tr, fp, plots=[True, True]):
        #print("DEBUG", aziIds)
        #tr_time, tr_phi,  tr_freq = tr.getBasics()
        def fpCallback(mode_freq=0):
            return [fp.getId(), fp.getTimeGPS(), fp.getFrequency(mode_freq), fp.getFidLength()]
        fp_id,   fp_time, fp_freq, fp_length = fp.loop(fpCallback, yokes=yokes, aziIds=int(aziIds))
        #print("DEBUG", aziIds, fp_id[10,:])

        fp_freq_f = []
        for probe in range(fp_freq.shape[1]):
            fp_freq_f.append(gm2.util.interp1d(fp_time[:,probe], fp_freq[:,probe], kind='linear'))

        # interpolation of measuerements because each sample is taken at a different time
        fp_freq_f = []
        for probe in range(fp_freq.shape[1]):
            id_ = int(fp_id[5, probe])
            if id_ in selection:
                sel = fp_length[:,probe] > selection[id_]
            else:
                sel = np.full(fp_freq.shape[0], True)
            fp_freq_f.append(gm2.util.interp1d(fp_time[sel, probe], fp_freq[sel, probe], kind='linear'))

        tr_freq_f = []
        for probe in range(tr_freq.shape[1]):
            tr_freq_f.append(gm2.util.interp1d(tr_time[:,probe], tr_freq[:,probe], kind='linear'))


        # extract one m per cycle at the mean time
        skip = 50
        ts = fp_time[skip:-skip].mean(axis=1) # common timestamps to extract m's, exlcude first and last to be sure to be in interpolation range
        m_n = fp_freq.shape[1] # number of potential ms
        fp_m = np.zeros([ts.shape[0], m_n])
        tr_m = np.zeros([ts.shape[0], m_n])

        fp_freqs_ = np.array([fp_freq_f[j](ts) for j in np.arange(fp_freq.shape[1])]).T
        sel = np.isin(fp_id[100,:], exclude) == False
        #fp_m[:,0] = self.getM0(fp_freqs_[:,sel])
        fp_m[:,0] = getM0(fp_freqs_[:,sel])

        tr_freqs_ = np.array([tr_freq_f[j](ts) for j in np.arange(tr_freq.shape[1])]).T
        #tr_freqs_ = np.fromfunction(lambda i, j: tr_freq_f[j](ts[i]), [ts.shape[0], tr_freq.shape[1]])
        #tr_m[:,0] = self.getM0(tr_freqs_)
        tr_m[:,0] = getM0(tr_freqs_)



        ### Plot as a function of time
        if plots[0]:
            t0 = ts[0]
            ax1 = plt.subplot(211)
            for probe in np.arange(fp_freq.shape[1]):
                plt.plot((fp_time[skip:-skip, probe]-t0)/60e9, fp_freq[skip:-skip, probe]-fp_freq[skip:-skip, probe].mean(), '.', label=str(int(fp_id[100,probe])), alpha=0.4)
            plt.plot((ts-t0)/60e9, fp_m[:,0]-fp_m[:,0].mean(), '.', color='black', label=r'$m_{0}^{\rm{fp}}$', alpha=0.7)
            plt.ylabel("fixed probes\nrel freq [Hz]")
            plt.legend(ncol=7, mode='expand', prop={'size': 8})
            ax1.tick_params(labelbottom=False)
            ylim = ax1.get_ylim()

            ax2 = plt.subplot(212, sharex=ax1)
            sel = tr_time[skip:-skip].min(axis=1) > 1e5
            plt.plot((tr_time[skip:-skip]-t0)[sel]/60e9, tr_freq[skip:-skip][sel]-np.nanmean(tr_freq[skip:-skip][sel], axis=0), '.', alpha=0.4)
            plt.plot((ts-t0)/60e9, tr_m[:,0]-np.nanmean(tr_m[:,0]), '.', color='black', alpha=0.7)
            plt.ylabel("trolley\nrel freq [Hz]")
            plt.xlabel("time [min]")
            plt.ylim(ylim)
            gm2.despine()
            plt.show()

        # Investigate individual correlations
        if single:
            tr_probe = 0
            f_ref = 0.0 #61.74e6
            c = np.full([17,6], np.nan)   # correlations
            s = np.full([17,6,4], np.nan) # slopes
            #o = np.full([17,6,2], np.nan) # offsets
            for tr_probe in range(17): 
              for fp_probe in range(fp_freq.shape[1]):
                plt.subplot(321+fp_probe)
                fp_f_ = fp_freqs_[:fitrange, fp_probe] + f_ref
                tr_f_ = tr_freqs_[:fitrange, tr_probe] + f_ref
                plt.plot(fp_f_, tr_f_, '.')
                c_ = np.corrcoef(fp_f_, tr_f_)[1,0]
                c[tr_probe, fp_probe] = c_
                fit = gm2.util.FitLin(fp_f_, tr_f_, fp_sigma, tr_sigma)
                fit.fit([0.0, 1.0], ifixb=[forceZero, 1])
                xx, yy = fit.getFit(xadd=1)
                df = fit.getBand(xx)
                plt.fill_between(xx, yy+df, yy-df, alpha=0.3, color=gm2.colors[1]) 
                plt.plot(xx, yy, label="fit", color=gm2.colors[1])
                #plt.title("c: %.3f" % c)
                plt.gca().xaxis.set_ticklabels([])
                plt.gca().yaxis.set_ticklabels([])
                B_, Bsd_ = fit.getPara()
                s[tr_probe, fp_probe, :2] = B_
                s[tr_probe, fp_probe, 2:] = Bsd_
                plt.text(0.3, 0.8, r'c: '+ ("%.2f" % c_),
                     horizontalalignment='center',
                     verticalalignment='center',
                     transform = plt.gca().transAxes)
                plt.text(0.6, 0.15, "o="+("% 5.0f" % B_[0])+r'$\pm$'+("% 5.0f" % Bsd_[0])+"\n"+
                                    "s="+("%.3f" % B_[1])+r'$\pm$'+("%.3f" % Bsd_[1]),
                     horizontalalignment='center',
                     verticalalignment='center',
                     transform = plt.gca().transAxes)
              #gm2.despine()
              #plt.show()
        


            plt.close('all')
            ## predict trolley results
            n_fp = fp_freq.shape[1]
            #tr_probe = 0
            tr_freqs_pred = np.empty_like(tr_freqs_)
            for tr_probe in range(17):
                tr_freqs_pred[:,tr_probe] = (((s[tr_probe, :n_fp, 0] + s[tr_probe, :n_fp, 1] * (fp_freqs_ + f_ref)) - f_ref) / c[tr_probe,:n_fp]**2).sum(axis=1) / ((1./c[tr_probe,:n_fp]**2).sum())
           
            tr_m_pred = np.empty_like(tr_m)
            tr_m_pred[:,0] = getM0(tr_freqs_pred)
 
        #tr_freqs_pred = (s[tr_probe, :n_fp, 0] + s[tr_probe, :n_fp, 1] * fp_freqs_).sum(axis=1) / n_fp



        # Fit m0
        t0 = ts[0]
        tm = ((ts-t0)/60e9).max()
        f_offset = 0.0#-61.74e6
        fit = gm2.util.FitLin(fp_m[:fitrange,0]-f_offset, tr_m[:fitrange,0]-f_offset, fp_sigma, tr_sigma)
        fit.fit([0.0, 1.0], ifixb=[forceZero, 1])
        B, Bsd = fit.getPara()
        c = np.corrcoef(fp_m[:,0], tr_m[:,0])
        if plots[1]:
            xx, yy = fit.getFit(xadd=1)
            df = fit.getBand(xx)
            fig = plt.figure(figsize=[6.4*1.3, 4.8*1.3])
            n = ts.shape[0]//6
            for i in range(6):
                plt.errorbar(fp_m[n*i:n*(i+1),0]-f_offset, tr_m[n*i:n*(i+1),0]-f_offset, xerr=fp_sigma, yerr=tr_sigma, fmt=' ', alpha=0.3, color=gm2.colors[i])
                plt.plot(fp_m[n*i:n*(i+1),0]-f_offset, tr_m[n*i:n*(i+1),0]-f_offset, '.', color=gm2.colors[i])
            #plt.errorbar(fp_m[fitrange:,0]-f_offset, tr_m[fitrange:,0]-f_offset, xerr=fp_sigma, yerr=tr_sigma, fmt=' ', alpha=0.3, color=gm2.colors[0])
            #plt.plot(fp_m[fitrange:,0]-f_offset, tr_m[fitrange:,0]-f_offset, '.', color=gm2.colors[0])
            #plt.errorbar(fp_m[:fitrange,0]-f_offset, tr_m[:fitrange,0]-f_offset, xerr=fp_sigma, yerr=tr_sigma, fmt=' ', alpha=0.3, color=gm2.colors[2])
            #plt.plot(fp_m[:fitrange,0]-f_offset, tr_m[:fitrange,0]-f_offset, '.', color=gm2.colors[2])
            #plt.fill_between(xx, yy+df, yy-df, alpha=0.3, color=gm2.colors[1]) 
            #plt.plot(xx, yy, label="fit", color=gm2.colors[1])
            #ct = plt.cm.jet((ts-t0)/60e9/tm)
            #ct = [[0,0,0] for cc in ts]
            #plt.errorbar(fp_m[:,0]-f_offset, tr_m[:,0]-f_offset, xerr=fp_sigma, yerr=tr_sigma, fmt=' ', alpha=0.3, color=ct)
            #for i in np.arange(fp_m.shape[0]):
            #plt.errorbar(fp_m[:,0]-f_offset, tr_m[:,0]-f_offset, xerr=fp_sigma, yerr=tr_sigma, fmt=' ', alpha=0.3, color=ct)
            #    plt.plot(fp_m[i,0]-f_offset,     tr_m[i,0]-f_offset,                                   '.',            c=[0., 1. - (ts[i]-t0)/60e9/tm, (ts[i]-t0)/60e9/tm])
            plt.xlabel(r'fixed probe $m_{0}^{\rm{fp}}$ [Hz]')
            plt.ylabel(r'trolley $m_{0}^{\rm{try}}$ [Hz]')
            #Bsd = odrout.sd_beta
            plt.text(0.5, 0.95,r'$m_{0}^{\rm{try}} = ($'+("%.0f" % B[0])+r'$\pm$'+(("%.0f" % Bsd[0]))+")Hz + ("+("%.6f" % B[1])+r'$\pm$'+("%.6f" % Bsd[1])+r') $\cdot m_{0}^{\rm{fp}}$',
                 horizontalalignment='center',
                 verticalalignment='center',
                 transform = plt.gca().transAxes) 
            plt.text(0.3, 0.85, r'correlation: '+ ("%.3f" % c[1, 0]),
                 horizontalalignment='center',
                 verticalalignment='center',
                 transform = plt.gca().transAxes)
            gm2.despine()
            plt.show()

        tr_m_pred2 = (B[0] + B[1] * (fp_m[:,0] - f_offset)) + f_offset

        plt.close('all')
        t0 = ts[0]
        if single:
            plt.plot((ts[:]-t0)/60e9, tr_freqs_pred[:,0] - tr_freqs_[:,0],'.')
            plt.plot((ts[:]-t0)/60e9, tr_m_pred[:,0]     - tr_m[:,0], '.')
        plt.plot((ts[:fitrange]-t0)/60e9, tr_m_pred2[:fitrange] - tr_m[:fitrange,0], '.')
        plt.plot((ts[fitrange:]-t0)/60e9, tr_m_pred2[fitrange:] - tr_m[fitrange:,0], '.')
        plt.xlabel("time [min]")
        plt.ylabel(r'$m_{0\rm{,pred}}^{\rm{tr}} - m_{0}^{\rm{tr}}$')
        gm2.despine()
        #plt.title("run 3930")
        plt.show()
        #return B, Bsd, c[1, 0]

#ip = Interpolation()
#
#B_, Bsd_, c_ = ip.calibration(tr, fp, [True, True])
'''
yokes_l = ['A','B','C','D','E','F','G','H','I','J','K','H']
yokes_l = ['G']

Bs  = []
Bsd = []
c = []
for yokes in yokes_l:
    for aziIds in np.arange(1,7):
        B_, Bsd_, c_ = ip.calibration(tr, fp, [False, True])
        Bs.append(B_)
        Bsd.append(Bsd_)
        c.append(c_)
'''

# TODOs
## move m calculation to utils or class
## studies
### - difference is fp which are further away?
### - how stable is ist over time? subsamples



