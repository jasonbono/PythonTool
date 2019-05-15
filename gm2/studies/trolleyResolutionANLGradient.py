import Trolley
from util import *
from gm2constants import *

t = Trolley.Trolley([])
t.basedir = '~/ana/data/'
t.fname_suffix = '.root'
t.fname_prefix = 'TrolleyGraphOut'
t.fname_path = 'ExtractTrolleyFid/trolleyWf'
t.add(1056)
def callback():
    return [t.getFrequencys(True)[:,0], t.getPhis()[:,0], t.getTimesGPS(), t.getFidLengths()]

freq, phi, time, length = t.loop(callback)#t.getBasics()

I = [ 0.01, 3.01, 6.08, 9.17, 0.01]
grad = np.array(I)*67.55 + 13.3

res = [freq[  1:50 ,:].std(axis=0),    # 0
       freq[ 50:100,:].std(axis=0),    # 3
       freq[100:150,:].std(axis=0),    # 6
       freq[150:200,:].std(axis=0),    # 9
       freq[200:   ,:].std(axis=0)]    # 0
res = np.array(res)

mean = [freq[  1:50 ,:].mean(axis=0),  # 0
        freq[ 50:100,:].mean(axis=0),  # 3
        freq[100:150,:].mean(axis=0),  # 6
        freq[150:200,:].mean(axis=0),  # 9
        freq[200:   ,:].mean(axis=0)]  # 0

leng = [length[  1:50 ,:].mean(axis=0),  # 0
        length[ 50:100,:].mean(axis=0),  # 3
        length[100:150,:].mean(axis=0),  # 6
        length[150:200,:].mean(axis=0),  # 9
        length[200:   ,:].mean(axis=0)]  # 0
leng = np.array(leng)



slopes_slow = np.array([
 [7.19457262, 0.79043593],
 [11.43171957,  0.56069298],
 [6.46640348, 0.72250876],  
 [8.1121022 , 0.90680095],
 [7.35353118, 0.63487174],
 [15.89741875,  1.02540372],
 [12.91003162,  0.38129294],
 [8.50958752, 0.37295069],
 [7.87952638, 0.5272919 ],
 [7.79725218, 0.55986803],
 [20.83051919,  1.90723658],
 [11.76077553,  1.16456966],
 [9.91665297, 0.93979064],
 [7.61938065, 0.34664415],
 [7.3328637 , 0.79899212],
 [12.37807754,  0.89202807],
 [18.7019823,  2.0698851]])


'''
tb = Trolley.Trolley([])
tb.basedir = '~/ana/data/'
tb.fname_suffix = '.root'
tb.fname_prefix = 'TrolleyGraphOut'
tb.fname_path = 'ExtractTrolleyFid/trolleyWf'
tb.add(1049)
def callbackb():
        return [tb.getFrequencys(True)[:,0], tb.getPhis()[:,0], tb.getTimesGPS(), tb.getFidLengths()]
freq_b, phi_b, time_b, length_b = tb.loop(callbackb)#t.getBasics()

Ib = np.arange(0-7.0, 8.0, 1.0)
res_b = []
for i in range(len(Ib)):
    res_b.append(np.nanstd(freq_b[1+50*i:50+50*i ,:], axis=0))
res_b = np.array(res_b)
'''

'''
# step 1: check linearity
import matplotlib.pyplot as plt
plt.plot(I, mean, '.')
plt.show()
'''

### Overlay with high resolution data in ring
if True:
    t_hr = Trolley.Trolley([3687])
    def callback_hr():
        return [t_hr.getFrequencys(True)[:,0], t_hr.getPhis()[:,0], t_hr.getTimesGPS(), t_hr.getFidLengths(), t_hr.getAmplitudes()]

    freq_hr, phi_hr, time_hr, length_hr, amp_hr = t_hr.loop(callback_hr)#t_hr.getBasics()

    freq_hr_mean =   []
    freq_hr_std  =   []
    phi_hr_mean  =   []
    length_hr_mean = []
    length_hr_std  = []
    amp_hr_mean    = []
    freq_   = [freq_hr[1,:]]
    phi_    = [phi_hr[1,:]]
    length_ = [length_hr[1,:]]
    amp_    = [amp_hr[1,:]]
    for i in range(1,freq_hr.shape[0]):
        if phi_hr[i,0] - np.array(phi_)[:,0].mean() < -0.0001:
            freq_hr_mean.append(np.array(freq_).mean(axis=0))
            freq_hr_std.append(np.array(freq_).std(axis=0))
            phi_hr_mean.append( np.array(phi_ ).mean(axis=0))
            length_hr_mean.append( np.array(length_ ).mean(axis=0))
            length_hr_std.append( np.array(length_ ).std(axis=0))
            amp_hr_mean.append( np.array(amp_ ).std(axis=0))
            freq_   = [freq_hr[i,:]]
            phi_    = [phi_hr[i,:]]
            length_ = [length_hr[i,:]]
            amp_    = [amp_hr[i,:]]
        else:
            freq_.append(freq_hr[i,:])
            phi_.append(phi_hr[i,:])
            length_.append(length_hr[i,:])
            amp_.append(amp_hr[i,:])

    freq_hr_mean = np.array(freq_hr_mean)
    freq_hr_std  = np.array(freq_hr_std)
    phi_hr_mean  = np.array(phi_hr_mean)
    length_hr_mean = np.array(length_hr_mean)
    length_hr_std  = np.array(length_hr_std)
    amp_hr_mean    = np.array(amp_hr_mean)

    freq_hr_df   = freq_hr_mean[2:,:] - freq_hr_mean[:-2,:]
    freq_hr_ds   = (phi_hr_mean[2:,:]  - phi_hr_mean[:-2,:])*7110
    freq_hr_grad = freq_hr_df/freq_hr_ds



    from gm2plotsettings import *
'''
    plt.plot(freq_hr_grad[:,0], length_hr_mean[1:-1,0], '.')
    plt.xlabel("gradient [Hz/mm]")
    plt.ylabel("FID length [?]")
    plt.savefig("plots/gradientVsFidLength.pdf")
    sns.despine()
    plt.show()
'''


def func(x, a, b, c):
    return a + b*np.exp(x*c)

from scipy.optimize import curve_fit
from gm2plotsettings import *
plt.tight_layout()
probe = 0
slopes = []


from matplotlib.backends.backend_pdf import PdfPages
with PdfPages('plots/trolleyResolutionStatic_anl.pdf') as pdf:

    figsize = [plt.rcParams['figure.figsize'][1] * 2.0, plt.rcParams['figure.figsize'][1] * 2.0]
    f = plt.figure(figsize=figsize)
    plt.subplots_adjust(wspace=0.0, hspace=0.0)
    ax = []

    for probe in range(17):
        ax.append(plt.subplot2grid((5,5), TR.probes.grid[probe]))
        popt, pcov = curve_fit(func, grad, res[:, probe], p0=[0.2, 0.01, 1.0/200])
        sel = [0,1,2,4]
        popt2 = fit_lin(grad[sel]*PPB2HZ, res[sel, probe])
        #slopes.append(popt2)
        l1,_,_ = ax[probe].errorbar(grad, np.array(res)[:, probe], xerr=np.array(I)*0.1, fmt='x', color=sns.color_palette()[0], label="ANL")
        gg = np.arange(0, grad.max()*0.8, 10)
        #plt.plot(gg, func(gg, *popt),'--')
        ax[probe].plot(gg, func_lin(gg*PPB2HZ, *popt2),'--', color=sns.color_palette()[0])
        gg = np.arange(0, 300, 10)
        ax[probe].plot(gg, func_lin(gg*PPB2HZ, slopes_slow[probe][0], slopes_slow[probe][1])*PPB2HZ, '-.',color=sns.color_palette()[1])
        l2, = ax[probe].plot(np.abs(freq_hr_grad[:, probe])*HZ2PPB, freq_hr_std[1:-1, probe], '.', label="FNAL", color=sns.color_palette()[1], alpha=0.1)
        if TR.probes.grid[probe] == (4,2):
            ax[probe].set_xlabel("gradient [ppb/mm]")
        else:
            ax[probe].xaxis.set_ticklabels([])
        if TR.probes.grid[probe] == (2,0):
            ax[probe].set_ylabel("resolution [Hz]")
        else:
            ax[probe].yaxis.set_ticklabels([])
        ax[probe].set_ylim([0,2])
        ax[probe].text(0.15, 0.9,"#"+str(probe),\
                      horizontalalignment='center',\
                      verticalalignment='center',\
                      transform = ax[probe].transAxes)

    ax_leg = plt.subplot2grid((5,5), (0,4))
    ax_leg.legend([l1, l2],["ANL", "FNAL\n(slow)"])
    ax_leg.xaxis.set_ticklabels([])
    ax_leg.yaxis.set_ticklabels([])
    #plt.show()
    pdf.savefig(f)
    #plt.clf(f)

    plt.clf()
    f = plt.figure()
    for probe in range(17):
        popt, pcov = curve_fit(func, grad, res[:, probe], p0=[0.2, 0.01, 1.0/200])
        sel = [0,1,2,4]
        popt2 = fit_lin(grad[sel]*PPB2HZ, res[sel, probe])
        slopes.append(popt2)
        plt.errorbar(grad, np.array(res)[:, probe], xerr=np.array(I)*0.1, fmt='x', color=sns.color_palette()[0], label="ANL")
        gg = np.arange(0, grad.max()*0.8, 10)
        #plt.plot(gg, func(gg, *popt),'--')
        plt.plot(gg, func_lin(gg*PPB2HZ, *popt2),'--', color=sns.color_palette()[0])
        gg = np.arange(0, 300, 10)
        plt.plot(gg, func_lin(gg*PPB2HZ, slopes_slow[probe][0], slopes_slow[probe][1])*PPB2HZ, '-.',color=sns.color_palette()[1])
        plt.xlabel("gradient [ppb/mm]")
        ax1 = plt.gca()
        ax1.set_xlim([-50,700])
        ax1.set_ylim([0, 3])
        ax2 = ax1.twiny()
        ax2.set_xlim([ax1.get_xlim()[0]*61.78/1000.0 , ax1.get_xlim()[1]*61.78/1000.0 ])
        ax2.set_xlabel("gradient [Hz/mm]")
        ax1.set_ylabel("resolution [Hz]")
        ax3 = ax1.twinx()
        ax3.set_ylabel("resolution [ppb]")
        ax3.set_ylim([ax1.get_ylim()[0]/61.78*1000.0 , ax1.get_ylim()[1]/61.78*1000.0 ])
        ax1.plot(freq_hr_grad[:, probe]*HZ2PPB, freq_hr_std[1:-1, probe], '.', label="FNAL", color=sns.color_palette()[1], alpha=0.1)
        #sns.despine()
        #plt.show()
        ax1.legend()
        pdf.savefig(plt.gcf())
        plt.clf()

    figsize = [plt.rcParams['figure.figsize'][1] * 2.0, plt.rcParams['figure.figsize'][1] * 2.0]
    f = plt.figure(figsize=figsize)
    plt.subplots_adjust(wspace=0.0, hspace=0.0)
    ax = []
    for probe in range(17):
        ax.append(plt.subplot2grid((5,5), TR.probes.grid[probe]))
        l2, = ax[probe].plot(np.abs(freq_hr_grad[:, probe]/61.78*1000.0), length_hr_mean[1:-1, probe],'.', label="FNAL", color=sns.color_palette()[1], alpha=0.1)
        l1, = ax[probe].plot(grad, leng[:, probe],'x', label="ANL", color=sns.color_palette()[0])
        
        if TR.probes.grid[probe] == (4,2):
            ax[probe].set_xlabel("gradient [ppb/mm]")
        else:
            ax[probe].xaxis.set_ticklabels([])
        if TR.probes.grid[probe] == (2,0):
            ax[probe].set_ylabel("fid length []")
        else:
            ax[probe].yaxis.set_ticklabels([])
        #ax[probe].set_ylim([0,0.0001])
        ax[probe].text(0.15, 0.9,"#"+str(probe),\
                      horizontalalignment='center',\
                      verticalalignment='center',\
                      transform = ax[probe].transAxes)

    ax_leg = plt.subplot2grid((5,5), (0,4))
    ax_leg.legend([l1, l2],["ANL", "FNAL\n(slow)"])
    ax_leg.xaxis.set_ticklabels([])
    ax_leg.yaxis.set_ticklabels([])
    #plt.show()
    pdf.savefig(f)
    plt.clf()

'''
probe = 0
for probe in range(5, 17):
    plt.plot(np.abs(freq_hr_grad[:, probe]/61.78*1000.0), length_hr_mean[1:-1, probe],'.', label="FNAL", color=sns.color_palette()[0], alpha=0.1)
    plt.plot(grad, leng[:, probe],'x', label="ANL", color=sns.color_palette()[1])
    plt.xlabel("gradient [Hz/mm]")
    plt.ylabel("fid length [?]")
    plt.legend()
    sns.despine()
    plt.show()


freq_hr_grad1 = ( 1./12 * freq_hr_mean[4:,:] - 2./3*freq_hr_mean[3:-1,:] + 0.0  * freq_hr_mean[2:-2,:]  + 2./3 * freq_hr_mean[1:-3,:] - 1./12 * freq_hr_mean[:-4,:])
freq_hr_grad2 = (-1./12 * freq_hr_mean[4:,:] + 4./3*freq_hr_mean[3:-1,:] - 5./2 * freq_hr_mean[2:-2,:]  + 4./3 * freq_hr_mean[1:-3,:] - 1./12 * freq_hr_mean[:-4,:])
freq_hr_grad3 = (-1./12 * freq_hr_mean[4:,:] + 1./1*freq_hr_mean[3:-1,:] - 0./2 * freq_hr_mean[2:-2,:]  - 1./1 * freq_hr_mean[1:-3,:] + 1./12 * freq_hr_mean[:-4,:])
freq_hr_grad4 = ( 1./1 * freq_hr_mean[4:,:]  - 4./1*freq_hr_mean[3:-1,:] + 6./1 * freq_hr_mean[2:-2,:]  - 4./1 * freq_hr_mean[1:-3,:] + 1./1  * freq_hr_mean[:-4,:])
probe = 0
#for probe in range(5, 17):
plt.plot((freq_hr_grad1[:, probe]/61.78*1000.0), freq_hr_std[2:-2, probe],'.', label="FNAL")

s = np.abs(freq_hr_grad3[:,probe])/61.78*1000.0 > 100
plt.plot((freq_hr_grad1[:, probe][s]/61.78*1000.0), freq_hr_std[2:-2, probe][s],'.', label="FNAL")
plt.show()

#plt.plot(grad, leng[:, probe],'o', label="ANL")
#plt.xlabel("gradient [Hz/mm]")
#plt.ylabel("fid length [?]")
#plt.legend()
#sns.despine()
plt.show()
'''
