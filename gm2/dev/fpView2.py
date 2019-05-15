import gm2
from gm2 import plt, np
from gm2.plotutil import plot_ts

#runs = np.arange(5807, 5835+1) #[5807]
runs = np.arange(5807, 5808+1) #[5807]
fp = gm2.FixedProbe(runs)


def callback():
    return [fp.getTimeSystem(), fp.getFrequency(0)]
#_, fp_time, fp_freq = fp.getBasics()
fp_time, fp_freq = fp.loop(callback)

fp_phi = fp.getPhi()

skip = 1

ylim = 500.
ylim2 = 50.

def selectStation(yoke, azi):
    return (fp.id['yoke'] == yoke)&(fp.id['azi'] == azi)

def selectProbe(rad, layer):
    return (fp.id['rad'] == ord(rad))&(fp.id['layer'] == ord(layer))

layers = ["T", "B"]
radIds = ["O", "M", "I"]

alpha = 1.0
figsize = [gm2.plt.rcParams['figure.figsize'][1] * 2.0, gm2.plt.rcParams['figure.figsize'][1] * 1.5]
from matplotlib.dates import DateFormatter
formatter = DateFormatter('%m/%d\n%H:%M')


ddt = (gm2.util.datetime2ts(2018,10,12,17,31,0) - gm2.util.datetime2ts(2018,10,12,17,30,0)) * 1e9

times = [(gm2.util.datetime2ts(2018,10,12,17,20,0) + ii*60)*1e9 for ii in np.arange(180)]

mean = np.zeros([12*6,len(times)])
phis = np.zeros([12*6])

plot = False
test = 0
ii = 0
for yoke in np.arange(ord('A'), ord('L')+1):
    for aziId in np.arange(1,6+1):
        ii += 1
        print("Yoke "+chr(yoke)," : "+str(aziId))
        if test:
            break
        s_station = selectStation(yoke, aziId)
        phis[ii-1] = fp_phi[s_station][0]
 
        for jj, tt in enumerate(times): 
            ss = (fp_time[:,s_station] > tt-ddt)&(fp_time[:,s_station] < tt+ddt)
            mean[ii-1,jj] = fp_freq[:,s_station][ss].mean()

for tt in np.arange(1,180):
    plt.plot(phis, mean[:,tt]-mean[:,0])
    plt.xlabel("azimuth [degree]")
    plt.ylabel('relative "jump" amplitude [Hz]')
    plt.title("%i minutes after 17:20 (10/12)" % tt)
    plt.ylim([-500,200])
    gm2.despine()
    plt.savefig("plots/t_%03i.png" % tt)
    plt.close('all')

'''
        freq_n    = 0
        f = plt.figure(figsize=figsize)
        ax1 = plt.subplot(211)
        for radId in radIds:
            for layer in layers:
                s = selectProbe(radId, layer)&s_station
                probe = np.argwhere(s)
                if len(probe) == 1:
                    mean = fp_freq[skip:, s].mean()
                    probe = probe[0]
                    label_ = layer
                    if radId in ["I"]:
                        label_ += radId+"  "
                    else:
                        label_ += radId
                    phi_ = fp_phi[s]
                    label_ += (r" $f_{0}^{\#%03i}=$%.3fKHz"  % (probe, mean/1e3))
                    s_t = fp_time[skip:, s] > 0
                    if len(fp_time[skip:, s][s_t]) > 0:
                        if freq_n == 0:
                            freq_mean = fp_freq[:, s]-mean
                        else:
                            freq_mean += fp_freq[:, s]-mean
                        freq_n += 1
                        plot_ts(fp_time[skip:, s][s_t], fp_freq[skip:, s][s_t]-mean, '.', markersize=2, label=label_, alpha=alpha)
        plt.ylabel(r'$f^{\#} - f_{0}^{\#}$ [Hz]')
        plt.ylim([-ylim, ylim])

        freq_mean /= freq_n
        plt.subplot(212, sharex=ax1)
        for radId in radIds:
            for layer in layers:
                s = selectProbe(radId, layer)&s_station
                probe = np.argwhere(s)
                if len(probe) == 1:
                    mean = fp_freq[skip:, s].mean()
                    s_t = fp_time[skip:, s] > 0
                    if len(fp_time[skip:, s][s_t]) > 0:
                        plot_ts(fp_time[skip:, s][s_t], fp_freq[skip:, s][s_t]-mean-freq_mean[skip:][s_t], '.', markersize=2, label=label_, alpha=alpha)

        plt.xlabel("time")
        plt.ylabel(r'$(f^{\#} - f_{0}^{\#}) - f_{\rm{mean}}^{\rm{station}}$ [Hz]')
        plt.ylim([-ylim2, ylim2])

        plt.gca().xaxis.set_major_formatter(formatter)
        #ax.xaxis.set_tick_params(rotation=30, labelsize=10)
        gm2.despine()
        plt.subplot(211)
        plt.setp(ax1.get_xticklabels(), visible=False)
        #plt.gca().set_xticklabels([]);
        lgnd = plt.legend(loc='upper center', bbox_to_anchor=(0.2, -.22, 0.6, 0.2), ncol=3, fontsize=12)
        for lgndHandl in lgnd.legendHandles:
            lgndHandl._legmarker.set_markersize(12)
        plt.title("Yoke "+chr(yoke)+", azi "+str(aziId)+(" @ %.0f°" % ((phi_[0]+360)%360))  )
        f.savefig("plots/fp_"+chr(yoke)+"_"+str(aziId)+".png")
        #plt.show()
        #test = 1'''


'''
import mlpy
omega0 = 8
wavelet_fct = "morlet"

test = 0
for yoke in np.arange(ord('A'), ord('L')+1):
    for aziId in np.arange(1,6+1):
        if test:
            break
        s_station = selectStation(yoke, aziId)

        for radId in radIds:
            for layer in layers:
                s = selectProbe(radId, layer)&s_station
                probe = np.argwhere(s)
                if len(probe) == 1:
                    mean = fp_freq[skip:, s].mean()
                    probe = probe[0]
                    label_ = layer
                    if radId in ["I"]:
                        label_ += radId+"  "
                    else:
                        label_ += radId
                    phi_ = fp_phi[s]
                    label_ += (r" $f_{0}^{\#%03i}=$%.3fKHz"  % (probe, mean/1e3))
                    s_t = fp_time[skip:, s] > 0
                    if len(fp_time[skip:, s][s_t]) > 0:
                        freq = fp_freq[skip:, s][s_t]


                        scales = mlpy.wavelet.autoscales(N=len(freq), dt=1.8, dj=0.05, wf=wavelet_fct, p=omega0)
                        spec = mlpy.wavelet.cwt(freq, dt=1.8, scales=scales, wf=wavelet_fct, p=omega0)
                        # approximate scales through frequencies
                        freq_ = (omega0 + np.sqrt(2.0 + omega0 ** 2)) / (4 * np.pi * scales[1:])
                        t = np.arange(tr.stats.npts) / tr.stats.sampling_rate 
                        img = plt.gca().imshow(np.abs(spec), extent=[t[0], t[-1], freq[-1], freq[0]],
                                        aspect='auto', interpolation='nearest', cmap=obspy_sequential)
# Hackish way to overlay a logarithmic scale over a linearly scaled image.
                        twin_ax = ax2.twinx()
                        twin_ax.set_yscale('log')
                        twin_ax.set_xlim(t[0], t[-1])
                        twin_ax.set_ylim(freq[-1], freq[0])
                        plt.gca().tick_params(which='both', labelleft=False, left=False)
                        twin_ax.tick_params(which='both', labelleft=True, left=True, labelright=False)
                        plt.show()

                        #f, t, Sxx = signal.spectrogram(freq, 1.8, nfft=100000)
                        #plt.pcolormesh(t, f, np.log(Sxx))
                        #plt.semilogy()
                        #plt.ylim([0, 0.002])
                        #plt.show()
                        #sp = np.fft.fft(freq)
                        #freq_ = np.fft.fftfreq(freq.shape[-1], d=(fp_time[skip:, s][s_t].max() - fp_time[skip:, s][s_t].min())/1e9/fp_time[skip:, s][s_t].shape[-1]) # in Hz
                        #plt.semilogx(1./freq_/60, np.absolute(sp), '-', markersize=2, label=label_, alpha=alpha)
                        #plt.semilogx(60./freq_, np.absolute(sp), '-', markersize=2, label=label_, alpha=alpha)
                        #plt.xlim([0,60*24*2.])


        #plt.xlabel("frequency")
        #plt.ylabel("amplitude")
        #plt.show()

        test = 1
'''
