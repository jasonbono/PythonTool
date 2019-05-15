import gm2
from gm2 import plt, np
from gm2.plotutil import plot_ts

runs = np.arange(5809, 5835+1) #[5807]
#runs2 = np.arange(3153, 3192+1) #[5807]
#runs2 = np.arange(3153, 3192+1) #[5807]
runs2 = np.arange(2815, 2872+1) #[5807]
#runs = np.arange(5807, 5808+1) #[5807]
fp = gm2.FixedProbe(runs)
fp2 = gm2.FixedProbe(runs2)

def callback():
    return [fp.getTimeSystem(), fp.getFrequency(0)]
#_, fp_time, fp_freq = fp.getBasics()
fp_time, fp_freq = fp.loop(callback)

def callback2():
    return [fp2.getTimeSystem(), fp2.getFrequency(0)]
#_, fp_time, fp_freq = fp.getBasics()
fp2_time, fp2_freq = fp2.loop(callback2)

fp_phi = fp.getPhi()

skip = 1
skip2b = 1

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
formatter = DateFormatter('day %d\n%H:%M')

dt = gm2.util.datetime2ts(2018,10,12,0,0,0)*1e9 - gm2.util.datetime2ts(2018,2,7,0,0,0)*1e9
dt = gm2.util.datetime2ts(2018,10,12,0,0,0)*1e9 - gm2.util.datetime2ts(2018,3,3,0,0,0)*1e9
offset = gm2.util.datetime2ts(2018,10,12,0,0,0)*1e9 - gm2.util.datetime2ts(2018,10,1,0,0,0)*1e9

sq  = fp_freq[skip:,:].std(axis=0)  < 150
sq2 = fp2_freq[skip:-10000,:].std(axis=0) < 150
sq2 = fp2_freq[skip:-skip2b,:].std(axis=0) < 150
sqq = (fp2_freq[skip:-skip2b,sq2]>10000)&(fp2_freq[skip:-skip2b,sq2]<100000)
plot_ts(fp_time[skip:,sq].mean(axis=1) - offset, (np.nanmean(fp_freq[skip:, sq], axis=1)   - np.nanmean(fp_freq[skip:, sq]))/61.78,   '.', markersize=2, label="With Insulation: 10/12")
plot_ts(fp2_time[skip:-skip2b,sq2].mean(axis=1) + dt - offset, (np.nanmean(fp2_freq[skip:-skip2b, sq2], axis=1) - np.nanmean(fp2_freq[skip:-skip2b,sq2][sqq]))/61.78, '.', markersize=2, label="Before Insulation: 02/10")
#plt.xlabel("date")
plt.ylabel("dipole variation [ppm]")
plt.gca().xaxis.set_major_formatter(formatter)
from matplotlib.ticker import MaxNLocator
plt.gca().xaxis.set_major_locator(MaxNLocator(8))
plt.legend()
gm2.despine()
plt.show()


#norm_quad = np.zeros([fp_time[skip:,:].shape[0], 12*6])
#skew_quad = np.zeros([fp_time[skip:,:].shape[0], 12*6])



layers = ["T", "B"]
radIds = ["O", "M", "I"]

test = 0
formatter2 = DateFormatter('%m/%d\n%H:%M')
for yoke in np.arange(ord('A'), ord('L')+1):
    for aziId in np.arange(1,6+1):
        print("Yoke "+chr(yoke)," : "+str(aziId))
        if test:
            break
        s_station = selectStation(yoke, aziId)
        freq_mean  =  np.nanmean(fp_freq[skip:, s_station] -  np.nanmean(fp_freq[skip:, s_station], axis=0), axis=1)
        freq_mean2 =  np.nanmean(fp2_freq[skip:, s_station] - np.nanmean(fp2_freq[skip:, s_station], axis=0), axis=1)
        freq_n    = 0
        f = plt.figure(figsize=figsize)
        ax1 = plt.subplot(211)
        for radId in radIds:
            for layer in layers:
                s = selectProbe(radId, layer)&s_station
                label_ = layer
                if radId in ["I"]:
                    label_ += radId+"  "
                else:
                    label_ += radId
                if len(fp_phi[s])>0:
                    phi_ = fp_phi[s]
                probe = np.argwhere(s)
                if len(probe) == 1:
                    #mean = fp_freq[skip:, s].mean()
                    probe = probe[0]
                    label_ = "#"#(r" $f_{0}^{\#%03i}=$%.3fKHz"  % (probe, mean/1e3))
                    s_t = fp_time[skip:, s] > 0
                    if len(fp_time[skip:, s][s_t]) > 0:
                        plot_ts(fp_time[skip:, s][s_t], fp_freq[skip:, s][s_t]-np.nanmean(fp_freq[skip:, s], axis=0) - freq_mean, '.', markersize=2, label=label_, alpha=alpha)
        plt.ylabel(r'$(f^{\#} - f_{0}^{\#}) - f_{\rm{mean}}^{\rm{station}}$ [Hz]')
        plt.ylim([-ylim2, ylim2])
        plt.gca().xaxis.set_major_formatter(formatter2)
        ax2 = plt.subplot(212)
        for radId in radIds:
            for layer in layers:
                s = selectProbe(radId, layer)&s_station
                probe = np.argwhere(s)
                print("DEBUG", probe)
                if len(probe) == 1:
                #    plot_ts(fp2_time[skip:, s], fp2_freq[skip:, s]-freq_mean2 - np.nanmean(fp2_freq[skip:, s], axis=0), '.', markersize=2, label=label_, alpha=alpha)
                    s_t = (fp2_time[skip:, s] > -10)&(fp2_time[skip:, s] < 1e20)&(fp2_freq[skip:, s]>10000)&(fp2_freq[skip:, s] < 100000)
                    print("DEBUG2", len(fp2_time[skip:, s][s_t]))
                    if len(fp2_time[skip:, s][s_t]) > 0:
                        plot_ts(fp2_time[skip:, s][s_t], fp2_freq[skip:, s][s_t] - freq_mean2[s_t[:,0]] - np.nanmean(fp2_freq[skip:, s][s_t], axis=0), '.', markersize=2, label=label_, alpha=alpha)
        plt.ylabel(r'$f^{\#} - f_{0}^{\#}$ [Hz]')
        plt.ylim([-ylim2, ylim2])
        plt.ylabel(r'$(f^{\#} - f_{0}^{\#}) - f_{\rm{mean}}^{\rm{station}}$ [Hz]')
        #plt.gca().xaxis.set_major_formatter(formatter2)
        #ax.xaxis.set_tick_params(rotation=30, labelsize=10)
        gm2.despine()
        plt.subplot(211)
        plt.setp(ax1.get_xticklabels(), visible=False)
        #plt.gca().set_xticklabels([]);
        lgnd = plt.legend(loc='upper center', bbox_to_anchor=(0.2, -.22, 0.6, 0.2), ncol=3, fontsize=12)
        for lgndHandl in lgnd.legendHandles:
            lgndHandl._legmarker.set_markersize(12)
        plt.title("Yoke "+chr(yoke)+", azi "+str(aziId)+(" @ %.0f°" % ((phi_[0]+360)%360))  )
        f.savefig("plots/fp_comp_"+chr(yoke)+"_"+str(aziId)+".png")
        #plt.show()
        #test = 1



norm_quad = []
skew_quad = []
for yoke in np.arange(ord('A'), ord('L')+1):
    for aziId in np.arange(1,6+1):
        print("Yoke "+chr(yoke)," : "+str(aziId))
        s_station = selectStation(yoke, aziId)
        
        s_top = s_station&(fp.id['layer'] == ord('T'))&sq
        s_bot = s_station&(fp.id['layer'] == ord('B'))&sq
        skew_quad.append(fp_freq[skip:,s_top].mean(axis=1) - fp_freq[skip:,s_bot].mean(axis=1))
        s_i = s_station&(fp.id['rad'] == ord('I'))&sq
        s_o = s_station&(fp.id['rad'] == ord('O'))&sq
        if(fp_freq[skip:,s_i].shape[1]>0)&(fp_freq[skip:,s_o].shape[1]>0):
            norm_quad.append(fp_freq[skip:,s_o].mean(axis=1) - fp_freq[skip:,s_i].mean(axis=1))

norm_quad = np.array(norm_quad)/18.*4.5
skew_quad = np.array(skew_quad)/18.*4.5

norm_quad2 = []
skew_quad2 = []
for yoke in np.arange(ord('A'), ord('L')+1):
    for aziId in np.arange(1,6+1):
        print("Yoke "+chr(yoke)," : "+str(aziId))
        s_station = selectStation(yoke, aziId)
        
        s_top = s_station&(fp.id['layer'] == ord('T'))&sq2
        s_bot = s_station&(fp.id['layer'] == ord('B'))&sq2
        skew_quad2.append(np.nanmean(fp2_freq[skip:-skip2b,s_top], axis=1) - np.nanmean(fp2_freq[skip:-skip2b,s_bot], axis=1))
        s_i = s_station&(fp.id['rad'] == ord('I'))&sq2
        s_o = s_station&(fp.id['rad'] == ord('O'))&sq2
        if(fp2_freq[skip:,s_i].shape[1]>0)&(fp2_freq[skip:,s_o].shape[1]>0):
            norm_quad2.append(fp2_freq[skip:,s_o].mean(axis=1) - fp2_freq[skip:,s_i].mean(axis=1))

norm_quad2 = np.array(norm_quad2)/18.*4.5
skew_quad2 = np.array(skew_quad2)/18.*4.5




qq = (norm_quad2[:,:-skip2b]>-1000)&(norm_quad2[:,:-skip2b]<1000)
plot_ts(fp2_time[skip:-skip2b,sq2].mean(axis=1) + dt - offset, (np.nanmean(norm_quad2[:,:-skip2b], axis=0) - np.nanmean(norm_quad2[:,:-skip2b][qq]))/61.78-0.5+39.0, '.', markersize=2, label="Before Insulation: 03/03", color=gm2.colors[1])
plot_ts(fp_time[skip:,sq].mean(axis=1) - offset,             (np.nanmean(norm_quad, axis=0) - np.nanmean(norm_quad))/61.78, '.', markersize=2, label="With Insulation: 10/12", color=gm2.colors[0])
plt.ylabel("normal quadrupole variations \n [ppm @ 4.5cm]")
plt.xlabel("date")
plt.gca().xaxis.set_major_formatter(formatter)
from matplotlib.ticker import MaxNLocator
plt.gca().xaxis.set_major_locator(MaxNLocator(8))
plt.legend()
gm2.despine()
plt.show()


qq = (skew_quad2[:,:-skip2b]>-1000)&(skew_quad2[:,:-skip2b]<1000)
plot_ts(fp2_time[skip:-skip2b,sq2].mean(axis=1) + dt - offset, (np.nanmean(skew_quad2[:,:-skip2b], axis=0))/61.78-1.8-0.5, '.', markersize=2, label="Before Insulation: 03/03", color=gm2.colors[1])
plot_ts(fp_time[skip:,sq].mean(axis=1) - offset,             (np.nanmean(skew_quad, axis=0) - np.nanmean(skew_quad))/61.78, '.', markersize=2, label="With Insulation: 10/12", color=gm2.colors[0])
plt.ylabel("skew quadrupole variations \n [ppm @ 4.5cm]")
plt.xlabel("date")
plt.gca().xaxis.set_major_formatter(formatter)
from matplotlib.ticker import MaxNLocator
plt.gca().xaxis.set_major_locator(MaxNLocator(8))
plt.legend()
gm2.despine()
plt.show()

'''
test = 0
for yoke in np.arange(ord('A'), ord('L')+1):
    for aziId in np.arange(1,6+1):
        print("Yoke "+chr(yoke)," : "+str(aziId))
        if test:
            break
        s_station = selectStation(yoke, aziId)

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
        #test = 1
'''


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
