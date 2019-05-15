#!/usr/bin/env python

import matplotlib as mpl
mpl.use('Agg')

import gm2


import sys
if len(sys.argv) < 2:
    print "USAGE: fpViewer run_no_start [run_no_end] [1:signals]"
    raise SystemExit


no_start = int(sys.argv[1])
if len(sys.argv) > 2:
    no_end = int(sys.argv[2])
else:
    no_end = no_start

mode = 0
if len(sys.argv) > 3:
    mode = int(sys.argv[3])

ff = gm2.fpViewer(no_start, no_end)
ff.view(gif=True, mode=mode)
'''
#runs = np.arange(5807, 5835+1) #[5807]
runs = np.arange(no_start, no_end+1) #[5807]
fp = gm2.FixedProbe(runs, True)

#runs = np.arange(no_start, no_end+1) #[5807]
#fp = gm2.FixedProbe([], False)
#fp.fname_path    = "TreeGenFixedProbe/fixedProbe_DAQ"
#fp.loadFiles(list(runs))
#
#
#def callback():
#        return [fp.getTimeGPS(), fp.getFrequency(0), fp.getAmplitude(), fp.getPower(), fp.getFidLength()]
#    ##_, fp_time, fp_freq = fp.getBasics()
#fp_time, fp_freq, fp_amp, fp_power, fp_length  = fp.loop(callback)


#def callback():
#    return [fp.getTimeSystem(), fp.getFrequency(0)]
##_, fp_time, fp_freq = fp.getBasics()
#fp_time, fp_freq = fp.loop(callback)

fp_phi = fp.getPhi()

fp_time = fp.time
fp_freq = fp.freq

skip = 1

ylim = 200.
ylim2 = 50.

def selectStation(yoke, azi):
    return (fp.id['yoke'] == yoke)&(fp.id['azi'] == azi)

def selectProbe(rad, layer):
    return (fp.id['rad'] == rad)&(fp.id['layer'] == layer)

layers = ["T", "B"]
radIds = ["O", "M", "I"]

alpha = 1.0
figsize = [gm2.plt.rcParams['figure.figsize'][1] * 2.0, gm2.plt.rcParams['figure.figsize'][1] * 1.5]
from matplotlib.dates import DateFormatter
formatter = DateFormatter('%m/%d\n%H:%M')

# create folder for pngs
import os
dirname = str(runs[0])+"to"+str(runs[-1])
try:
    os.mkdir("plots/")
except:
    pass
try:
    os.mkdir("plots/"+dirname)
except:
    pass

test = 0
for yoke_ in np.arange(ord('A'), ord('L')+1):
    yoke = chr(yoke_)
    for aziId in np.arange(1,6+1):
        print("Yoke "+yoke," : "+str(aziId))
        if test:
            break
        s_station = selectStation(yoke, aziId)
        #print("Station", np.argwhere(s_station).shape)
        freq_mean = 0
        freq_n    = 0
        f = gm2.plt.figure(figsize=figsize)
        ax1 = gm2.plt.subplot(211)
        for radId in radIds:
            for layer in layers:
                s = selectProbe(radId, layer)&s_station
                #print("Number", np.argwhere(s).shape)
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

        gm2.plt.title("Yoke "+(yoke)+", azi "+str(aziId)+(" at %.0f" % ((phi_[0]+360)%360))  )
        gm2.plt.ylabel(r'$f^{\#} - f_{0}^{\#}$ [Hz]')
        gm2.plt.ylim([-ylim, ylim])

        if freq_n > 0:
            freq_mean /= freq_n
            gm2.plt.subplot(212, sharex=ax1)
            for radId in radIds:
                for layer in layers:
                    s = selectProbe(radId, layer)&s_station
                    probe = np.argwhere(s)
                    if len(probe) == 1:
                        mean = fp_freq[skip:, s].mean()
                        s_t = fp_time[skip:, s] > 0
                        if len(fp_time[skip:, s][s_t]) > 0:
                            plot_ts(fp_time[skip:, s][s_t], fp_freq[skip:, s][s_t]-mean-freq_mean[skip:][s_t], '.', markersize=2, label=label_, alpha=alpha)

        gm2.plt.xlabel("time")
        gm2.plt.ylabel(r'$(f^{\#} - f_{0}^{\#}) - f_{\rm{mean}}^{\rm{station}}$ [Hz]')
        gm2.plt.ylim([-ylim2, ylim2])

        gm2.plt.gca().xaxis.set_major_formatter(formatter)
        #ax.xaxis.set_tick_params(rotation=30, labelsize=10)
        gm2.despine()
        gm2.plt.subplot(211)
        gm2.plt.setp(ax1.get_xticklabels(), visible=False)
        #plt.gca().set_xticklabels([]);
        lgnd = gm2.plt.legend(loc='upper center', bbox_to_anchor=(0.2, -.22, 0.6, 0.2), ncol=3, fontsize=12)
        for lgndHandl in lgnd.legendHandles:
            lgndHandl._legmarker.set_markersize(12)
        f.savefig("plots/"+dirname+"/fp_"+yoke+"_"+str(aziId)+".png")
        #plt.show()
        #test = 1

# combine plots to gif
os.system("convert -delay 100 plots/"+dirname+"/fp_*_*.png -loop 0 plots/fp_"+dirname+".gif")
print("plots/fp_"+dirname+".gif")
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
