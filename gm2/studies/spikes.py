# coding: utf-8
import gm2
import numpy as np
import matplotlib.pyplot as plt


def loadE989(runs):
    tr = gm2.Trolley(runs)
    def callback():
        return [tr.getPhi(0), tr.getFrequency(0)]
    return tr.loop(callback)

def getSpikes(phi, freq, th): #[:,17]
    skip = 0
    freq_ = freq[skip:].copy()
    outl = np.full([freq_.shape[0]], np.nan)
    cor = np.full(freq_.shape, False)

    def outlier(event, th):
        n = 1
        nn = 0
        if(event-n >=0)&(event+n<freq.shape[0]):
            #for probe in range(17):
            mean = (freq_[event-1] + freq_[event+1])/2.0
            dphi  = phi[event + 1] - phi[event - 1]
            dfreq = (freq_[event+1] - freq_[event -1])
            if(abs(dphi) > 0.1e-4):  # use interpolation if dphi is large enough ...
                mean = freq_[event-1] + dfreq * (phi[event] - phi[event-1]) / dphi;
            outl[event] = freq_[event] - mean
            d_pre  = freq_[event] - freq_[event-1]
            d_post = freq_[event] - freq_[event+1]
            if ((np.abs(outl[event]) > th)&(np.abs(d_pre) > th)&(np.abs(d_post) > th)&((d_pre * d_post)> 0)):
                freq_[event] = mean
                nn += outlier(event-1, th)
        return nn

    for event in range(freq.shape[0]):
        outlier(event, th)
    return outl

new_phi, new_freq = loadE989([3997])
old = gm2.E821("/Users/scorrodi/Documents/E821/trlyData/trolleyData_21_3_01.txt")

new_outl = []
old_outl = []
for probe in np.arange(17):
    new_outl.append(getSpikes(new_phi[:, probe], new_freq[:, probe], 30.0))
    old_outl.append(getSpikes(old.getPhi(probe), old.getFrequency(probe), 30.0))


def localTrlyDist(ax, probe, data):
    bins = np.arange(-500,500,5)
    nsigma = 3
    gm2.plotutil.trlyDist(ax, probe, data, bins, nsigma)
    ax.semilogy()

def localTrlyDistLarge(ax, probe, data):
    bins = np.arange(-1000,1000,61.78)
    s = abs(data[probe]) > 61.78
    ax.hist(data[probe][s], bins=bins, histtype='stepfilled')
    #gm2.plotutil.trlyName(ax, probe)
    #ax.semilogy()


def localTrlyDistLargeCombined(ax, probe, data):
    new = data[0]
    old = data[1]
    bins = np.arange(-1000,1000,61.78)
    s_new = abs(new[probe]) > 61.78*3
    ax.hist(new[probe][s_new], bins=bins, histtype='stepfilled')
    s_old = abs(old[probe]) > 61.78*3
    ax.hist(old[probe][s_old], bins=bins, histtype='stepfilled', alpha=0.5)
    #gm2.plotutil.trlyName(ax, probe)
    #ax.semilogy()


f_new = gm2.plotutil.trlyPlot(localTrlyDist, new_outl, hr=500, title="E989: run 3997", xlabel="spikes [Hz]")
gm2.plt.show()
f_old = gm2.plotutil.trlyPlot(localTrlyDist, old_outl, hr=500, title="E821: run 03/21/2001", xlabel="spikes [Hz]")
gm2.plt.show()


f_new = gm2.plotutil.trlyPlot(localTrlyDistLarge, new_outl, hr=1000, title="E989: run 3997", xlabel="spikes [Hz]")
gm2.plt.show()
f_old = gm2.plotutil.trlyPlot(localTrlyDistLarge, old_outl, hr=1000, title="E821: run 03/21/2001", xlabel="spikes [Hz]")
gm2.plt.show()


f_old = gm2.plotutil.trlyPlot(localTrlyDistLargeCombined, [new_outl, old_outl], hr=1000, title="E821: run 03/21/2001", xlabel="spikes [Hz]")
gm2.plt.show()
