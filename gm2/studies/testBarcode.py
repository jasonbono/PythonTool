import numpy as np
from scipy.interpolate import interp1d

from util import *
import matplotlib.pyplot as plt
import seaborn as sns

trolleyRun = [1, 1]

# settings
plot = True
plotError = True
onlyCalib = True
show = True
plotAll = True
plot = False
plotError = False
onlyCalib = False
show = False
plotAll = False

if trolleyRun == [0, 0]:
    runs = [5216, 5217]
    phisCalib = np.arange(np.pi+0, np.pi*1.5-0.1, 0.1)
    #                     0      1      2      3      4      5      6      7      8      9     10     11     12     13     14
    starts=np.array([[20000, 60000, 40000, 20000, 40000, 80000, 40000, 61000, 60000, 20000, 60000, 30000, 20000, 40000, 60000],
                     [20000, 20000, 20000,  8000, 20000, 20000, 20000,  1600, 20000,     0, 20000, 10000, 10000, 20000, 30000]])
if trolleyRun == [0, 1]:

    runs = [5218, 5217]
    phisCalib = np.arange((np.pi*(-0.5)) - 0.1, np.pi*1.0 - 0.1, 0.1)
    starts = np.array([[20, 20], #  0
                       [53, 53], #  1
                       [40,  5], #  2
                       [65, 65], #  3
                       [41, 41], #  4
                       [ 2, 85], #  5
                       [45, 45], #  6
                       [16, 16], #  7
                       [50, 50], #  8
                       [50, 20], #  9
                       [28,  4], # 10
                       [43, 25], # 11
                       [20,  0], # 12
                       [60, 20], # 13
                       [15, 15], # 14
                       [40, 20], # 15
                       [75, 25], # 16
                       [60, 20], # 17
                       [20, 20], # 18
                       [50, 50], # 19
                       [35, 35], # 20
                       [57, 57], # 21
                       [50, 20], # 22
                       [80,  0], # 23
                       [60, 15], # 24
                       [40, 15], # 25
                       [82, 51], # 26
                       [40, 15], # 27
                       [60, 20], # 28
                       [ 0, 100], # 29
                       [20,  0], # 30
                       [-1, -1], # 31 broken
                       [-1, -1], # 32 broken
                       [40, 15], # 33
                       [20,  0], # 34
                       [50,  0], # 35
                       [80, 60], # 36
                       [-1, -1], # 37 broken
                       [40, 15], # 38
                       [60, 40], # 39
                       [40, 10], # 40
                       [70, 20], # 41
                       [40,  0], # 42
                       [20,  0], # 43
                       [60,  0], # 44
                       [40,  0], # 45
                       [60, 20], # 46
                       [40,  0] # 47           
                       ], dtype='int').T * 1000# + 51000

if trolleyRun == [1, 0]:
    runs = [3996, 3997]
    phisCalib = np.arange(np.pi+0, np.pi*1.5-0.1, 0.1)
    starts = np.array([[30, 30], #  0
                       [61, 61], #  1
                       [40,  5], #  2
                       [40,  2], #  3
                       [50, 20], #  4
                       [20,  0], #  5
                       [60, 20], #  6
                       [40,  0], #  7
                       [60, 20], #  8
                       [40,  0], #  9
                       [60, 30], # 10
                       [40,  0], # 11
                       [50,  0], # 12
                       [50, 20], # 13
                       ], dtype='int').T * 1000# + 51000

if trolleyRun == [1, 1]:
    runs = [3998, 3997]
    phisCalib = np.arange((np.pi*(-0.5)) - 0.1, np.pi*1.0 - 0.1, 0.1)
    starts = np.array([[28, 28], #  0
                       [20,  0], #  1
                       [50, 20], #  2
                       [30,  0], #  3
                       [41, 41], #  4
                       [30,  0], #  5
                       [60, 40], #  6
                       [40, 20], #  7
                       [50, 20], #  8
                       [50, 20], #  9
                       [50,  0], # 10
                       [50,  0], # 11
                       [20,  0], # 12
                       [60, 20], # 13
                       [40,  0], # 14
                       [40, 20], # 15
                       [30,  0], # 16
                       [60, 20], # 17
                       [40,  0], # 18
                       [20,  0], # 19
                       [40,  0], # 20
                       [60, 50], # 21
                       [60, 50], # 22
                       [30,  0], # 23
                       [60, 40], # 24
                       [40, 15], # 25
                       [40,  0], # 26
                       [40, 15], # 27
                       [60, 40], # 28
                       [40,  0], # 29
                       [20,  0], # 30
                       [-1, -1], # 31 broken
                       [-1, -1], # 32 broken
                       [-1, -1], # 33
                       [80,  0], # 34
                       [50,  0], # 35
                       [40,  0], # 36
                       [-1, -1], # 37 broken
                       [40, 15], # 38
                       [60, 40], # 39
                       [40, 10], # 40
                       [70, 20], # 41
                       [40,  0], # 42
                       [20,  0], # 43
                       [60, 10], # 44
                       [40,  0], # 45
                       [60, 30], # 46
                       [40,  0] # 47           
                       ], dtype='int').T * 1000# + 51000

riseTimeIndex = 500
probe_unc = 5e-6 * 61.79e3


start_offset = 2000 # 1 run -> 5100
startCalib = 0
n = 20 # 40
n_short = 10 # used for barcode runs (not calibration)
nn = 2000
offset = [8, -8]


def getBarcode(run, start, nframes):
    tr = np.zeros([0])
    ts = np.zeros([0])
    for ev in np.arange(start, start+nframes):
        tr = np.concatenate((tr, trs[run][ev]), axis=0)
        ts = np.concatenate((ts, trs_ts[run][ev]), axis=0)
    s = (tr>1.0)&(tr<5.0)
    return tr[s], ts[s]

import os
fname = "tmp/input_"+"%05i" % runs[0]+"-"+"%05i" % runs[1]+'.npz'
if os.path.isfile(fname):
    data_ = np.load(fname)
    phi     = data_['phi']
    time    = data_['time']
    timeNMR = data_['timeNMR']
    freqAll = data_['freqAll']
    trs     = data_['trs']
    trs_ts  = data_['trs_ts']
else:
    import Trolley
    t = []
    phi = [[],[]]  # phi
    time = [[],[]] # Associated Barcode Time
    timeNMR = [[],[]]
    freqAll = [[],[]]
    trs     = [[],[]]
    trs_ts  = [[],[]]
    for i, r in enumerate(runs):
        t.append(Trolley.Trolley([r]))

        t[i].activateBranches(["Position", 
                               "TimeStamp",
                               "ProbeFrequency",
                               "Barcode"])
        #t[i].activateBranches(["Position_Phi", 
        #                       "TimeStamp_BarcodeStart",
        #                       "TimeStamp_NMRStart",
        #                       "ProbeFrequency_Frequency",
        #                       "Barcode_traces"])
        def callback():
            #tr_l = t[i].getBarcodeTrace()[0,:]
            #tr_ts_l = t[i].getBarcodeTs()
            #s = (tr_l>1.0)&(tr_l<5.0)
            #trs[i] = np.concatenate((trs[i], tr_l[s]))
            #trs_ts[i] = np.concatenate((trs_ts[i], tr_ts_l[s]))
            return [t[i].getPhis(), t[i].getTimeBarcodes(), t[i].getTimeNMRs(), t[i].getFrequencys(True), t[i].getBarcodeTrace()[0,:], t[i].getBarcodeTs()]
        phi[i], time[i], timeNMR[i], freqAll[i], trs[i], trs_ts[i] = t[i].loop(callback)


        np.savez_compressed("tmp/input_"+"%05i" % runs[0]+"-"+"%05i" % runs[1]+'.npz',phi=phi, time=time, timeNMR=timeNMR,freqAll=freqAll, trs=trs, trs_ts=trs_ts)

    #s = (trs[i][0,:]>1.0)&(trs[i][0,:]<5.0) 
    #trs[i]    = trs[i][:,s]
    #trs_ts[i] = trs_ts[i][s]

probe = 0

# chose a run
freqs     = np.zeros([nn, 17, 2])
freqs_unc = np.zeros([nn, 17, 2, 2])
phis      = np.zeros([nn, 17, 2])
times     = np.zeros([nn, 17, 2])

#start = [20000, 20000]
start = np.array([20000, 20000])
run = np.array([0, 0])

phi_no = startCalib
forceCalib = True
isCalib = False
for ii in range(nn):
    print ii,
    #run   = [] # run/event number within root file
    tr    = []  # barcode trace of n runs/events
    tr_ts = []  # timestamp associated to barcode trace

    if (forceCalib)|(phi[0][run[0]+offset[0],probe,0] >  phisCalib[phi_no]):
        if phi_no == len(phisCalib) - 1:
            break
        forceCalib = False
        isCalib = True
        print "Callibration ", phi_no ," at ", phisCalib[phi_no], phisCalib[phi_no]/np.pi*180,
        for i in range(len(runs)):
            run[i] = nearestIndex(phi[i][:, probe, 0], phisCalib[phi_no])
        #start = starts[:,phi_no]
        phi_no += 1
    else:
        for i in range(len(runs)):
            run[i] = run[i] + offset[i]
        isCalib = False
        #start = np.array([300*17/2 * 4, 300*17/2 * 4])
        if onlyCalib:
            continue

    for i, r in enumerate(runs):
        phis[ii,:,i] = phi[i][run[i],:,0]
        tr.append([])
        tr_ts.append([])    
        if isCalib:
            start = starts[:,phi_no-1]
            tr[i], tr_ts[i] = getBarcode(i,run[i]-n/2, n)
        else:
            start[i] = start_offset # len(tr[i])/2
            if i in [0]:
                tr[i], tr_ts[i] = getBarcode(i,run[i], n_short)
            else:
                tr[i], tr_ts[i] = getBarcode(i,run[i]-n_short, n_short)
        #tr[i], tr_ts[i] = t[i].getBarcode(run[i]-n/2, n)
    tr[1]    = np.flip(tr[1])
    tr_ts[1] = np.flip(tr_ts[1])
    print "run", str(run[0])+"/"+str(run[1]), "\tphi: ", "%1.4f" % phis[ii,0,0], "%3.2f" % (phis[ii,0,0]/np.pi*180),

    # align abs markers
    thr = [tr[0].min()/2.0 + tr[0].max()/2.0, tr[1].min()/2.0 + tr[1].max()/2.0]

    index = []
    for i, r in enumerate(runs):
        if start[i] in [-1]:
            forceCalib = True
            continue
        try:
            index.append(firstIndexBelowThr(tr[i][:], thr[i], start[i]))
        except:
            index.append(0)
            print "Problem in firstIndexBelowThr, file ", r, "("+str(i)+")"
            forceCalib = True
            continue

    if plot:
        # Plot the barcode at the moment
        import matplotlib.pyplot as plt
        import seaborn as sns

        #print phi_no-1, ")", ii, ")", start
        f, ax = plt.subplots(2)
        ax[0].plot(tr[0][:], '-', color=sns.color_palette()[1], label="CW")
        ax[0].plot(tr[1][:], '-', color=sns.color_palette()[2], label="CCW")
        ax[1].plot(np.arange(len(tr[0][:])), tr[0][:], '-', color=sns.color_palette()[1])
        ax[1].plot(np.arange(len(tr[1][:])) + (index[0]-index[1]), tr[1][:], '-', color=sns.color_palette()[2])
        ax[1].set_xlabel("barcode sampling point")
        ax[0].set_ylabel("barcode raw reading")
        ax[1].set_ylabel("barcode raw reading")
        sns.despine()
        ax[0].legend()
        #plt.show()
        f.savefig("tmp/calib_"+"%05i" % runs[0]+"-"+"%05i" % runs[1]+"_no"+str(phi_no-1)+".pdf")
        if show:
            plt.show()
        #plt.clf()
        plt.close(f)
        #raw_input()
        #ax[0].clear()
        #ax[1].clear()


    ## Find nearest frame for each probe
    freq = []
    freq_unc = []
    for i, r in enumerate(runs):
        freq.append(np.array([0.0] * 17))
        freq_unc.append(np.zeros([17,2]))
        for probe in np.arange(17):
            try:
                runAligned = nearestIndex(time[i][:,probe], tr_ts[i][index[i]])
            except:
                if plotError:
                    import matplotlib.pyplot as plt
                    import seaborn as sns

                    f, ax = plt.subplots(2)
                    ax[0].plot(tr[0][:], '-')
                    ax[0].plot(tr[1][:], '-')
                    ax[1].plot(np.arange(len(tr[0][:])), tr[0][:], '-')
                    ax[1].plot(np.arange(len(tr[1][:])) + (index[0]-index[1]), tr[1][:], '-')
                    sns.despinei()
                    plt.savefig("tmp/error_"+"%05i" % runs[0]+"-"+"%05i" % runs[1]+"_no"+str(phi_no-1)+"-"+str(ii)+".pdf")
                    plt.close(f)

                print "Problem in nearestIndex, file ", r, "("+str(i)+"), probe", probe
                forceCalib = True
                break
            run[i] = runAligned
            #freq_ = np.zeros([n])
            #freq_time_ = np.zeros([n])
            freq_ = freqAll[i][runAligned - n/2:runAligned + n/2, probe, 0]
            freq_time_ = timeNMR[i][runAligned - n/2:runAligned + n/2, probe]
            #for j in np.arange(-n/2, n/2):
            #    ev_ = runaligned + j
            #    freq[j + n/2] = freq[ev_][probe]
            #    #t[i].getEntry(runAligned + j)
            #    #freq_[j + n/2] = t[i].getFrequencys()[0][probe, 0]
            #    freq_time_[j + n/2] = t[i].getTimeNMRs()[probe]
            freq_time_s_ = smooth(freq_time_, 7, 'ref')
            freq_s_      = smooth(freq_, 5, 'lr')
            freq_f = interp1d(freq_time_s_, freq_s_, kind='linear')
            freq[i][probe] = freq_f(tr_ts[i][index[i]])
            ttt = np.arange(-3.0e9, 3.0e9,0.1e9)
            if plotAll:
                if probe in [0]:
                    if i in [0]:
                        #try:
                        ttt_s = (tr_ts[i][index[i]] + ttt > freq_f.x.min())&(tr_ts[i][index[i]] + ttt < freq_f.x.max())
                        plt.plot(freq_time_   - tr_ts[i][index[i]], freq_, '.--', color=sns.color_palette()[0])
                        plt.plot(ttt[ttt_s], freq_f(tr_ts[i][index[i]] + ttt[ttt_s]), '-',  color=sns.color_palette()[0])
                        #except:
                        #    print "oops"
                    if i in [1]:
                        #try:
                        ttt_s = (tr_ts[i][index[i]] - ttt > freq_f.x.min())&(tr_ts[i][index[i]] - ttt < freq_f.x.max())
                        plt.plot(-freq_time_   + tr_ts[i][index[i]], freq_,  '.--', color=sns.color_palette()[1])
                        plt.plot(ttt, freq_f(tr_ts[i][index[i]] - ttt), '-', color=sns.color_palette()[1])
                        #except:
                        #    print "oops"
                        ymean = (freq[0][probe]+freq[1][probe])/2.
                        plt.ylim(ymean-100, ymean+100)
                        plt.plot([-4e9, 4e9],[ymean, ymean],'--', color='grey')
                        plt.plot([0,0], [ymean-100, ymean+100],'--', color='grey')
                        plt.show()

            try:
                freq_unc_ = freq_f(tr_ts[i][index[i]-riseTimeIndex/2:index[i]+riseTimeIndex/2])
            except:
                freq_unc_ = np.nan
            if len(freq_unc_) > 0:
                freq_unc[i][probe,:] = [freq[i][probe] -  freq_unc_.min(), freq_unc_.max() - freq[i][probe]]
            else:
                freq_unc[i][probe,:] = [np.nan, np.nan]
            times[ii,probe,i] = time[i][run[i], probe] #t[i].getTimeGPSs()
        freqs[ii,:,i] = freq[i]
        freqs_unc[ii,:,:,i] = freq_unc[i] 
        #times[ii,:,i] = timeNMR[i][run[i],:] #t[i].getTimeGPSs()
    print "delta freq ", "%3.1f" % ((freq[0]-freq[1]).mean()), "+/-", "%3.1f" % ((freq[0]-freq[1]).std())
    #print freq[0], freq[1], freq[1]-freq[0], (freq[1]-freq[0])/(freq[1]+freq[0])/2.0

np.savez_compressed("data/correctedFreqs_"+"%05i" % runs[0]+"-"+"%05i" % runs[1]+'.npz',times=times, phis=phis, freqs=freqs, freqs_unc=freqs_unc)


import matplotlib.pyplot as plt
s1 = freqs[:,0,0] > 0
probe = 0
df = freqs[:,:,0]-freqs[:,:,1]
dt = times[:,:,0]-times[:,:,1]
dphi = phis[:,:,0] - phis[:,:,1]
freqs_unc_tot = np.sqrt(freqs_unc[:,:,:,0].max(axis=2)**2 + freqs_unc[:,:,:,1].max(axis=2)**2 + probe_unc**2)
plt.errorbar(phis[s1,probe,0], df[s1,probe], yerr=freqs_unc_tot[s1, probe], xerr=np.abs(dphi[s1, probe])/2., fmt=' ')
#plt.show()
s2 = s1&(np.abs(dphi[:,probe])<0.010)&(freqs_unc_tot[:,0]<2)
plt.errorbar(phis[s2,probe,0], df[s2,probe], yerr=freqs_unc_tot[s2, probe], xerr=np.abs(dphi[s2, probe])/2., fmt=' ')
s3 = s2&(df.std(axis=1)<40)
plt.errorbar(phis[s3,probe,0], df[s3,probe], yerr=freqs_unc_tot[s3, probe], xerr=np.abs(dphi[s3, probe])/2., fmt=' ')
plt.xlabel("phi")
plt.ylabel("frequency difference [Hz]")
plt.show()


ss = s3
plt.errorbar(dt[ss, probe]/1e9/60, df[ss, probe], yerr=freqs_unc_tot[ss, probe], xerr=[5/60.]*len(dt[ss, probe]), fmt =' ')
plt.xlabel("time difference [min]")
plt.ylabel("frequency difference [Hz]")
plt.show()

print "RMS: ", df[s2,0].std(), "/", df[s2,:].std(axis=1), " :: ", df[s2,0].std()/61.8, "/", df[s2,:].std(axis=1)/61.8

plt.hist(df[s2,probe], bins=np.arange(-200,200,2))
plt.xlabel("frequency difference [Hz]")
plt.show()
