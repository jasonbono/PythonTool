# coding: utf-8
import gm2
import numpy as np
import matplotlib.pyplot as plt


def run(runs):
    tr = gm2.Trolley(runs)
    def callback():
        return [tr.getTimeGPS(), tr.getPhi(), tr.getFrequency(), tr.getFidLength(), tr.getAmplitude()]
    time, phi, freq, length, amp = tr.loop(callback)

    skip = 0
    freq_ = freq[skip:,:,0].copy()
    outl = np.full([freq_.shape[0],17], np.nan)
    cor = np.full(freq_.shape, False)

    def outlier(event, th):
        n = 1
        nn = 0
        if(event-n >=0)&(event+n<time.shape[0]):
            for probe in range(17):
                mean = (freq_[event-1, probe] + freq_[event+1, probe])/2.0
                outl[event, probe] = freq[event, probe, 0] - mean
                d_pre  = freq[event, probe, 0] - freq_[event-1, probe]
                d_post = freq[event, probe, 0] - freq_[event+1, probe]
                if ((np.abs(outl[event, probe]) > th)&(np.abs(d_pre) > th)&(np.abs(d_post) > th)&((d_pre * d_post)> 0)):
                    nn += 1
                    freq_[event, probe] = mean
                    cor[event, probe] = True
            if nn > 0:
                nn += outlier(event-1, th)
        return nn

    for event in range(freq.shape[0]):
        outlier(event, 30.0)

    probe = 0
    skip - 6
    print("DEBUG ", outl[skip:-skip, probe][cor[skip:-skip,probe]], (outl[skip:-skip, probe][cor[skip:-skip,probe]]).sum())

    if False:
        probe = 0
        skip = 6
        #plt.errorbar(phi[skip:-skip,probe, 0]/np.pi*180, freq[skip:-skip, probe,0], yerr=outl[skip:-skip, probe], fmt='.')
        #plt.errorbar(phi[skip:-skip,probe, 0][cor[skip:-skip,probe]]/np.pi*180, freq[skip:-skip,probe,0][cor[skip:-skip,probe]], yerr=outl[skip:-skip, probe][cor[skip:-skip,probe]], fmt='.')
        #print((freq[skip:-skip, probe][cor[skip:-skip,probe]==False]).sum()/(freq[skip:-skip, probe][cor[skip:-skip,probe]==False]).shape[0]) 
        #print((freq[skip:-skip, probe]).sum()/(freq[skip:-skip, probe]).shape[0]) 
        #plt.plot(phi[skip:-skip,probe, 0][cor[skip:-skip,probe]]/np.pi*180, freq_[skip:-skip,probe][cor[skip:-skip,probe]], '.', alpha=0.3)
        ss = np.abs(outl[skip:-skip, probe]) > 10000
        print(phi[skip:-skip,probe, 0][ss].shape)
        plt.plot(phi[skip:-skip,probe, 0][ss]/np.pi*180, freq_[skip:-skip,probe][ss], '.')
        plt.xlabel("azimuth [deg]")
        plt.ylabel("frequency [Hz]")
        gm2.despine()
        plt.show()

    if False: # used to generate plots for accelerometer studies
        probe = 0
        skip = 6
        #plt.errorbar(phi[skip:-skip,probe, 0]/np.pi*180, freq[skip:-skip, probe,0], yerr=outl[skip:-skip, probe], fmt='.')
        plt.subplot(211+len(tmp))
        ax = plt.plot(phi[skip:-skip,probe, 0]/np.pi*180, outl[skip:-skip, :], '-', alpha=0.6, color=gm2.sns.color_palette()[len(tmp)],label=ll[len(tmp)])
        plt.gca().set_xlim([90, 180])
        #print((freq[skip:-skip, probe][cor[skip:-skip,probe]==False]).sum()/(freq[skip:-skip, probe][cor[skip:-skip,probe]==False]).shape[0]) 
        #print((freq[skip:-skip, probe]).sum()/(freq[skip:-skip, probe]).shape[0]) 
        #plt.plot(phi[skip:-skip,probe, 0][cor[skip:-skip,probe]]/np.pi*180, freq_[skip:-skip,probe][cor[skip:-skip,probe]], '.', alpha=0.3)
        #ss = np.abs(outl[skip:-skip, probe]) > 10000
        #print(phi[skip:-skip,probe, 0][ss].shape)
        #plt.plot(phi[skip:-skip,probe, 0][ss]/np.pi*180, freq_[skip:-skip,probe][ss], '.')
        if len(tmp) == 1:
            plt.gca().set_xlabel("azimuth [deg]")
            plt.gca().set_ylabel("outlier frequency [Hz]\nCW")
        else:
            plt.gca().set_ylabel("outlier frequency [Hz]\nCCW")
        #plt.gca().legend()
        return ax
        #gm2.sns.despine()
        #plt.show()

    '''


        while np.abs(dmax) > 61*2:
            s = 1
            d = freq_[s:-s, p] - (freq_[:-s*2,p] + freq_[s*2:,p])/2.
            n = np.argmax(np.abs(d)) + s
            dmax = d[n-s]
            freq_[n,p] = (freq_[n-s,p] + freq_[n+s,p])/2.
            cor[n+skip, p] = True
     '''

    #print(outl[skip:-skip,:][np.abs(outl[skip:-skip,:])>10000])
    #outl[np.abs(outl)>10000] = np.nan
    
    if False:
        print(np.nanmin(outl), np.nanmax(outl))
        bins = np.arange(np.nanmin(outl), np.nanmax(outl), (np.nanmax(outl) - np.nanmin(outl))/1000.)
        for probe in range(1):
            plt.hist(outl[:,probe], bins=bins)
        plt.yscale('log', nonposy='clip')
        plt.xlabel("distance [Hz]")
        plt.ylabel("counts")
        gm2.despine()
        plt.show()


    if True: # used to estimate effect of outliers/spikes
        ths = np.concatenate([np.arange(0, 1000, 0.1), np.arange(1000, 10000, 1)]) 
        s = ths.copy()
        nnn = ths.copy()
        skip = 6
        plt.figure(figsize=[6.4, 4.8*1.5])
        for probe in range(17):
            norm = (freq_[skip:-skip, probe] * np.abs(phi[skip+1:-(skip-1), probe, 0] - phi[(skip-1):-(skip+1), probe, 0]) / 2.).sum()
            norm_ = phi[skip:-skip,probe,0].max() - phi[skip:-skip,probe,0].min()
            for i, th in enumerate(ths):
                index =      np.argwhere((np.abs(outl[skip:-skip,probe]))>th)
                #sss_last = (np.abs(outl[skip-1:-(skip+1),probe]))>th
                #sss_next = (np.abs(outl[skip+1:-(skip-1),probe]))>th
                #print(sss.shape, sss_last.shape, sss_next.shape, outl[skip:-skip, probe].shape, phi[skip:-skip, probe,0].shape)
                nnn[i] = index.shape[0]

                s[i] = (outl[skip:-skip, probe][index] * np.abs(phi[:,probe, 0][index+1+skip] - phi[:,probe, 0][index-1+skip]) / 2.).sum()
            plt.subplot(211)
            plt.plot(ths, s/norm_/62.8*1000, '--', label="#%i" % probe)
            plt.gca().set_xlim([0,1000])
            plt.gca().set_ylim([-50,50])
            plt.subplot(212)
            plt.plot(ths, s/norm_/62.8*1000, '--', label="#%i" % probe)
            plt.gca().set_xlim([1000,10000])
            plt.gca().set_ylim([-50,50])

        #print(outl[:,probe][np.abs(outl[:,probe])>th])
        plt.xlabel("outlier threshold [Hz]")
        plt.subplot(211)
        plt.ylabel(r'integrated (trapeze) error [ppb]')
        plt.title("run "+str(runs))
        plt.subplot(212)
        plt.legend(ncol=4, bbox_to_anchor=(0.5, 1.0), loc='upper center', fontsize='xx-small')
        gm2.sns.despine()
        plt.savefig("plots/trolleySpikesIntegration_run"+str(runs[0])+".pdf")
        plt.show()


    sel = np.array([0,1,2,3,4,5,6,7,8,9,11,12,13,14,15,])
    bins = np.arange(-1.7,4.5,0.01)
    v = np.zeros([bins.shape[0]-1, 17])
    for probe in np.arange(17):
        v[:, probe], _, _ = plt.hist(phi[:,probe,0][cor[:,probe]], bins=bins, histtype='stepfilled')
    plt.close('all')
    bc = np.array(0.5*(bins[1:]+bins[:-1]))
    return v, bc

## accelerometer studies ##
'''

#f = plt.figure(figsize=[6.4*1.3, 4.8*1.4])
#tmp = []
#ll = ["CCW","CW"]
tmp.append(run([5260]))
tmp.append(run([5259]))
gm2.sns.despine()
plt.show()
sys.exit()
'''
#run([5260])
#run([5259, 5261])
#run([3614])
run([3613])
run([3997])
run([3997])
run([4058])
sys.exit()


v1, bc1 = run([3997])
v2, bc2 = run([3996, 3998])


plt.step(bc1/np.pi*180,  v1[:,0],where='mid')
plt.step(bc2/np.pi*180, -v2[:,0], where='mid')
plt.xlabel("azimuth [deg]")
plt.xlim([-4,4])
plt.ylabel("#spikes/ %.1f deg" % ((bc1[3]-bc1[2])/np.pi*180))
plt.show()

plt.subplots_adjust(hspace=0.0, wspace=0.1)
for p in np.arange(9):
  for m in np.arange(2):
    i = p*2 + m
    plt.subplot(9,2, 1+i)
    if i < 17:
        plt.step(bc1/np.pi*180,  v1[:,i],where='mid')
        plt.step(bc2/np.pi*180, -v2[:,i], where='mid')
        plt.ylim([-4,4])
#plt.subplot(212)
#plt.step(bc1/np.pi*180,  v1[:,1],where='mid')
#plt.step(bc2/np.pi*180, -v2[:,1], where='mid')

plt.subplot(9,2,9*2-3)
plt.xlabel("azimuth [deg]")
#plt.subplot(9,2,9*2-2)
#plt.xlabel("azimuth [deg]")
plt.subplot(9,2,9*2/2)
plt.ylabel("#spikes/ %.1f deg" % ((bc1[3]-bc1[2])/np.pi*180))
plt.show()


sel = np.array([0,1,2,3,4,5,6,7,8,9,11,12,13,14,15,])
plt.step(bc1/np.pi*180,  v1[:,sel].sum(axis=1), where='mid')
plt.step(bc2/np.pi*180, -v2[:,sel].sum(axis=1), where='mid')
plt.xlabel("azimuth [deg]")
plt.ylabel("#spikes/ %.1f deg" % ((bc1[3]-bc1[2])/np.pi*180))
plt.show()


v1_2, bc1_2 = run([4058])
v2_2, bc2_2 = run([4057, 4059])


plt.subplots_adjust(hspace=0.0, wspace=0.1)
plt.subplot(211)
plt.step(bc1/np.pi*180,  v1[:,sel].sum(axis=1), where='mid')
plt.step(bc2/np.pi*180, -v2[:,sel].sum(axis=1), where='mid')
plt.subplot(212)
plt.step(bc1_2/np.pi*180,  v1_2[:,sel].sum(axis=1), where='mid')
plt.step(bc2_2/np.pi*180, -v2_2[:,sel].sum(axis=1), where='mid')
plt.xlabel("azimuth [deg]")
plt.ylabel("#spikes/ %.1f deg" % ((bc1[3]-bc1[2])/np.pi*180))
plt.show()



'''
for p in range(17):
    plt.plot(phi[:,p,0], freq[:,p,0], '.')
    plt.plot(phi[cor[:,p],p,0], freq[cor[:,p],p,0], '.')
    plt.show()

for p in range(17):
    plt.plot(phi[:,p,0], freq[:,p,0], '.')
    plt.plot(phi[cor[:,p],p,0], freq[cor[:,p],p,0], '.')
plt.show()

for p in range(17):
    plt.plot(phi[cor[:,p],p,0], freq_[cor[:,p],p] - freq[cor[:,p],p,0], '.')
plt.show()
'''

'''
sel = np.array([0,1,2,3,4,5,6,7,8,9,11,12,13,14,15,])
plt.subplot(211)
plt.hist(phi[:,sel,0][cor[:,sel]], bins=np.arange(-1.7,4.5,0.01), histtype='stepfilled')
plt.subplot(212)
plt.hist(phi2[:,sel,0][cor2[:,sel]], bins=np.arange(-1.7,4.5,0.01), histtype='stepfilled')
plt.show()
'''









#d = freq[1:-1,:,0] - (freq[:-2,:,0] + freq[2:,:,0])/2.
#d2 = freq[2:-2,:,0] - (freq[:-4,:,0] + freq[1:-3,:,0] + freq[3:-1,:,0] + freq[4:,:,0] )/4.
#h1 = (phi[2:,:,0] - phi[:-2,:,0])/2.
